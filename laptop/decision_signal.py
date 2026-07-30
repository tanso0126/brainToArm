"""Cross-process decision channel for the real-arm target selector.

A keyboard test, dashboard button, or future ErrP callback can emit the same
decision without being coupled to the robot controller:

    python3 laptop/decision_signal.py reject
    python3 laptop/decision_signal.py accept
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_PATH = ROOT / "data" / "control" / "target_decision.json"
VALID_DECISIONS = frozenset(("accept", "reject"))


@dataclass(frozen=True)
class TargetDecision:
    sequence: int
    decision: str
    source: str
    created_at: float


class DecisionMailbox:
    """Atomic, sequence-numbered target decision mailbox."""

    def __init__(self, path=DEFAULT_SIGNAL_PATH):
        self.path = Path(path)

    def read(self):
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            decision = str(payload["decision"]).lower()
            if decision not in VALID_DECISIONS:
                return None
            return TargetDecision(
                sequence=int(payload["sequence"]),
                decision=decision,
                source=str(payload.get("source", "external")),
                created_at=float(payload["createdAt"]),
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError,
                json.JSONDecodeError):
            return None

    def emit(self, decision, source="manual"):
        decision = str(decision).lower()
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"decision must be one of {sorted(VALID_DECISIONS)}")
        current = self.read()
        sequence = 1 if current is None else current.sequence + 1
        payload = {
            "sequence": sequence,
            "decision": decision,
            "source": str(source),
            "createdAt": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)
        return TargetDecision(
            sequence=sequence,
            decision=decision,
            source=str(source),
            created_at=float(payload["createdAt"]),
        )

    def cursor(self):
        current = self.read()
        return 0 if current is None else current.sequence

    def wait_after(self, sequence, timeout_s, poll_s=0.05):
        """Return only a decision newer than ``sequence``."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            current = self.read()
            if current is not None and current.sequence > int(sequence):
                return current
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(float(poll_s), max(0.0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", choices=sorted(VALID_DECISIONS))
    parser.add_argument("--source", default="manual")
    parser.add_argument("--path", type=Path, default=DEFAULT_SIGNAL_PATH)
    args = parser.parse_args()
    emitted = DecisionMailbox(args.path).emit(
        args.decision, source=args.source)
    print(
        f"[decision] seq={emitted.sequence} "
        f"{emitted.decision.upper()} source={emitted.source}")


if __name__ == "__main__":
    main()
