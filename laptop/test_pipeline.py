"""Hardware-free tests for the parts that must be exactly right.

Run: python test_pipeline.py
Exercises the real code paths (no mocks of the units under test):
  - LXSDF encode -> parse roundtrip, channel auto-detect, resync after garbage,
    packet-count drop detection.
  - IK produces in-range servo commands and points the base toward the target.
  - ErrP detector separates a synthetic error epoch from a clean epoch.
"""
import math
import config
from lxsdf import LXSDFParser, build_packet
import kinematics
from errp import ErrPDetector
from eeg_bridge import EEGBridge


def check(cond, msg):
    print(("  ok  " if cond else " FAIL ") + msg)
    assert cond, msg


def test_lxsdf_roundtrip():
    print("[lxsdf] roundtrip + autodetect")
    chans = [1000, 1500, 2000, 2500, 3000, 500, 100, 4000,
             2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]  # 16 slots
    p = LXSDFParser(total_channels=None)
    stream = b"".join(build_packet(chans, pc=i & 0xFF) for i in range(5))
    out = []
    # feed in awkward chunk sizes to test buffering
    for i in range(0, len(stream), 7):
        out += p.feed(stream[i:i + 7])
    check(p.total_channels == 16, f"autodetected 16 channels (got {p.total_channels})")
    check(len(out) >= 4, f"parsed {len(out)} packets")
    got, pc = out[0]
    check(got[:8] == chans[:8], "channel values recovered exactly")


def test_lxsdf_resync():
    print("[lxsdf] resync after garbage")
    chans = list(range(0, 32, 2))       # 16 slots
    p = LXSDFParser(total_channels=16)
    good = build_packet(chans, pc=0)
    stream = b"\x12\x34\x99" + good + build_packet(chans, pc=1)
    out = p.feed(stream)
    check(len(out) >= 1, f"recovered {len(out)} packets past leading garbage")
    check(out[0][0][:4] == chans[:4], "values correct after resync")


def test_lxsdf_drops():
    print("[lxsdf] packet-drop detection")
    chans = [2048] * 16
    p = LXSDFParser(total_channels=16)
    p.feed(build_packet(chans, pc=10))
    p.feed(build_packet(chans, pc=13))   # skipped 11,12 -> 2 missing
    check(p.dropped == 2, f"counted 2 dropped (got {p.dropped})")


def test_ik():
    print("[ik] servo range + base aim")
    for x, y in [(12, 4), (8, -3), (-6, 9), (0, 15)]:
        a = kinematics.solve(x, y)
        check(len(a) == 7, "7 servo values")
        check(all(config.SERVO_MIN[i] <= a[i] <= config.SERVO_MAX[i] for i in range(7)),
              f"all servos in range for ({x},{y}) -> {a}")
    # base yaw should differ between a +x and a -x target
    ax = kinematics.solve(15, 0)[config.J_BASE]
    anx = kinematics.solve(-15, 0)[config.J_BASE]
    check(ax != anx, "base yaw responds to target direction")


def _synth_epoch(fs, error=False):
    n = int((config.ERRP_BASELINE_S + config.ERRP_WINDOW_S) * fs)
    win = []
    for k in range(n):
        t = k / fs
        row = []
        for ch in range(config.EEG_CHANNELS):
            v = 5 * math.sin(2 * math.pi * 10 * t)
            if error and ch in config.ERRP_FRONTOCENTRAL and 0.25 < t < 0.5:
                v -= 40    # negative deflection ~ErrP
            row.append(v)
        win.append(row)
    return win


def test_errp():
    print("[errp] separates error vs clean epoch")
    det = ErrPDetector(backend="baseline")
    det.update_baseline(_synth_epoch(config.EEG_FS, error=False))
    p_ok = det.p_error(_synth_epoch(config.EEG_FS, error=False))
    p_err = det.p_error(_synth_epoch(config.EEG_FS, error=True))
    print(f"       P(ok)={p_ok:.2f}  P(err)={p_err:.2f}")
    check(p_err > p_ok, "error epoch scores higher than clean")


def test_eeg_packet_timestamps():
    print("[eeg] batched packets receive distinct sample timestamps")
    eeg = EEGBridge()
    chans = [config.ADC_ZERO] * 16
    eeg._emit_from_bytes(b"".join(build_packet(chans, pc=i) for i in range(8)))
    entries = eeg.ring.entries()
    times = [t for t, _ in entries]
    check(len(times) == 8, "all packets entered the ring")
    check(all(a < b for a, b in zip(times, times[1:])), "timestamps are strictly increasing")
    span = times[-1] - times[0]
    expected = 7 / config.EEG_FS
    check(abs(span - expected) < 1e-6, "batch timing follows configured sampling rate")


def test_pick_place():
    print("[pickplace] grasp verify + delivery removes object from table")
    import sim
    from arm_serial import ArmSerial
    from policy import Policy
    from vision import Vision, Detection
    import orchestrator as orch

    sim.WORLD = sim.World(sim.DEFAULT_OBJECTS)   # fresh world
    arm = ArmSerial()                            # mock (no board)
    policy = Policy()
    vision = Vision(mock=True)
    target = Detection(1, "nail_small", 8.0, -3.0, {"size": "small"})

    n_before = len(sim.WORLD.objects)
    ok = orch.grasp_object(arm, vision, policy, target)
    check(ok, "grasp verified in mock")
    orch.place_object(arm, policy, config.PLACE_LOCATION)
    check(len(sim.WORLD.objects) == n_before - 1, "object removed from table after delivery")
    check(sim.WORLD._holding is None, "gripper released after place")
    arm.close()


if __name__ == "__main__":
    test_lxsdf_roundtrip()
    test_lxsdf_resync()
    test_lxsdf_drops()
    test_ik()
    test_errp()
    test_eeg_packet_timestamps()
    test_pick_place()
    print("\nALL TESTS PASSED")
