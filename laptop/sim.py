"""Tiny toy world so the mock camera + arm are self-consistent and the full
pick-and-place loop (including visual servoing and grasp verification) runs and
CONVERGES without hardware.

Models the one thing that matters for cheap cameras: systematic error. The arm
tip does NOT land exactly where commanded — a fixed BIAS (imperfect IK
calibration + imperfect camera->world homography) plus small per-frame noise.
Visual servoing cancels that bias by closing the loop: a phone/laptop camera is
good enough because the loop corrects, not the optics.

Also tracks which objects have been picked, so the mock scene shrinks as the arm
removes objects and grasp-verification ("did it leave its spot?") returns true.
"""
import random

# Fixed systematic offset (cm) between commanded and actual tip position, as a
# cheap camera + rough IK would produce. Servoing removes it.
BIAS = (2.5, -1.8)
NOISE_CM = 0.25          # per-observation camera jitter


class World:
    def __init__(self, objects):
        self._all = list(objects)           # (label, x, y, meta)
        self.picked = set()                 # indices removed from the table
        self._commanded = (0.0, 0.0)        # last workspace target commanded
        self._holding = None                # index currently in the gripper

    @property
    def objects(self):
        return [o for i, o in enumerate(self._all) if i not in self.picked]

    def index_of(self, label, x, y):
        for i, (lb, ox, oy, _) in enumerate(self._all):
            if i not in self.picked and lb == label and abs(ox - x) < 1e-6 and abs(oy - y) < 1e-6:
                return i
        return None

    def set_command(self, xy):
        self._commanded = xy

    def grasp(self, idx):
        self._holding = idx

    def lifted(self):
        """Mark the held object as removed from the table (picked up)."""
        if self._holding is not None:
            self.picked.add(self._holding)

    def release(self):
        self._holding = None

    def tip(self):
        cx, cy = self._commanded
        return (cx + BIAS[0] + random.gauss(0, NOISE_CM),
                cy + BIAS[1] + random.gauss(0, NOISE_CM))


# the "big nail vs small nail" ambiguity — AI can't tell which the human wants
DEFAULT_OBJECTS = [
    ("nail_big",   12.0, 4.0,  {"size": "big"}),
    ("nail_small", 8.0,  -3.0, {"size": "small"}),
    ("screw",      -6.0, 9.0,  {"size": "small"}),
]

WORLD = World(DEFAULT_OBJECTS)
