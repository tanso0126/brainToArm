"""Decides WHICH object to go for and HOW to move the arm there.

Two responsibilities, kept separate on purpose:

  1. Target selection (the part EEG corrects):
     rank candidate objects by preference. Starts naive (nearest first), which is
     exactly why it'll sometimes pick wrong — and the human's ErrP then bumps that
     object down the ranking. reject() is the shared-autonomy hook.

  2. Reaching (the part RL / IK owns):
     turn a target (x,y) into joint angles via inverse kinematics. Swap in a
     trained RL reacher later without touching the selection/EEG logic.

Rejection is by POSITION, not by detection id: markerless detectors renumber
objects every frame (obj0/obj1 shuffle as objects are removed), so a transient
id can't identify "the one the human vetoed". A vetoed workspace location stays
vetoed no matter how the ids get reassigned.
"""
import math
import config

import kinematics

REJECT_RADIUS_CM = 3.0     # a detection within this of a vetoed spot = same object


class Policy:
    def __init__(self):
        self.rejected_pts = []     # workspace (x,y) the human vetoed this round
        self.preference = {}       # rounded-position -> learned score bump
        self.unreachable = []      # most recent detections filtered for geometry

    def reset_trial(self):
        self.reset_selection()

    def reset_selection(self):
        """Forget vetoes after one object is accepted; the next goal may differ."""
        self.rejected_pts = []
        self.unreachable = []

    @staticmethod
    def _key(x, y):
        return (round(x), round(y))

    def _is_rejected(self, det):
        return any(math.hypot(det.x - rx, det.y - ry) <= REJECT_RADIUS_CM
                   for rx, ry in self.rejected_pts)

    # ---- target selection ----
    def score(self, det, arm_xy):
        # lower = more preferred. Naive prior: nearest object first.
        d = math.hypot(det.x - arm_xy[0], det.y - arm_xy[1])
        return d - self.preference.get(self._key(det.x, det.y), 0.0)

    def choose(self, detections, arm_xy):
        self.unreachable = [d for d in detections if not self._reachable_for_pick(d)]
        live = [d for d in detections
                if not self._is_rejected(d) and d not in self.unreachable]
        if not live:
            return None
        return min(live, key=lambda d: self.score(d, arm_xy))

    @staticmethod
    def _reachable_for_pick(det):
        return all(kinematics.reachable(det.x, det.y, z) for z in (
            config.Z_APPROACH, config.Z_GRASP, config.Z_LIFT))

    def reject(self, det, learn=None):
        """Human ErrP said this target is wrong."""
        learn = config.POLICY_SPATIAL_LEARNING if learn is None else learn
        self.rejected_pts.append((det.x, det.y))
        if learn:
            k = self._key(det.x, det.y)
            self.preference[k] = self.preference.get(k, 0.0) - 3.0

    def confirm(self, det, learn=None):
        """Placed with no veto -> reinforce this choice."""
        learn = config.POLICY_SPATIAL_LEARNING if learn is None else learn
        if learn:
            k = self._key(det.x, det.y)
            self.preference[k] = self.preference.get(k, 0.0) + 1.0

    # ---- reaching (IK now; RL-ready) ----
    def target_to_angles(self, target_xy, z=0.0):
        """Map a workspace point to 6 servo commands via inverse kinematics.
        Replace body with an RL policy later; keep the 6-value return shape."""
        x, y = target_xy
        return kinematics.solve(x, y, z)
