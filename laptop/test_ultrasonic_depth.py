import math
import unittest

import numpy as np

import arm_fk
from ultrasonic_depth import (
    FloorCalibrationSample,
    SonarMount,
    fit_floor_mount,
    robust_profile,
)


class RangeProfileTests(unittest.TestCase):
    def test_densest_echo_cluster_rejects_multipath(self):
        profile = robust_profile([
            76, 77, 77, 78, 76, 77, 78, 77, 76, 77,
            152, 293, 310, None, 77,
        ])
        self.assertTrue(profile.stable)
        self.assertEqual(profile.distance_mm, 77.0)
        self.assertGreaterEqual(profile.support_fraction, 0.75)

    def test_two_competing_reflectors_are_not_stable(self):
        profile = robust_profile([
            64, 64, 65, 64, 77, 78, 77, 78, 95, 94, 95, 94,
        ])
        self.assertFalse(profile.stable)
        self.assertIn("dominant cluster", profile.reason)

    def test_timeouts_cannot_form_confident_depth(self):
        profile = robust_profile(
            [None] * 8 + [81, 82, 81, 82, 81, 82, 81])
        self.assertFalse(profile.stable)
        self.assertIn("valid echoes", profile.reason)


class SonarGeometryTests(unittest.TestCase):
    def test_motor4_changes_sensor_ray_but_motor6_does_not(self):
        mount = SonarMount(0.018, 0.052, 24.0)
        pose_a = [90, 100, 100, 160, 90, 170]
        pose_b = [90, 100, 100, 180, 90, 170]
        pose_roll = [90, 100, 100, 160, 180, 30]
        origin_a, direction_a = mount.ray(pose_a)
        origin_b, direction_b = mount.ray(pose_b)
        origin_roll, direction_roll = mount.ray(pose_roll)
        self.assertGreater(np.linalg.norm(origin_a - origin_b), 1e-3)
        self.assertGreater(np.linalg.norm(direction_a - direction_b), 0.1)
        np.testing.assert_allclose(origin_a, origin_roll, atol=1e-12)
        np.testing.assert_allclose(direction_a, direction_roll, atol=1e-12)

    def test_fit_recovers_synthetic_forearm_mount_with_outlier(self):
        actual = SonarMount(0.021, 0.049, 27.0)
        poses = [
            [90, 98, 80, 180, 90, 170],
            [90, 105, 90, 180, 90, 170],
            [90, 112, 100, 180, 90, 170],
            [90, 119, 110, 180, 90, 170],
            [90, 126, 120, 180, 90, 170],
            [90, 133, 130, 180, 90, 170],
        ]
        samples = []
        for index, pose in enumerate(poses):
            distance = actual.plane_range_mm(pose)
            self.assertIsNotNone(distance)
            if index == 2:
                distance += 13.0
            samples.append(FloorCalibrationSample(tuple(pose), distance))
        result = fit_floor_mount(samples)
        fitted = result["mount"]
        self.assertLess(result["rms_mm"], 6.0)
        self.assertLess(abs(fitted.origin_x_m - actual.origin_x_m), 0.018)
        self.assertLess(abs(fitted.origin_z_m - actual.origin_z_m), 0.018)
        self.assertLess(abs(fitted.beam_pitch_deg - actual.beam_pitch_deg), 4.0)

    def test_fit_requires_pose_angle_span(self):
        samples = [
            FloorCalibrationSample(
                (90, 124 + index, 90, 180, 90, 170), 80 + index)
            for index in range(4)
        ]
        with self.assertRaisesRegex(ValueError, "angle span"):
            fit_floor_mount(samples)


if __name__ == "__main__":
    unittest.main()
