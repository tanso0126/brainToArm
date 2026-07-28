"""Regression tests for the browser-facing MuJoCo 3D studio."""

import time
import unittest

import mujoco

from simul.studio import MuJoCoStudio


class MuJoCoStudioTests(unittest.TestCase):
    def setUp(self):
        self.studio = MuJoCoStudio()

    def tearDown(self):
        self.studio.close()

    def _accelerate(self):
        original_step = self.studio._step_seconds

        def no_wall_clock(seconds, *, real_time_scale=0):
            return original_step(seconds, real_time_scale=0)

        self.studio._step_seconds = no_wall_clock

        def fast_wait(seconds):
            if seconds is None:
                while self.studio._running:
                    if self.studio._reject.wait(timeout=0.01):
                        return True
                return self.studio._reject.is_set()
            return self.studio._reject.is_set()

        self.studio._wait_for_veto = fast_wait

    def _wait_complete(self, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.studio.status()
            if not status["running"]:
                return status
            time.sleep(0.01)
        self.fail("studio task did not finish")

    def _wait_status(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.studio.status()
            if predicate(status):
                return status
            time.sleep(0.01)
        self.fail("studio task did not reach the expected state")

    def test_scene_uses_original_arm_physics_rgb_and_fixed_real_base(self):
        status = self.studio.status()
        yaw = mujoco.mj_name2id(
            self.studio._model, mujoco.mjtObj.mjOBJ_JOINT,
            "studio_base_yaw")
        self.assertEqual(yaw, -1)
        self.assertEqual(self.studio._model.nu, 6)
        self.assertTrue(status["physics"])
        self.assertTrue(status["cameraOnlySelection"])
        self.assertEqual(status["workspace"]["baseMode"], "fixed-90")
        self.assertEqual(status["servoDeg"][0], 90.0)
        self.assertEqual(status["workspace"]["yawDeg"], [0.0, 0.0])
        requested = self.studio._pose.copy()
        requested[0] = 180
        self.assertTrue(self.studio._drive(requested, 0.05))
        self.assertEqual(self.studio.status()["servoDeg"][0], 90.0)
        self.assertGreater(len(self.studio.render_jpeg("overview", 640, 360)), 1000)
        self.assertGreater(len(self.studio.render_jpeg("wrist", 320, 180)), 1000)
        self.studio.render_jpeg("overview", 640, 360)
        self.assertEqual(len(self.studio._renderers), 2)

    def test_scene_edit_is_bounded_to_the_reachable_annulus(self):
        object_id = self.studio.status()["objects"][0]["id"]
        status = self.studio.update_object(object_id, {
            "xMm": 900,
            "yMm": 900,
            "sizeMm": 30,
            "color": "#f25a19",
        })
        item = next(value for value in status["objects"] if value["id"] == object_id)
        radius = (item["xMm"] ** 2 + item["yMm"] ** 2) ** 0.5
        self.assertLessEqual(radius, 414.1)
        self.assertLessEqual(item["sizeMm"], 8.0)
        self.assertEqual(item["yMm"], 0.0)

    def test_physical_delivery_and_late_reject_select_the_next_object(self):
        self._accelerate()
        self.studio.start()
        first = self._wait_status(
            lambda status: status["phase"] == "evaluating")
        first_id = first["lastDeliveredId"]
        self.assertIsNotNone(first_id)
        first_item = next(value for value in first["objects"]
                          if value["id"] == first_id)
        self.assertEqual(first_item["status"], "basket")

        self.studio.reject()
        second = self._wait_status(
            lambda status: (
                status["phase"] == "evaluating"
                and status["lastDeliveredId"] != first_id
            ))
        self.assertIn(first_id, second["rejectedIds"])
        self.assertNotEqual(second["lastDeliveredId"], first_id)
        returned = next(value for value in second["objects"]
                        if value["id"] == first_id)
        self.assertEqual(returned["status"], "table")
        error = ((returned["xMm"] - returned["originXmm"]) ** 2
                 + (returned["yMm"] - returned["originYmm"]) ** 2) ** 0.5
        self.assertLess(error, 7.0)
        delivered = next(value for value in second["objects"]
                         if value["id"] == second["lastDeliveredId"])
        self.assertEqual(delivered["status"], "basket")
        basket_error = (
            (delivered["xMm"] - second["basket"]["xMm"]) ** 2
            + (delivered["yMm"] - second["basket"]["yMm"]) ** 2
        ) ** 0.5
        self.assertLess(basket_error, 7.0)

    def test_delivery_waits_for_rejection_or_explicit_stop(self):
        self._accelerate()
        waits = []

        def record_window(seconds):
            waits.append(seconds)
            if seconds is None:
                self.studio._running = False
                self.studio._phase = "paused"
            return self.studio._reject.is_set()

        self.studio._wait_for_veto = record_window
        self.studio.start()
        status = self._wait_complete()
        self.assertIsNone(status["postDeliveryReviewSeconds"])
        self.assertEqual(status["postDeliveryReviewMode"], "until-stopped")
        self.assertEqual(waits, [1.6, None])


if __name__ == "__main__":
    unittest.main(verbosity=2)
