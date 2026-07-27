import os
import unittest

from wrist_publish import _respawn_after_stall
from wrist_vision import NamedAVFoundationCamera


class FakeCamera:
    def __init__(self, reads):
        self.reads = list(reads)
        self.timeouts = []
        self.released = False

    def read(self, timeout_s=None):
        self.timeouts.append(timeout_s)
        return self.reads.pop(0)

    def release(self):
        self.released = True


class WristPublishTests(unittest.TestCase):
    def test_named_pipe_read_times_out_without_a_frame(self):
        read_fd, write_fd = os.pipe()

        class FakeProcess:
            def __init__(self):
                self.stdout = os.fdopen(read_fd, "rb", buffering=0)

            @staticmethod
            def poll():
                return None

        camera = NamedAVFoundationCamera.__new__(NamedAVFoundationCamera)
        camera.width = 2
        camera.height = 2
        camera.process = FakeProcess()
        try:
            self.assertEqual(camera.read(timeout_s=0.01), (False, None))
        finally:
            camera.process.stdout.close()
            os.close(write_fd)

    def test_stall_releases_and_returns_warmed_replacement(self):
        stalled = FakeCamera([])
        replacement = FakeCamera([
            (True, object()),
            (True, object()),
        ])

        result = _respawn_after_stall(
            stalled, camera_factory=lambda: replacement, warmup_frames=2)

        self.assertIs(result, replacement)
        self.assertTrue(stalled.released)
        self.assertEqual(replacement.timeouts, [3.0, 3.0])
        self.assertFalse(replacement.released)

    def test_failed_respawn_releases_child_and_raises(self):
        stalled = FakeCamera([])
        replacement = FakeCamera([(False, None)])

        with self.assertRaisesRegex(RuntimeError, "respawn failed"):
            _respawn_after_stall(
                stalled, camera_factory=lambda: replacement, warmup_frames=1)

        self.assertTrue(stalled.released)
        self.assertTrue(replacement.released)


if __name__ == "__main__":
    unittest.main()
