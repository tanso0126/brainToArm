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
from polyg_hid import (
    ADC_VOLTS_PER_COUNT, PolyGIHID, command_report, counts_to_adc_mv, decode_report)
from eeg_dashboard import EEGSignalProcessor, analyze_signal_quality, sanitize_recording_name


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


def test_lxsdf_rejects_invalid_shapes():
    print("[lxsdf] rejects corrupt channel counts and lossy sample values")
    try:
        LXSDFParser(total_channels=0)
        check(False, "zero channel count rejected")
    except ValueError:
        check(True, "zero channel count rejected")
    try:
        build_packet([0xFE00])
        check(False, "unencodable high byte rejected")
    except ValueError:
        check(True, "unencodable high byte rejected")


def _encode_polyg_value(value):
    if value % 2:
        raise ValueError("D1WD10 ADC counts must be even; bit 0 is the marker")
    return bytes([((value // 256) + 0x80) & 0xFF, value & 0xFF])


def test_polyg_hid_protocol():
    print("[polyg-hid] D1WD10 commands + offset-binary/marker-bit decoder")
    check(command_report(1, 1, 0) == bytes([0, 1, 1, 0, 0, 0, 0, 0, 0]),
          "start command has report ID plus 8-byte vendor payload")
    eeg_pattern = [-32768, -518, -2, 0, 2, 258, 12344, 32766]
    physical_pattern = eeg_pattern + [1000] * 8
    payload = b"".join(_encode_polyg_value(value) for value in physical_pattern * 32)
    marked = bytearray(payload)
    marked[1] |= 1
    rows = decode_report(marked)
    check(len(rows) == 32, "1024-byte report decodes to 32 rows over 16 channels")
    check(rows[0] == eeg_pattern and rows[-1] == eeg_pattern,
          "offset-binary conversion and marker-bit removal match D1WD10 DLL")
    check(decode_report(b"\x00" + payload) == rows,
          "optional Windows report-ID byte is accepted")
    mv = counts_to_adc_mv([32766, 0, -32768])
    check(abs(mv[0] - 32766 * ADC_VOLTS_PER_COUNT * 1000) < 1e-12,
          "embedded vendor coefficient converts counts to ADC millivolts")
    try:
        decode_report(payload[:-1])
        check(False, "short HID report rejected")
    except ValueError:
        check(True, "short HID report rejected")

    class FakeDevice:
        def __init__(self):
            self.writes = []

        def open_path(self, path):
            self.path = path

        def set_nonblocking(self, value):
            self.nonblocking = value

        def write(self, report):
            self.writes.append(bytes(report))
            return len(report)

        @staticmethod
        def read(_size):
            return []

        def close(self):
            self.closed = True

    fake_device = FakeDevice()

    class FakeHID:
        @staticmethod
        def enumerate(_vid, _pid):
            return [{"path": b"fake-polyg"}]

        @staticmethod
        def device():
            return fake_device

    with PolyGIHID(hid_module=FakeHID) as device:
        device.start()
    expected = [
        command_report(1, 0, 0), command_report(5, 16, 0),
        command_report(4, 8, 0), command_report(11, 6, 0),
        command_report(1, 1, 0), command_report(1, 0, 0),
    ]
    check(fake_device.writes == expected,
          "initialization and cleanup reproduce the exact vendor command sequence")


def test_eeg_dashboard_helpers():
    print("[dashboard] safe filenames + honest signal-presence labels")
    safe = sanitize_recording_name("../ 참가자 A / 세션 1")
    check("/" not in safe and ".." not in safe, "recording filename traversal removed")
    check(analyze_signal_quality([0.0] * 64, [0] * 64)["state"] == "flat",
          "constant channel is labeled flat")
    check(analyze_signal_quality([1.0, -1.0] * 32, [32766, -32768] * 32)["state"] == "saturated",
          "clipped channel is labeled saturated")
    check(analyze_signal_quality([-1.0, 0.5, 1.2, -0.4] * 16, [0] * 64)["state"] == "present",
          "varying unclipped channel is labeled signal-present")

    processor = EEGSignalProcessor(fs=256, channels=8)
    rows = []
    for n in range(256 * 4):
        value = 2000 * math.sin(2 * math.pi * 10 * n / 256)
        rows.append([value] * 8)
    _raw, filtered = processor.process(rows)
    settled = [row[0] for row in filtered[512:]]
    check(max(settled) - min(settled) > 50,
          "stateful 0.5–45 Hz filter preserves a 10 Hz EEG-band signal")

    chunked_processor = EEGSignalProcessor(fs=256, channels=8)
    chunked = []
    for start in range(0, len(rows), 32):
        _raw_chunk, filtered_chunk = chunked_processor.process(rows[start:start + 32])
        chunked.extend(filtered_chunk)
    max_difference = max(
        abs(expected[0] - actual[0])
        for expected, actual in zip(filtered, chunked)
    )
    check(max_difference < 1e-9,
          "filter state makes 32-row HID boundaries numerically continuous")

    notch_processor = EEGSignalProcessor(fs=256, channels=8)
    mains_rows = [
        [2000 * math.sin(2 * math.pi * 60 * n / 256)] * 8
        for n in range(256 * 4)
    ]
    _raw_mains, filtered_mains = notch_processor.process(mains_rows)
    mains_rms = (sum(row[0] ** 2 for row in filtered_mains[512:]) / 512) ** 0.5
    signal_rms = (sum(value ** 2 for value in settled) / len(settled)) ** 0.5
    check(mains_rms < signal_rms * 0.1,
          "60 Hz notch and 45 Hz low-pass strongly reject mains-frequency input")


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
    try:
        kinematics.solve(1000, 1000, 0)
        check(False, "unreachable target rejected")
    except ValueError:
        check(True, "unreachable target rejected")
    calibrated = config.ARM_CALIBRATED
    try:
        config.ARM_CALIBRATED = True
        try:
            kinematics.joint_to_servo(config.J_BASE, 1000)
            check(False, "calibrated servo saturation rejected")
        except ValueError:
            check(True, "calibrated servo saturation rejected")
    finally:
        config.ARM_CALIBRATED = calibrated


def test_policy_veto_scope():
    print("[policy] veto scope resets for the next requested object")
    from policy import Policy
    from vision import Detection
    policy = Policy()
    a = Detection(1, "a", 8.0, -3.0)
    b = Detection(2, "b", 12.0, 4.0)
    policy.reject(a)
    check(policy.choose([a, b], (0, 0)) is b, "veto excludes target in current selection")
    check(not policy.preference, "spatial learning is off without task identity")
    policy.reset_selection()
    check(policy.choose([a], (0, 0)) is a, "veto clears after an accepted delivery")


def test_arm_command_validation():
    print("[arm] rejects malformed or unsafe commands before serial write")
    from arm_serial import ArmSerial, parse_status_line
    arm = ArmSerial(mock=True)
    check(parse_status_line("C 90 90 90 180 90 180 180") == config.HOME_POSE,
          "strict firmware status is parsed")
    check(config.HOME_POSE[3] == 180 and config.HOME_POSE[5] == 180,
          "physical servos 4 and 6 use the requested 180-degree home angle")
    for invalid in ("C 90 90", "C 90 90 90 bad 90 180 180",
                    "C 90 90 90 180 90 180 181"):
        try:
            parse_status_line(invalid)
            check(False, f"malformed status rejected: {invalid}")
        except ValueError:
            check(True, f"malformed status rejected: {invalid}")
    try:
        arm.send_angles([90] * (config.N_JOINTS - 1))
        check(False, "wrong joint count rejected")
    except ValueError:
        check(True, "wrong joint count rejected")
    unsafe = list(config.HOME_POSE)
    unsafe[0] = config.SERVO_MAX[0] + 1
    try:
        arm.send_angles(unsafe)
        check(False, "out-of-range angle rejected")
    except ValueError:
        check(True, "out-of-range angle rejected")


def test_planar_pick_calibration_and_detection():
    print("[planar] fixed-base calibration, safe sequencing, portable perception")
    from planar_pick import pick_pose_for_object_x, stepped_values
    from vision_segment import (
        ObjectDetection, select_gripper, select_pick_target)

    check(config.GRIP_OPEN == 90 and config.GRIP_CLOSED == 180,
          "physical gripper direction is 90=open, 180=closed")
    check(config.PLANAR_SERVO_MIN[config.J_BASE]
          == config.PLANAR_SERVO_MAX[config.J_BASE] == 90,
          "broken base is locked at 90 in planar safety limits")
    check(config.PLANAR_SERVO_MIN[config._UNUSED]
          == config.PLANAR_SERVO_MAX[config._UNUSED] == 90,
          "unused servo3 is locked at 90 in planar safety limits")
    check(pick_pose_for_object_x(1435) == (140, 90),
          "verified image reference maps to the successful physical pick pose")
    right_pose = pick_pose_for_object_x(1505)
    left_pose = pick_pose_for_object_x(1365)
    check(right_pose[1] < 90 and left_pose[1] > 90,
          "elbow correction follows measured camera x direction")
    check(stepped_values(90, 95, 2) == [92, 94, 95]
          and stepped_values(95, 90, 2) == [93, 91, 90],
          "two-degree safety waypoints include the exact endpoint")

    target = ObjectDetection((1415, 905), (1320, 870, 190, 70), 13300, .94)
    gripper = ObjectDetection((1412, 700), (1372, 620, 80, 160), 12800, .70)
    picture = ObjectDetection((1795, 780), (1730, 763, 130, 35), 4550, .83)
    wall = ObjectDetection((960, 430), (0, 0, 1920, 860), 1651200, .99)
    candidates = [wall, picture, gripper, target]
    selected = select_pick_target(candidates, (1080, 1920, 3))
    check(selected is target,
          "segment selector chooses the compact tabletop object without a background")
    check(select_gripper(candidates, target, (1080, 1920, 3)) is gripper,
          "segment selector locates the nearest gripper section above the object")

    lifted = ObjectDetection((1405, 825), (1310, 790, 190, 70), 13300, .58)
    selected = select_pick_target(
        [picture, gripper, lifted], (1080, 1920, 3), previous=target)
    check(selected is lifted,
          "target identity follows the object upward for grasp verification")


def test_validate_handles_short_arrays():
    print("[config] malformed servo arrays are reported without crashing")
    import validate
    original = config.SERVO_MIN
    try:
        config.SERVO_MIN = [0]
        errors, _warnings = validate.validate()
        check(any("SERVO_MIN has" in error for error in errors),
              "short servo array reported")
    finally:
        config.SERVO_MIN = original


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


def test_errp_model_metadata():
    print("[errp] trained model roundtrip validates acquisition metadata")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("  skip scikit-learn is not installed (baseline backend remains available)")
        return
    import os
    import tempfile
    det = ErrPDetector(backend="baseline")
    windows = [_synth_epoch(config.EEG_FS, error=bool(i % 2)) for i in range(6)]
    labels = [i % 2 for i in range(6)]
    det.fit(windows, labels)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.pkl")
        det.save(path)
        loaded = ErrPDetector(backend="model", model_path=path)
        check(loaded.backend == "model", "metadata-bearing model loads")
        check(loaded.p_error(windows[1]) > loaded.p_error(windows[0]),
              "loaded model preserves classification")


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


def test_ring_recent_is_time_based():
    print("[eeg] recent snapshot uses timestamps rather than a fixed count")
    from eeg_bridge import RingBuffer
    import time
    ring = RingBuffer(10)
    now = time.monotonic()
    ring.push([1], now - 2.0)
    ring.push([2], now - 0.1)
    check(ring.recent(1.0) == [[2]], "stale samples are excluded from health checks")


def test_pick_place():
    print("[pickplace] visual correction is preserved through grasp + delivery")
    import random
    import sim
    from arm_serial import ArmSerial
    from policy import Policy
    from vision import Vision, Detection
    import orchestrator as orch

    sim.WORLD = sim.World(sim.DEFAULT_OBJECTS)   # fresh world
    random.seed(7)
    arm = ArmSerial(mock=True)
    policy = Policy()
    vision = Vision(mock=True)
    target = Detection(1, "nail_small", 8.0, -3.0, {"size": "small"})

    n_before = len(sim.WORLD.objects)
    grasp_xy, err, aligned = orch.servo_to_object(
        arm, vision, policy, target, config.Z_APPROACH)
    check(aligned and err <= config.SERVO_TOL_CM, "visual servo converged")
    check(grasp_xy != (target.x, target.y), "alignment produced a corrected command")
    ok = orch.grasp_object(arm, vision, policy, target, grasp_xy=grasp_xy)
    check(ok, "grasp verified in mock")
    orch.place_object(arm, policy, config.PLACE_LOCATION)
    check(len(sim.WORLD.objects) == n_before - 1, "object removed from table after delivery")
    check(sim.WORLD._holding is None, "gripper released after place")
    arm.close()


def test_servo_visibility_failure():
    print("[servo] missing arm tip prevents descent")
    from arm_serial import ArmSerial
    from policy import Policy
    from vision import Detection
    import orchestrator as orch

    class BlindVision:
        @staticmethod
        def arm_tip(expected_xy=None):
            return None

    arm = ArmSerial(mock=True)
    target = Detection(1, "obj", 8.0, -3.0)
    _xy, err, aligned = orch.servo_to_object(
        arm, BlindVision(), Policy(), target, config.Z_APPROACH)
    check(err is None and not aligned, "alignment fails closed when tip is unseen")


if __name__ == "__main__":
    test_lxsdf_roundtrip()
    test_lxsdf_resync()
    test_lxsdf_drops()
    test_lxsdf_rejects_invalid_shapes()
    test_polyg_hid_protocol()
    test_eeg_dashboard_helpers()
    test_ik()
    test_policy_veto_scope()
    test_arm_command_validation()
    test_planar_pick_calibration_and_detection()
    test_validate_handles_short_arrays()
    test_errp()
    test_errp_model_metadata()
    test_eeg_packet_timestamps()
    test_ring_recent_is_time_based()
    test_pick_place()
    test_servo_visibility_failure()
    print("\nALL TESTS PASSED")
