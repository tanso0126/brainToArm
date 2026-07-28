"""Hardware-free tests for the parts that must be exactly right.

Run: python test_pipeline.py
Exercises the real code paths (no mocks of the units under test):
  - LXSDF encode -> parse roundtrip, channel auto-detect, resync after garbage,
    packet-count drop detection.
  - IK produces in-range servo commands and points the base toward the target.
  - ErrP detector separates a synthetic error epoch from a clean epoch.
"""
import math
import time
import config
from lxsdf import LXSDFParser, build_packet
import kinematics
from errp import ErrPDetector
from eeg_bridge import EEGBridge
from cognitive_load import AutonomyAllocator, CognitiveLoadEstimator
from polyg_hid import (
    ADC_VOLTS_PER_COUNT, PolyGIHID, command_report, counts_to_adc_mv, decode_report)
from eeg_dashboard import (
    MAX_TRANSIENT_CLIPPING_PERCENT, EEGDashboardService, EEGSignalProcessor,
    analyze_signal_quality, sanitize_recording_name)
from wrist_vision import (
    LATEST_PREVIEW_PATH, LATEST_RAW_PATH, WristDetector, frame_quality)
from wrist_search import PlanarSearchSafety, WristSearcher
from visual_contact import JawBaseline, JawSample
from capture_esp32_camera import correct_neutral_background, validate_frame_quality


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
    transient_counts = [32766] * 5 + [0] * 95
    check(analyze_signal_quality(
        [-1.0, 0.5, 1.2, -0.4] * 25,
        transient_counts)["state"] == "present",
        f"up to {MAX_TRANSIENT_CLIPPING_PERCENT:.0f}% transient rail samples are tolerated")
    sustained_counts = [32766] * 6 + [0] * 94
    check(analyze_signal_quality(
        [-1.0, 0.5, 1.2, -0.4] * 25,
        sustained_counts)["state"] == "saturated",
        "sustained rail occupancy still blocks calibration")
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
        check(len(a) == config.N_JOINTS, "6 servo values")
        check(all(config.SERVO_MIN[i] <= a[i] <= config.SERVO_MAX[i]
                  for i in range(config.N_JOINTS)),
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
    from arm_serial import (ArmSerial, _serial_candidates,
                            assert_home_pose_match, parse_home_pose_line,
                            parse_status_line)
    from unittest.mock import patch

    with patch("arm_serial.glob.glob") as mocked_glob:
        mocked_glob.side_effect = [
            [],
            ["/dev/cu.usbserial-0001", "/dev/cu.usbserial-110"],
            [],
            [],
        ]
        check(_serial_candidates() == ["/dev/cu.usbserial-110"],
              "arm auto-discovery excludes the ESP32 camera CP2102 port")
    arm = ArmSerial(mock=True)
    check(config.N_JOINTS == 6
          and config.SERVO_PINS == [13, 12, 11, 10, 9, 8]
          and config.JOINT_NAMES
          == ["base", "shoulder", "elbow", "wrist_pitch", "gripper", "wrist_roll"],
          "physical six-servo pin and joint map is exact")
    check(config.HOME_POSE == [90, 70, 90, 140, 170, 170],
          "six-servo HOME pose matches the latest raised camera pose")
    check(config.SERVO_MIN == [0, 0, 0, 130, 90, 0]
          and config.SERVO_MAX == [180, 150, 180, 180, 180, 180],
          "host limits match the physical wrist/gripper/roll ranges")
    home_status = "C " + " ".join(str(angle) for angle in config.HOME_POSE)
    check(parse_status_line(home_status) == config.HOME_POSE,
          "strict firmware status is parsed")
    firmware_home = "H " + " ".join(str(angle) for angle in config.HOME_POSE)
    check(parse_home_pose_line(firmware_home) == config.HOME_POSE,
          "compiled firmware HOME pose is parsed")
    check(assert_home_pose_match(config.HOME_POSE),
          "matching local and firmware HOME poses pass")
    stale_home = list(config.HOME_POSE)
    stale_home[config.J_SHOULDER] += 1
    try:
        assert_home_pose_match(stale_home)
        check(False, "stale firmware HOME pose rejected")
    except RuntimeError:
        check(True, "stale firmware HOME pose rejected")
    base_turn = list(config.HOME_POSE)
    base_turn[config.J_BASE] = 180
    check(arm.send_angles(base_turn) == "OK",
          "restored base servo accepts a 180-degree jog command")
    for invalid in ("C 90 90", "C 90 70 bad 90 0 180",
                    "C 90 70 90 90 0 181",
                    "C 90 70 90 90 90 0 180"):
        try:
            parse_status_line(invalid)
            check(False, f"malformed status rejected: {invalid}")
        except ValueError:
            check(True, f"malformed status rejected: {invalid}")
    for invalid in ("H 90 90", "H 90 70 bad 90 0 180",
                    "H 90 70 90 90 0 181",
                    "H 90 70 90 90 90 0 180"):
        try:
            parse_home_pose_line(invalid)
            check(False, f"malformed firmware HOME rejected: {invalid}")
        except ValueError:
            check(True, f"malformed firmware HOME rejected: {invalid}")
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
          "side-camera calibration keeps the working base at 90")
    check(config.N_JOINTS == 6 and config.J_ELBOW == 2
          and config.J_GRIP == 4 and config.J_ROLL == 5,
          "six-servo map has gripper on 5 and wrist roll on 6")
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
            if error and ch in config.ERRP_CHANNELS and 0.25 < t < 0.5:
                v -= 40    # negative deflection ~ErrP
            row.append(v)
        win.append(row)
    return win


def _synth_load_window(theta_amplitude, alpha_amplitude, seconds=4.0):
    rows = []
    for sample in range(int(config.EEG_FS * seconds)):
        t = sample / config.EEG_FS
        theta = theta_amplitude * math.sin(2 * math.pi * 6 * t)
        alpha = alpha_amplitude * math.sin(2 * math.pi * 10 * t)
        row = []
        for channel in range(config.EEG_CHANNELS):
            if channel in config.COG_THETA_CHANNELS:
                row.append(theta + 0.2 * alpha)
            elif channel in config.COG_ALPHA_CHANNELS:
                row.append(alpha + 0.2 * theta)
            else:
                row.append(math.sin(2 * math.pi * (9 + channel * 0.1) * t))
        rows.append(row)
    return rows


def test_cognitive_load_and_autonomy():
    print("[load] channel-specific TAR + rest normalization + autonomy allocation")
    rest = _synth_load_window(theta_amplitude=4.0, alpha_amplitude=8.0)
    high = _synth_load_window(theta_amplitude=8.0, alpha_amplitude=4.0)
    low = _synth_load_window(theta_amplitude=2.0, alpha_amplitude=12.0)

    high_estimator = CognitiveLoadEstimator()
    baseline = high_estimator.calibrate(rest)
    high_state = high_estimator.update(high)
    low_estimator = CognitiveLoadEstimator()
    low_estimator.calibrate(rest)
    low_state = low_estimator.update(low)
    check(baseline.relative_tar == 0.0,
          "resting TAR is the zero point")
    check(high_state.relative_tar > 0 and low_state.relative_tar < 0,
          "(current-rest)/rest follows theta-up/alpha-down load direction")
    check(len(high_state.theta_powers) == 4
          and len(high_state.alpha_powers) == 1,
          "TAR uses theta CH1-CH4 and alpha CH8")

    allocator = AutonomyAllocator()
    rest_allocation = allocator.allocate(baseline)
    high_allocation = allocator.allocate(high_state)
    low_allocation = allocator.allocate(low_state)
    check(rest_allocation.robot_weight == config.AUTONOMY_ROBOT_BASE
          and rest_allocation.errp_apply_stride == 1,
          "rest/deadband keeps balanced authority and every ErrP checkpoint")
    check(high_allocation.robot_weight > config.AUTONOMY_ROBOT_BASE
          and high_allocation.errp_threshold > config.ERRP_THRESHOLD
          and high_allocation.errp_apply_stride > 1,
          "higher TAR shifts authority toward robot and reduces ErrP application")
    check(low_allocation.robot_weight < config.AUTONOMY_ROBOT_BASE
          and low_allocation.errp_threshold < config.ERRP_THRESHOLD
          and low_allocation.errp_apply_stride == 1,
          "lower TAR shifts authority toward human and applies every ErrP checkpoint")
    invalid_state = high_estimator.update(
        [[0.0] * config.EEG_CHANNELS
         for _ in range(int(config.COG_WINDOW_S * config.EEG_FS))])
    invalid_allocation = allocator.allocate(invalid_state)
    check(not invalid_state.valid
          and invalid_allocation.robot_weight == config.AUTONOMY_ROBOT_MIN
          and invalid_allocation.errp_apply_stride == 1,
          "flat/invalid PSD fails toward minimum robot authority")
    first = allocator.decide(0.8, high_state)
    second = allocator.decide(0.8, high_state)
    override = allocator.decide(config.AUTONOMY_ERRP_OVERRIDE_THRESHOLD, high_state)
    check(first.applied and not second.applied,
          "high-load ErrP is still calculated but only scheduled checkpoints apply")
    check(override.veto and override.override and override.applied,
          "very strong ErrP remains a load-independent veto")


def test_dashboard_integrates_tar_allocation():
    print("[dashboard-load] one rest window drives live TAR + autonomy status")
    service = EEGDashboardService(enumerate_fn=lambda: [])

    def install_window(window):
        now = time.monotonic()
        first = now - (len(window) - 1) / config.EEG_FS
        service._rows.clear()
        for index, values in enumerate(window):
            counts = [int(round(value * 100)) for value in values]
            service._rows.append((
                index + 1,
                first + index / config.EEG_FS,
                counts,
                list(values),
                list(values),
            ))
        service._running = True
        service._started_mono = first
        service._last_report_mono = now

    try:
        install_window(_synth_load_window(
            theta_amplitude=4.0, alpha_amplitude=8.0, seconds=8.0))
        calibrated = service.calibrate_errp(seconds=8.0)
        check(calibrated["cognitiveLoad"]["thetaChannels"] == [1, 2, 3, 4]
              and calibrated["cognitiveLoad"]["alphaChannels"] == [8],
              "dashboard rest calibration uses theta CH1-4 and alpha CH8")

        install_window(_synth_load_window(
            theta_amplitude=8.0, alpha_amplitude=4.0, seconds=4.0))
        service._load_last_update_mono = 0.0
        payload = service.status()["cognitiveLoad"]
        check(payload["smoothedRelativeTar"] > 0,
              "dashboard publishes positive rest-relative TAR under higher load")
        check(payload["robotWeight"] > config.AUTONOMY_ROBOT_BASE
              and payload["errpApplyStride"] > 1,
              "web simulation receives robot-biased adaptive allocation")
    finally:
        service.close()


def test_errp():
    print("[errp] separates error vs clean epoch")
    check(config.ERRP_CHANNELS == [7],
          "ErrP consumes only project-mapped EEG CH8")
    det = ErrPDetector(backend="baseline")
    det.update_baseline(_synth_epoch(config.EEG_FS, error=False))
    p_ok = det.p_error(_synth_epoch(config.EEG_FS, error=False))
    error_epoch = _synth_epoch(config.EEG_FS, error=True)
    diagnostics = det.diagnose(error_epoch)
    p_err = diagnostics["probability"]
    print(f"       P(ok)={p_ok:.2f}  P(err)={p_err:.2f}")
    check(p_err > p_ok, "error epoch scores higher than clean")
    check(diagnostics["zScore"] is not None
          and diagnostics["zScore"] > 3.3,
          "CH8 sustained negative deflection crosses the audited z threshold")
    wrong_channel = _synth_epoch(config.EEG_FS, error=False)
    for sample, row in enumerate(wrong_channel):
        t = sample / config.EEG_FS
        if 0.25 < t < 0.5:
            row[0] -= 40
    check(det.p_error(wrong_channel) < config.ERRP_THRESHOLD,
          "a CH1-only deflection cannot masquerade as the specified CH8 ErrP")


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


def test_camera_neutral_correction():
    print("[camera] bounded gray-background color correction")
    import numpy as np
    from PIL import Image

    pixels = np.empty((40, 40, 3), dtype=np.uint8)
    pixels[:] = [90, 110, 100]  # deliberately green-tinted neutral background
    pixels[5:12, 5:12] = [220, 30, 30]  # colored object must stay red
    corrected, gains, neutral_count = correct_neutral_background(Image.fromarray(pixels))
    result = np.asarray(corrected)
    background = result[20:, 20:].mean(axis=(0, 1))
    check(background.max() - background.min() <= 2,
          "neutral background channels converge")
    check(result[8, 8, 0] > result[8, 8, 1] * 4,
          "strong object color is preserved")
    check(neutral_count > pixels.shape[0] * pixels.shape[1] * 0.8,
          "colored object is excluded from the gain estimate")
    check(all(0.85 <= gain <= 1.15 for gain in gains),
          "per-channel correction remains bounded")

    clean = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    validate_frame_quality(clean, neutral_count=64 * 64)
    check(True, "neutral non-clipped frame passes quality validation")

    mosaicked = np.empty((64, 64, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            mosaicked[row * 8:(row + 1) * 8, column * 8:(column + 1) * 8] = \
                70 if (row + column) % 2 else 180
    try:
        validate_frame_quality(Image.fromarray(mosaicked), neutral_count=64 * 64)
        check(False, "8-pixel mosaic corruption rejected")
    except ValueError:
        check(True, "8-pixel mosaic corruption rejected")


def test_wrist_marker_and_target_geometry():
    print("[wrist] blue-left/red-right eye-in-hand geometry + quality gate")
    import numpy as np
    import cv2

    check(LATEST_RAW_PATH != LATEST_PREVIEW_PATH
          and "raw" in LATEST_RAW_PATH.stem,
          "controller input is separate from the annotated operator preview")
    check(config.WRIST_MARKER_ROI == (0.22, 0.75, 0.72, 1.0),
          "rigid markers are extracted only from the measured lower-frame ROI")

    frame = np.full((720, 1280, 3), 125, dtype=np.uint8)
    # Small distractors are not allowed to replace the lower-center finger pair.
    cv2.rectangle(frame, (80, 80), (105, 95), (255, 0, 0), -1)
    cv2.rectangle(frame, (1120, 90), (1145, 105), (0, 0, 255), -1)
    cv2.rectangle(frame, (100, 390), (230, 470), (0, 105, 210), -1)  # skin/orange
    cv2.rectangle(frame, (430, 667), (529, 719), (255, 0, 0), -1)
    cv2.rectangle(frame, (705, 667), (797, 719), (0, 0, 255), -1)
    # A valid object can appear anywhere in the camera, including the upper half.
    measured_yellow = tuple(int(value) for value in cv2.cvtColor(
        np.uint8([[[22, 180, 245]]]), cv2.COLOR_HSV2BGR)[0, 0])
    target_box = cv2.boxPoints(((665, 180), (120, 55), 20)).astype(np.int32)
    cv2.fillConvexPoly(frame, target_box, measured_yellow)

    observation, _masks = WristDetector().detect(frame)
    check(observation.gripper is not None,
          "both physical finger colors produce one gripper pose")
    check(observation.target is not None,
          "colored object remains separate from finger markers")
    check(abs(observation.target.center[0] - 665) < 2
          and abs(observation.target.center[1] - 180) < 2,
          "vivid-yellow target detection covers the full frame")
    check(abs(observation.gripper.center[0] - 615.25) < 2
          and abs(observation.gripper.center[1] - 693) < 2,
          "open-state bottom-edge tapes define their measured midpoint")
    check(observation.gripper.jaw_axis[0] > 0.99,
          "jaw axis points mechanically from blue-left to red-right")
    check(observation.error_xy[1] < 0 and observation.distance_px > 100,
          "target-to-grasp alignment error is reported in image pixels")
    check(observation.orientation_error_deg is not None,
          "elongated target exposes wrist-roll orientation error")
    check(not observation.aligned,
          "target outside the fixed pixel tolerance is not ready")

    wrong_location = np.full((720, 1280, 3), 125, dtype=np.uint8)
    cv2.rectangle(wrong_location, (430, 260), (529, 312), (255, 0, 0), -1)
    cv2.rectangle(wrong_location, (705, 260), (797, 312), (0, 0, 255), -1)
    wrong_observation, _ = WristDetector().detect(wrong_location)
    check(wrong_observation.gripper is None,
          "correctly sized red/blue scene objects outside the mount zone are rejected")

    inverted = np.full((720, 1280, 3), 125, dtype=np.uint8)
    cv2.rectangle(inverted, (430, 667), (529, 719), (0, 0, 255), -1)
    cv2.rectangle(inverted, (705, 667), (797, 719), (255, 0, 0), -1)
    inverted_observation, _ = WristDetector().detect(inverted)
    check(inverted_observation.gripper is None,
          "red-left/blue-right distractor pair is rejected")

    closed = np.full((720, 1280, 3), 125, dtype=np.uint8)
    cv2.rectangle(closed, (500, 620), (590, 719), (255, 0, 0), -1)
    cv2.rectangle(closed, (635, 620), (725, 719), (0, 0, 255), -1)
    closed_observation, _ = WristDetector().detect(closed)
    check(closed_observation.gripper is not None,
          "measured closed-state tape profile remains valid")

    bottom_target = np.full((720, 1280, 3), 245, dtype=np.uint8)
    cv2.rectangle(bottom_target, (430, 667), (529, 719), (255, 0, 0), -1)
    cv2.rectangle(bottom_target, (705, 667), (797, 719), (0, 0, 255), -1)
    cv2.rectangle(bottom_target, (610, 645), (665, 719), measured_yellow, -1)
    bottom_observation, _ = WristDetector().detect(bottom_target)
    check(bottom_observation.target is not None,
          "bottom-clipped yellow target between known fingers is accepted")

    bottom_clutter = np.full((720, 1280, 3), 245, dtype=np.uint8)
    cv2.rectangle(bottom_clutter, (430, 667), (529, 719), (255, 0, 0), -1)
    cv2.rectangle(bottom_clutter, (705, 667), (797, 719), (0, 0, 255), -1)
    cv2.rectangle(bottom_clutter, (900, 645), (970, 719), measured_yellow, -1)
    clutter_observation, _ = WristDetector().detect(bottom_clutter)
    check(clutter_observation.target is None,
          "bottom-border yellow clutter outside the fingers is rejected")

    clipped = np.full((720, 1280, 3), 255, dtype=np.uint8)
    quality = frame_quality(clipped)
    check(not quality.valid and quality.clipped_fraction == 1.0,
          "overexposed view fails closed before visual alignment")


def test_wrist_full_frame_search():
    print("[wrist-search] collision-checked 2/3/4 search stops on target")
    from types import SimpleNamespace
    import numpy as np

    safety = PlanarSearchSafety(
        offsets=[0, 90, 90, 130, 0, 0], directions=[1] * 6,
        lengths=(3, 3, 2), base_height=10,
        table_clearance=.5, base_clearance=.5, self_clearance=.5,
        slew_step=1.5)
    safe_pose = [90, 90, 90, 130, 170, 0]
    base_collision = [90, 0, 90, 130, 170, 0]
    check(safety.pose_is_safe(safe_pose),
          "forward model accepts a clear horizontal three-link pose")
    check(not safety.pose_is_safe(base_collision),
          "forward model rejects forearm contact with the base column")
    target_pose_a = [90, 105, 90, 140, 170, 0]
    target_pose_b = [90, 105, 105, 150, 170, 0]
    check(safety.transition_is_safe(safe_pose, target_pose_a),
          "every firmware-slew intermediate is collision checked")

    class FakeArm:
        def __init__(self):
            self.pose = list(safe_pose)
            self.commands = []

        def status(self):
            return list(self.pose)

        def send_angles(self, pose):
            self.pose = list(pose)
            self.commands.append(list(pose))

        @staticmethod
        def wait_done():
            return True

    class FakeCamera:
        @staticmethod
        def read():
            return True, np.zeros((32, 32, 3), dtype=np.uint8)

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            target = SimpleNamespace(center=(10.0, 10.0)) if self.calls == 3 else None
            observation = SimpleNamespace(
                quality=SimpleNamespace(valid=True, reason=""),
                target=target, gripper=None, aligned=False,
                error_xy=None, distance_px=None)
            return observation, {}

    class FakePlanner:
        def __init__(self):
            self.safety = safety

        @staticmethod
        def plan(_pose, _max_poses=None):
            return [target_pose_a, target_pose_b]

    arm = FakeArm()
    found = WristSearcher(
        arm, FakeCamera(), FakeDetector(), FakePlanner()).find(max_poses=2)
    commanded = [tuple(pose[joint] for joint in config.WRIST_SEARCH_JOINTS)
                 for pose in arm.commands]
    check(found is not None and commanded == [(105, 90, 140), (105, 105, 150)],
          "coordinated search stops immediately at first target-visible pose")


def test_visual_gripper_contact():
    print("[contact] scene-independent visual jaw obstruction")
    from types import SimpleNamespace
    samples = [
        JawSample(90, 282.0, 1.2, 618.0, 701.0, 15),
        JawSample(130, 251.0, 1.5, 618.0, 688.0, 15),
        JawSample(150, 202.0, 1.8, 618.0, 681.0, 15),
        JawSample(170, 126.0, 1.4, 618.0, 675.0, 15),
        JawSample(180, 84.0, 1.5, 618.0, 672.0, 15),
    ]
    baseline = JawBaseline(samples, camera_name="test")
    expected = baseline.expected(160)
    check(abs(expected.opening_px - 164.0) < 1e-6,
          "empty-jaw curve interpolates between calibrated commands")

    def observation(opening):
        return SimpleNamespace(gripper=SimpleNamespace(opening_px=opening))

    free = baseline.assess(170, observation(128.0))
    blocked = baseline.assess(170, observation(154.0))
    missing = baseline.assess(170, None)
    check(free.state == "FREE", "normal close matches the empty-jaw baseline")
    check(blocked.contact and blocked.residual_px > blocked.threshold_px,
          "physically wider jaw is classified as contact")
    check(missing.state == "UNKNOWN",
          "missing markers fail closed instead of inventing contact")
    try:
        JawBaseline([
            samples[0],
            JawSample(130, 290.0, 1.0, 618.0, 688.0, 10),
            samples[-1],
        ])
        check(False, "non-monotonic empty-jaw calibration rejected")
    except ValueError:
        check(True, "non-monotonic empty-jaw calibration rejected")


def test_floor_motion_and_persistent_session():
    print("[floor] absolute floor curves + persistent Uno command owner")
    from arm_session import ArmSessionServer
    from floor_motion import floor_pose, floor_vector, floor_waypoints

    hover = floor_pose(90, "hover")
    grasp = floor_pose(90, "grasp")
    check(hover == [90, 124, 90, 180, 90, 170],
          "reproduced floor hover uses the physically level 170-degree roll")
    check(grasp == [90, 142, 90, 180, 90, 170],
          "reproduced floor grasp reaches the absolute floor reference")
    shifted = floor_pose(86, "hover")
    check(shifted[config.J_SHOULDER] == 126
          and shifted[config.J_ELBOW] == 86,
          "elbow motion receives simultaneous shoulder height compensation")
    shoulder_delta, elbow_delta = floor_vector(-4)
    check(abs(shoulder_delta - 24 / 11) < 1e-9 and elbow_delta == -4,
          "continuous floor vector cancels the measured vertical Jacobian")
    waypoints = floor_waypoints(90, 78, "grasp", step=4)
    check([pose[config.J_ELBOW] for pose in waypoints] == [86, 82, 78],
          "bounded ground path includes its exact target elbow")
    for pose in waypoints:
        shoulder_change = pose[config.J_SHOULDER] - grasp[config.J_SHOULDER]
        elbow_change = pose[config.J_ELBOW] - grasp[config.J_ELBOW]
        check(abs(11 * shoulder_change + 6 * elbow_change) <= 5.5,
              "rounded waypoints remain within half a shoulder-degree of floor height")
    try:
        floor_pose(77, "hover")
        check(False, "uncalibrated floor extent rejected")
    except ValueError:
        check(True, "uncalibrated floor extent rejected")

    class FakeArm:
        def __init__(self):
            self.pose = list(config.HOME_POSE)
            self.moves = []

        def send_angles(self, pose):
            self.pose = list(pose)
            self.moves.append(list(pose))
            return "OK"

        @staticmethod
        def wait_done(timeout=15):
            return True

        def status(self):
            return list(self.pose)

        @staticmethod
        def ping():
            return True

        @staticmethod
        def close():
            return None

    fake = FakeArm()
    server = ArmSessionServer("/tmp/brainToArm-test-arm.sock", arm=fake)
    response = server.handle({
        "command": "floor", "level": "hover", "elbow": 82, "step": 4,
    })
    check(response["pose"] == floor_pose(82, "hover")
          and len(fake.moves) == 2,
          "one persistent owner executes compensated waypoints without reopening Uno")


def test_floor_grasp_selection_and_reject():
    print("[floor-grasp] multi-object rank, marker/floor exclusion, reject cycle")
    from types import SimpleNamespace
    from vision_segment import ObjectDetection
    from floor_grasp import (
        CandidateSelector, FloorGraspController, consolidate_wrist_candidates,
        filter_wrist_candidates, rank_wrist_candidates)

    frame_shape = (720, 1280, 3)
    near = ObjectDetection((900, 420), (870, 400, 70, 60), 4200, 0.9)
    mid = ObjectDetection((640, 300), (610, 280, 50, 45), 2250, 0.9)
    far = ObjectDetection((300, 400), (280, 380, 60, 50), 3000, 0.9)
    tape = ObjectDetection((615, 690), (590, 667, 50, 52), 2600, 0.8)
    floor = ObjectDetection((640, 360), (0, 0, 1280, 720), 921600, 0.99)
    marker_boxes = [(590, 667, 50, 52), (705, 667, 50, 52)]

    valid = filter_wrist_candidates(
        [floor, near, tape, mid, far], frame_shape, marker_boxes)
    check(floor not in valid and tape not in valid and len(valid) == 3,
          "finger tapes and full-frame floor are excluded from candidates")
    ranked = rank_wrist_candidates(valid, frame_shape, gripper_center=(640, 690))
    check(ranked[0] is near,
          "several objects are recognized and the nearest jaw candidate ranks first")

    diagonal = math.hypot(1280, 720)
    selector = CandidateSelector(
        reject_radius_px=config.FLOOR_REJECT_RADIUS_RATIO * diagonal)
    check(selector.choose(ranked) is near, "step 1/2: one object is selected")
    selector.reject(near)
    check(selector.choose(ranked) is mid,
          "step 3: reject switches to the next object automatically")
    selector.reject(mid)
    check(selector.choose(ranked) is far, "successive rejects keep advancing")
    selector.reject(far)
    check(selector.choose(ranked) is None, "all-rejected leaves no live target")
    selector.confirm()
    check(selector.choose(ranked) is near,
          "confirm/deliver clears vetoes for the next goal (ErrP-ready hook)")

    # id-independent rejection: a renumbered frame with a shifted-but-same object.
    moved_near = ObjectDetection((905, 424), (875, 404, 70, 60), 4200, 0.9)
    selector = CandidateSelector(reject_radius_px=40.0)
    selector.reject(near)
    check(selector.choose([moved_near, mid]) is mid,
          "veto follows image position, not a per-frame segment id")

    # FastSAM commonly emits an outer mask and multiple part masks for one toy.
    # They must remain one physical choice or a reject would select another part
    # of the same object instead of advancing to the next object.
    outer = ObjectDetection((640, 500), (580, 450, 120, 100), 12000, 0.9)
    nested = ObjectDetection((642, 502), (605, 472, 75, 60), 4500, 0.8)
    separate = ObjectDetection((900, 500), (860, 465, 80, 70), 5600, 0.9)
    merged = consolidate_wrist_candidates([nested, separate, outer])
    check(merged == [outer, separate],
          "nested FastSAM masks collapse to one physical object choice")

    # Near contact the selected object is legitimately clipped by the lower
    # edge. Preserve it only when it is geometrically between both finger tapes.
    near_field = ObjectDetection((635, 680), (560, 640, 150, 80), 12000, 0.9)
    bottom_clutter = ObjectDetection((200, 680), (140, 640, 120, 80), 9600, 0.9)
    wide_markers = [(380, 665, 135, 55), (730, 665, 85, 55)]
    near_valid = filter_wrist_candidates(
        [near_field, bottom_clutter], frame_shape, wide_markers)
    check(near_valid == [near_field],
          "only a bottom-clipped object fully between the jaws survives")

    # Autonomous search must ignore segmented background objects until a
    # candidate enters the fixed-base jaw corridor. No remembered target pose is
    # involved in this choice.
    from floor_servo import FloorServo
    search_gripper = SimpleNamespace(center=(640.0, 700.0))
    background = ObjectDetection((665, 100), (620, 60, 90, 80), 7200, 0.9)
    off_axis = ObjectDetection((820, 600), (780, 560, 80, 80), 6400, 0.9)
    reachable = ObjectDetection((630, 560), (590, 520, 80, 90), 7200, 0.9)
    search_scene = SimpleNamespace(
        gripper=search_gripper,
        ranked=[background, off_axis, reachable])
    check(FloorServo.search_choice(search_scene) is reachable,
          "autonomous search selects only a candidate entering the jaw corridor")
    search_scene.ranked = [background, off_axis]
    check(FloorServo.search_choice(search_scene) is None,
          "autonomous search ignores distant and laterally unreachable clutter")

    retained_close = SimpleNamespace(contact=True, residual_px=80.0)
    retained_lift = SimpleNamespace(contact=True, residual_px=52.0)
    slipped_lift = SimpleNamespace(contact=True, residual_px=12.0)
    check(FloorServo.contact_retained(retained_close, retained_lift),
          "post-lift obstruction retains enough of the initial close residual")
    check(not FloorServo.contact_retained(retained_close, slipped_lift),
          "small residual after lift is rejected as a slipped object")

    # ---- fail-closed state machine (gated + executed paths) ----
    class FakeArm:
        def __init__(self):
            self.calls = []

        def status(self):
            return list(config.HOME_POSE)

        def move(self, pose, settle_s=0.0, timeout=15.0):
            self.calls.append(("move", list(pose)))
            return list(pose)

        def floor(self, level, elbow, settle_s=0.0, timeout=15.0):
            self.calls.append(("floor", level, int(elbow)))
            return list(config.HOME_POSE)

    def make_perceive(opening_px, target, keep_after_lift=True):
        # Object already on the jaw row (depth-aligned) and on the centerline,
        # so the measured-Jacobian depth servo converges on the first pass.
        def perceive():
            gripper = SimpleNamespace(center=(target.center[0], target.center[1] + 3.0))
            ranked = [target] if keep_after_lift else []
            scene = SimpleNamespace(
                ranked=ranked, gripper=gripper, marker_boxes=[],
                frame_shape=frame_shape)
            observation = SimpleNamespace(
                gripper=SimpleNamespace(opening_px=opening_px))
            return scene, observation
        return perceive

    target = ObjectDetection((640, 400), (610, 375, 60, 50), 3000, 0.9)

    gated = FloorGraspController(
        FakeArm(), make_perceive(126.0, target), baseline=None, execute=False)
    gated_result = gated.grasp(target)
    check(gated_result.ok and not gated_result.executed
          and gated_result.planned_elbow is not None,
          "gate off: recognize/align/plan succeeds without physical motion")

    samples = [
        JawSample(90, 282.0, 1.2, 618.0, 701.0, 15),
        JawSample(130, 251.0, 1.5, 618.0, 688.0, 15),
        JawSample(150, 202.0, 1.8, 618.0, 681.0, 15),
        JawSample(170, 126.0, 1.4, 618.0, 675.0, 15),
        JawSample(180, 84.0, 1.5, 618.0, 672.0, 15),
    ]
    baseline = JawBaseline(samples, camera_name="test")

    arm = FakeArm()
    contact = FloorGraspController(
        arm, make_perceive(154.0, target), baseline=baseline, execute=True)
    contact_result = contact.grasp(target)
    check(contact_result.ok and contact_result.executed
          and contact_result.state == "HOLD" and contact_result.contact == "CONTACT",
          "gate on + wide jaw + object still seen => verified lift")
    check(any(call[0] == "floor" and call[1] == "grasp" for call in arm.calls)
          and ("move", floor_pose_closed(contact_result.planned_elbow)) in arm.calls,
          "executed path descends to the floor and closes the gripper")

    no_baseline = FloorGraspController(
        FakeArm(), make_perceive(126.0, target), baseline=None, execute=True)
    nb_result = no_baseline.grasp(target)
    check(not nb_result.ok and nb_result.state == "STOP",
          "no jaw baseline fails closed instead of blindly lifting")

    free_arm = FakeArm()
    free = FloorGraspController(
        free_arm, make_perceive(84.0, target), baseline=baseline, execute=True)
    free_result = free.grasp(target)
    check(not free_result.ok and free_result.state == "STOP",
          "empty-baseline jaw (no contact) fails closed with OPEN/HOVER recovery")
    check(free_arm.calls[-1][0] == "move",
          "recovery issues an open-hover move before stopping")


def floor_pose_closed(elbow):
    from floor_motion import floor_pose
    return floor_pose(elbow, "grasp", gripper=config.GRIP_CLOSED)


def test_floor_homography():
    print("[floor-cal] pixel->floor homography from planar correspondences")
    import numpy as np
    from floor_calibrate import (
        apply_homography, reprojection_rms, solve_homography, FloorHomography)
    # A known floor->pixel projective map; recover its inverse pixel->floor and
    # check a held-out point. This is the depth-from-known-plane core.
    h_true = np.array([[2.1, 0.3, 640.0], [0.1, 2.4, 360.0],
                       [0.0005, 0.0009, 1.0]])

    def floor_to_pixel(point):
        d = h_true @ np.array([point[0], point[1], 1.0])
        return d[:2] / d[2]

    floor = np.array([[0, 0], [200, 0], [200, 150], [0, 150],
                      [100, 75], [50, 120], [180, 30]], dtype=float)
    pixels = np.array([floor_to_pixel(p) for p in floor])
    h = solve_homography(pixels, floor)
    check(reprojection_rms(h, pixels, floor) < 1e-6,
          "recovered homography reprojects the calibration points exactly")
    held = apply_homography(h, floor_to_pixel([130, 90]))
    check(abs(held[0] - 130) < 1e-3 and abs(held[1] - 90) < 1e-3,
          "held-out floor point recovered from its pixel (known-plane depth)")
    try:
        solve_homography([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
        check(False, "too-few correspondences rejected")
    except ValueError:
        check(True, "too-few correspondences rejected")
    saved = FloorHomography(matrix=h.tolist(), observation_pose=[90, 112, 90, 158, 90, 170],
                            rms_mm=0.0, source="test", created_at="t")
    check(abs(saved.pixel_to_floor(floor_to_pixel([50, 50]))[0] - 50) < 1e-3,
          "serializable calibration reprojects through its stored matrix")


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
    test_cognitive_load_and_autonomy()
    test_dashboard_integrates_tar_allocation()
    test_errp()
    test_errp_model_metadata()
    test_eeg_packet_timestamps()
    test_ring_recent_is_time_based()
    test_camera_neutral_correction()
    test_wrist_marker_and_target_geometry()
    test_wrist_full_frame_search()
    test_visual_gripper_contact()
    test_floor_motion_and_persistent_session()
    test_floor_grasp_selection_and_reject()
    test_floor_homography()
    test_pick_place()
    test_servo_visibility_failure()
    print("\nALL TESTS PASSED")
