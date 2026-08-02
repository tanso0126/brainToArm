"""노트북과 Arduino 로봇팔 사이의 USB 시리얼 연결입니다.

목표 관절 각도를 보내고 ACK/DONE 응답을 읽습니다. 실물 모드에서 필요한
패키지, 포트, 정상 응답이 없거나 움직임 시간이 초과되면 성공한 것처럼
넘어가지 않고 오류를 냅니다.
"""
import time
import glob
import statistics
import config

try:
    import serial  # pyserial
    _HAVE_SERIAL = True
except ImportError:
    _HAVE_SERIAL = False


def _serial_candidates():
    # macOS Arduino boards enumerate as /dev/cu.usbmodem* or /dev/cu.usbserial*
    cands = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*") \
        + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    excluded = set(getattr(config, "ARM_PORT_EXCLUDE", ()))
    return sorted(set(cands) - excluded)


def parse_status_line(line):
    """Parse firmware ``C a1..a6`` status without accepting partial data."""
    parts = str(line).strip().split()
    if len(parts) != config.N_JOINTS + 1 or parts[0] != "C":
        raise ValueError(f"로봇팔 상태 응답 형식이 잘못되었습니다: {line!r}")
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"로봇팔 상태에 정수가 아닌 값이 있습니다: {line!r}") from exc
    for index, value in enumerate(values):
        if not config.SERVO_MIN[index] <= value <= config.SERVO_MAX[index]:
            raise ValueError(
                f"로봇팔 상태의 {index + 1}번 관절={value} 값이 설정된 "
                "범위를 벗어났습니다.")
    return values


def parse_home_pose_line(line):
    """Parse the ``H a1..a6`` pose compiled into the connected Uno."""
    parts = str(line).strip().split()
    if len(parts) != config.N_JOINTS + 1 or parts[0] != "H":
        raise ValueError(f"펌웨어 HOME 응답 형식이 잘못되었습니다: {line!r}")
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"펌웨어 HOME에 정수가 아닌 값이 있습니다: {line!r}") from exc
    if any(not 0 <= value <= 180 for value in values):
        raise ValueError(f"펌웨어 HOME 각도가 0~180°를 벗어났습니다: {line!r}")
    return values


def parse_distance_line(line):
    """Parse firmware ``D millimetres``; ``D -1`` represents no valid echo."""
    parts = str(line).strip().split()
    if len(parts) != 2 or parts[0] != "D":
        raise ValueError(f"초음파 거리 응답 형식이 잘못되었습니다: {line!r}")
    try:
        value = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"초음파 거리가 정수가 아닙니다: {line!r}") from exc
    if value == -1:
        return None
    if not config.ULTRASONIC_MIN_MM <= value <= config.ULTRASONIC_MAX_MM:
        raise ValueError(f"초음파 거리 {value}mm가 유효 범위를 벗어났습니다.")
    return value


def assert_home_pose_match(compiled_pose, local_pose=None):
    """Fail closed when source and connected firmware describe different homes."""
    local_pose = list(config.HOME_POSE if local_pose is None else local_pose)
    compiled_pose = list(compiled_pose)
    if compiled_pose != local_pose:
        raise RuntimeError(
            "Uno 펌웨어 HOME 자세가 현재 home_pose.h와 다릅니다: "
            f"펌웨어={compiled_pose}, 현재 코드={local_pose}. "
            "로봇팔을 움직이기 전에 Windows에서는 OPEN_FIRMWARE.bat으로 "
            "현재 펌웨어를 업로드하세요.")
    return True


class ArmSerial:
    def __init__(self, port=None, baud=None, mock=None):
        self.port = port or config.ARM_PORT
        self.baud = baud or config.ARM_BAUD
        self.ser = None
        self.mock = config.ARM_MOCK if mock is None else bool(mock)
        if self.mock:
            print("[로봇팔] 모의 모드입니다. 실제로 움직이지 않고 명령만 표시합니다.")
        else:
            if not _HAVE_SERIAL:
                raise RuntimeError(
                    "실물 로봇팔을 사용하려면 PySerial이 필요합니다. "
                    "SETUP_WINDOWS.bat을 다시 실행하세요.")
            if self.port == "auto":
                candidates = _serial_candidates()
                if len(candidates) > 1:
                    raise RuntimeError(
                        "사용 가능한 시리얼 장치가 여러 개입니다. ARM_PORT "
                        "또는 ARM_PORT_EXCLUDE를 직접 지정하세요: "
                        + ", ".join(candidates))
                self.port = candidates[0] if candidates else None
            if not self.port:
                raise RuntimeError(
                    "실물 로봇팔을 요청했지만 시리얼 보드를 찾지 못했습니다. "
                    "Uno USB와 데이터 케이블을 확인하세요.")
            try:
                # TIOCEXCL prevents a second process from reopening this Uno and
                # toggling DTR while a persistent arm session owns the board.
                self.ser = serial.Serial(
                    self.port, self.baud, timeout=0.2, exclusive=True)
            except Exception as exc:
                raise RuntimeError(
                    f"로봇팔 시리얼 포트 {self.port}을(를) 열 수 없습니다: "
                    f"{exc}. Arduino Serial Monitor를 닫았는지 확인하세요."
                ) from exc
            time.sleep(2.0)  # wait out the auto-reset on connect
            self._drain()
            print(f"[로봇팔] {self.port}에 {self.baud}bps로 연결했습니다.")
            try:
                self.verify_firmware_home_pose()
            except Exception:
                self.close()
                raise

    def _drain(self):
        if self.ser:
            while self.ser.in_waiting:
                self.ser.readline()

    def send_angles(self, angles):
        """angles: list of 6 ints (0..180), or -1 to hold a joint."""
        if len(angles) != config.N_JOINTS:
            raise ValueError(
                f"관절값 {config.N_JOINTS}개가 필요하지만 "
                f"{len(angles)}개를 받았습니다.")
        values = []
        for i, value in enumerate(angles):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{i + 1}번 관절 각도는 숫자여야 합니다.")
            if not float(value).is_integer():
                raise ValueError(
                    f"{i + 1}번 관절 각도는 정수여야 합니다. 받은 값: {value}")
            value = int(value)
            if value != -1 and not (config.SERVO_MIN[i] <= value <= config.SERVO_MAX[i]):
                raise ValueError(
                    f"{i + 1}번 관절 각도 {value}°가 안전 범위 "
                    f"[{config.SERVO_MIN[i]}, {config.SERVO_MAX[i]}]°를 "
                    "벗어났습니다.")
            values.append(value)
        line = "A " + " ".join(str(a) for a in values) + "\n"
        if self.mock:
            print(f"[로봇팔 모의 명령] {line.strip()}")
            return "OK"
        self.ser.write(line.encode())
        reply = self._wait_for({"OK"}, timeout=2.0)
        return reply

    def _wait_for(self, expected, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(f"로봇팔 펌웨어가 명령을 거부했습니다: {line}")
            if line in expected:
                return line
        raise TimeoutError(
            f"로봇팔에서 {sorted(expected)} 응답을 기다리다 시간이 "
            "초과되었습니다. 전원과 USB 연결을 확인하세요.")

    def wait_done(self, timeout=8.0):
        """Block until firmware reports DONE (all joints reached) or timeout."""
        if self.mock:
            time.sleep(0.3)
            return True
        self._wait_for({"DONE"}, timeout=timeout)
        return True

    def ping(self):
        if self.mock:
            return True
        self.ser.write(b"P\n")
        return self._wait_for({"PONG"}, timeout=2.0) == "PONG"

    def status(self):
        """Read the firmware's current commanded servo angles without motion."""
        if self.mock:
            return list(config.HOME_POSE)
        self.ser.write(b"S\n")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(
                    f"로봇팔 펌웨어가 상태 요청을 거부했습니다: {line}")
            if line.startswith("C"):
                return parse_status_line(line)
        raise TimeoutError("로봇팔 상태 응답을 기다리다 시간이 초과되었습니다.")

    def firmware_home_pose(self):
        """Return the HOME pose compiled into the connected firmware."""
        if self.mock:
            return list(config.HOME_POSE)
        self.ser.write(b"H\n")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(
                    "연결된 Uno 펌웨어가 HOME 자세를 알려주지 못합니다. "
                    "OPEN_FIRMWARE.bat으로 현재 스케치를 업로드하세요.")
            if line.startswith("H"):
                try:
                    return parse_home_pose_line(line)
                except ValueError as exc:
                    raise RuntimeError(
                        "연결된 Uno가 다른 관절 배치 또는 통신 규칙을 "
                        "사용합니다. OPEN_FIRMWARE.bat으로 현재 6서보 "
                        "스케치를 업로드하세요.") from exc
        raise TimeoutError(
            "로봇팔 펌웨어가 HOME 자세를 보내지 않았습니다. "
            "OPEN_FIRMWARE.bat으로 현재 스케치를 업로드하세요.")

    def verify_firmware_home_pose(self):
        """Require the connected firmware to match local ``home_pose.h``."""
        return assert_home_pose_match(self.firmware_home_pose())

    def grip_feedback(self):
        """Return optional A0 pressure/current feedback, or None if uninstalled."""
        if self.mock:
            return None
        self.ser.write(b"F\n")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(
                    f"로봇팔 펌웨어가 집게 피드백 요청을 거부했습니다: {line}")
            if line.startswith("F "):
                try:
                    value = int(line.split()[1])
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"집게 피드백 형식이 잘못되었습니다: {line!r}") from exc
                if value == -1:
                    return None
                if not 0 <= value <= 1023:
                    raise ValueError(
                        f"집게 피드백 {value}가 ADC 범위를 벗어났습니다.")
                return value
        raise TimeoutError("집게 피드백을 기다리다 시간이 초과되었습니다.")

    def ultrasonic_distance_mm(self, samples=None):
        """Return a median wrist-sonar distance, or ``None`` without enough echoes.

        The sensor is sampled only on request so a missing echo cannot
        continuously block the firmware's servo loop. A single query may block
        for the firmware's bounded 25 ms echo timeout.
        """
        samples = config.ULTRASONIC_SAMPLES if samples is None else int(samples)
        if not 1 <= samples <= 9:
            raise ValueError("초음파 측정 횟수는 1~9 사이여야 합니다.")
        if self.mock:
            return None
        valid = []
        for index in range(samples):
            self.ser.write(b"D\n")
            deadline = time.monotonic() + 1.0
            reading_received = False
            while time.monotonic() < deadline:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode(errors="replace").strip()
                if line.startswith("ERR"):
                    raise RuntimeError(
                        f"로봇팔 펌웨어가 초음파 요청을 거부했습니다: {line}")
                # Streamed targets can finish while a distance request is in
                # flight, leaving an asynchronous ``DONE`` in the same serial
                # input. Only the exact distance record prefix is a reading.
                if line.startswith("D "):
                    value = parse_distance_line(line)
                    reading_received = True
                    if value is not None:
                        valid.append(value)
                    break
            if not reading_received:
                raise TimeoutError(
                    "초음파 거리 응답을 기다리다 시간이 초과되었습니다. "
                    "TRIG D7, ECHO D6 배선을 확인하세요.")
            if index + 1 < samples:
                time.sleep(config.ULTRASONIC_SAMPLE_INTERVAL_S)
        required = min(samples, config.ULTRASONIC_MIN_VALID_SAMPLES)
        if len(valid) < required:
            return None
        return float(statistics.median(valid))

    def gripper(self, open_=True):
        a = [-1] * config.N_JOINTS
        a[config.J_GRIP] = config.GRIP_OPEN if open_ else config.GRIP_CLOSED
        return self.send_angles(a)

    def stop_motion(self):
        """Cancel the remaining firmware slew and hold the last written pose."""
        if self.mock:
            return True
        self.ser.write(b"X\n")
        return self._wait_for({"STOPPED"}, timeout=2.0) == "STOPPED"

    def home(self):
        return self.send_angles(config.HOME_POSE)

    def close(self):
        if self.ser:
            self.ser.close()


if __name__ == "__main__":
    arm = ArmSerial()
    arm.home()
    arm.wait_done()
    arm.gripper(open_=True)
    arm.wait_done()
    arm.gripper(open_=False)
    arm.wait_done()
    arm.close()
