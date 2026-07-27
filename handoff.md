# brainToArm 전체 인수인계서

작성 시각: 2026-07-27 12:47 KST  
저장소: `/Users/watson/Desktop/programming/playground/brainToArm`  
현재 브랜치: `main`  
이 문서는 Claude가 이전 대화 없이도 작업을 이어받을 수 있도록 작성했다.

> 가장 중요한 원칙: 이 문서에서 **실제 검증됨**, **코드만 구현됨**,
> **아직 미완성**을 의도적으로 구분한다. 화면에 물체가 보인다는 사실을
> 잡을 수 있는 거리로 오해하거나, mock 테스트 성공을 실제 하드웨어 성공으로
> 과장하지 말 것.

## 1. 프로젝트의 최종 목적

PolyG-I 8채널 EEG와 카메라가 달린 6축 로봇팔을 한 MacBook에서 구동하는
shared-autonomy 시스템이다.

- 로봇은 카메라로 물체를 탐색하고 집어서 옮긴다.
- 사용자는 로봇의 행동을 보고, 잘못됐다고 느낄 때 발생하는 ErrP
  (Error-Related Potential)로 선택을 교정한다.
- 인지 부하는 TAR(theta/alpha ratio)로 계속 계산한다.
- TAR가 높으면 로봇 자율성 비중을 높이고 ErrP 적용 빈도/비중을 줄인다.
- TAR가 낮으면 인간/ErrP 비중을 높이고 ErrP 적용 빈도를 늘린다.
- ErrP 계산 자체는 계속하며, 매우 강한 ErrP는 TAR와 무관하게 veto로 남긴다.

의도는 사용자가 뇌파로 로봇의 모든 관절을 직접 조종하는 것이 아니다.
로봇이 대부분의 탐색·경로·집기를 수행하고 사람은 오류 판단을 제공한다.

## 2. 사용자 요구와 작업 규칙

다음 규칙은 계속 지켜야 한다.

1. 변경한 내용은 전부 `PATCH_NOTES.md`에 패치 단위로 기록한다.
2. 패치마다 테스트하고 Git commit을 남긴다.
3. 기존 사용자 변경을 덮어쓰거나 파괴적으로 되돌리지 않는다.
4. 하드웨어 수치가 불확실하면 화면상 추측을 사실로 쓰지 않는다.
5. 한 RGB 카메라의 2D 정렬만으로 실제 깊이 또는 접촉을 선언하지 않는다.
6. 물체 색·크기·배경·대회 장소가 바뀔 수 있다. 현재 장면에 과적합한
   좌표, 배경 사진, 흰 종이, 보정판을 필수 조건으로 만들지 않는다.
7. 물체는 로봇팔과 같은 고정된 바닥 평면에 놓인다는 조건은 사용할 수 있다.
8. 압력 센서, 전류 센서, 두 번째 카메라는 없는 것으로 설계한다.
9. 웹캠 화면과 인식 결과는 사용자가 항상 볼 수 있게 유지하는 편이 좋다.
10. 로봇팔 동작 전에 사람의 손, 케이블, 물체가 이동 경로에서 빠졌는지
    사용자에게 짧게 알린다.

민감정보(예: 대화에 등장한 관리자 암호)는 이 저장소나 문서에 기록하지 말 것.
이 프로젝트 작업에 sudo는 필요하지 않다.

## 3. 현재 Git 상태와 중요한 이력

이 문서 작성 직전 worktree는 clean이었다. 최신 구현 commit은 다음과 같다.

```text
30083ab feat: add persistent floor-plane arm control
5e5ee03 feat: add visual gripper contact sensing
493e7de fix: isolate wrist markers in rigid lower-frame ROI
bbf2956 fix: calibrate yellow target on white workspace
42248c3 fix: separate raw camera input from annotated preview
8955f4f fix: set gripper full-close endpoint to 180 degrees
dd89b02 fix: constrain wrist gripper detection by mount geometry
ea82016 fix: sync latest arm home and limits
b967136 feat: lock wrist target to vivid yellow
03b5ac2 feat: add collision-checked wrist target search
981de0e feat: add wrist-camera gripper perception
0c0cb6d feat: add TAR-adaptive shared autonomy
93d24ae fix: swap gripper and wrist roll mapping
f7f209b refactor: migrate arm to six active servos
20fb78a feat: verify and sync arm home firmware
```

전체 변경 역사는 `PATCH_NOTES.md`, 현재 사용자 문서는 `README.md`, PolyG-I의
정확한 통신 조사 내용은 `docs/EEG_DEVICE_COMMS.md`를 우선 참고한다.

주의: 오래된 README/PATCH_NOTES 문단에는 당시 배선이나 설정이 역사 기록으로
남아 있다. **현재 동작의 source of truth는 최신 `laptop/config.py`,
`firmware/arm_controller/home_pose.h`, 최신 패치 섹션이다.**

## 4. 하드웨어 구성

### 4.1 로봇팔과 Arduino Uno

현재 6개 servo를 사용한다. 과거의 사용하지 않던 3번 슬롯을 제거하고 뒤 관절을
앞으로 당겼으며, 이후 5/6번 배선 의미도 교정했다. 최종 매핑은 다음과 같다.

| 논리 모터 | Uno pin | 이름 | 실제 역할 | 현재 범위 |
|---:|---:|---|---|---|
| 1 | D13 | `base` | 바닥 Z축 중심 회전 | 0..180° |
| 2 | D12 | `shoulder` | 첫 번째 큰 관절 | 0..150° |
| 3 | D11 | `elbow` | 두 번째 큰 관절 | 0..180° |
| 4 | D10 | `wrist_pitch` | 손목 상하 각도 | 130..180° |
| 5 | D9 | `gripper` | 집게 열기/닫기 | 90..180° |
| 6 | D8 | `wrist_roll` | 집게 자체 회전 | 0..180° |

중요한 실제 수치:

- 모터 1 base는 한때 고장이었으나 이후 복구됐다.
- 모터 5: `90° = 완전 열림`, `180° = 완전 닫힘`.
- 모터 6: `170° = 집게가 정확히 수평`. `180°`는 수평을 지나 더 돌아간다.
- 모터 4는 현재 안전상 `130..180°`만 허용한다.
- 현재 firmware HOME은 `[90, 70, 90, 140, 170, 170]`이다.
- HOME의 gripper 170°는 거의 닫힌 초기 자세이지 완전 닫힘 endpoint가 아니다.

HOME의 유일한 원본은 `firmware/arm_controller/home_pose.h`다.

```c
#define ARM_HOME_SERVO_1 90
#define ARM_HOME_SERVO_2 70
#define ARM_HOME_SERVO_3 90
#define ARM_HOME_SERVO_4 140
#define ARM_HOME_SERVO_5 170
#define ARM_HOME_SERVO_6 170
```

HOME 값을 바꾸면 sketch를 다시 업로드해야 한다.

```bash
python3 laptop/arm_jog.py --upload
```

업로드/직렬 포트 open은 Uno를 reset하고 즉시 HOME으로 움직이므로 작업 공간을
먼저 비워야 한다.

현재 Uno CH340은 보통 `/dev/cu.usbserial-110`으로 잡힌다. `ARM_PORT="auto"`가
사용하며, ESP32 CP2102였던 `/dev/cu.usbserial-0001`은 자동 탐색에서 제외한다.

### 4.2 손목 웹캠

최종 선택은 `AVerMedia PW315` USB 웹캠이다.

- 1280x720, 요청 FPS 30.
- macOS AVFoundation 장치 이름으로 ffmpeg를 통해 연다.
- iPhone Continuity Camera나 FaceTime 카메라의 바뀌는 숫자 index를 쓰지 않는다.
- 웹캠은 로봇팔 끝에 달려 집게와 물체를 함께 본다.
- 집게 왼쪽 손가락에는 파란 테이프, 오른쪽에는 빨간 테이프가 붙어 있다.
- 카메라 화면 기준 파란색이 왼쪽, 빨간색이 오른쪽이어야 정상이다.

ESP32 + OV2640도 연결하고 진단 firmware/캡처 경로까지 만들었지만 해상도·색·전송
안정성 면에서 USB 웹캠이 훨씬 나아 최종 eye-in-hand 카메라는 PW315를 사용한다.
OV2640 경로는 `firmware/esp32_camera_diagnostic/`와
`laptop/capture_esp32_camera.py`에 남아 있다.

USB hub에서 Uno와 웹캠을 동시에 썼을 때 웹캠이 멈춘 적이 있다. 프레임 mtime이
갱신되지 않으면 powered hub 또는 Mac 직접 연결을 우선 확인한다. 단순히 장치명이
보이는 것과 실제 프레임이 계속 오는 것을 구분한다.

### 4.3 PolyG-I EEG

- 제품: LAXTHA PolyG-I.
- VID/PID: `0x0F1F / 0x0010`.
- vendor-defined USB HID, usage page `0xFF00`.
- COM/serial 장치가 아니다.
- 물리 입력은 16개이며 첫 8개가 EEG 1..8이다.
- HID input report 1개는 1,024 bytes = 512 words = 32시간행 x 16채널.
- selector 8은 정확히 `2^8 = 256 Hz`.
- 실제 반복 측정은 약 255.93..256.22 rows/s였다.

TeleScan은 Windows 앱이라 이 Mac에서 장치 그래프를 직접 실행하는 방식은 포기했다.
대신 설치된 `LXSM-D1WD10.dll`, 공식 개발 문서, 실제 장치 A/B 측정으로 프로토콜을
복원해 `laptop/polyg_hid.py`에서 native `hidapi`로 구현했다.

정확한 시작 순서(8-byte payload의 앞 3 bytes)는 다음과 같다.

```text
01 00 00  STOP
05 10 00  physical channels = 16
04 08 00  sample selector = 8 = 256 Hz
0B 06 00  EEG source group PGA index = 6
01 01 00  START
```

cleanup은 항상 `01 00 00` STOP을 보낸다.

ADC word 복원식:

```python
count = (high_byte - 0x80) * 256 + (low_byte & 0xFE)
```

low byte bit 0은 ADC가 아니라 marking bit다.

DLL에서 확인한 전압 계수는 약 `-1.25 / 32768 V/count`다. 이는 정확한
**ADC 입력 mV**이지 전극 입력 µV가 아니다. 전치증폭 전체 gain이 독립적으로
교정되지 않았으므로 UI나 논문에서 µV라고 부르면 안 된다.

## 5. EEG 처리와 UI

실시간 대시보드는 다음 명령으로 실행한다.

```bash
python3 laptop/eeg_dashboard.py
```

- React UI: `http://localhost:3000`
- Python device API: `http://127.0.0.1:8765`
- Python 한 프로세스가 HID를 독점해 시작/정지를 책임진다.
- CSV는 ignored local `recordings/` 아래 저장된다.

구현 기능:

- 8채널 실시간 파형.
- 모든 채널에 공통인 고정 Y scale과 0 mV 기준선.
- auto-scale로 진폭 기준이 계속 바뀌지 않는다.
- raw ADC count, raw ADC-input mV, filtered ADC-input mV를 구분한다.
- stateful 60 Hz notch(Q=30) + 0.5..45 Hz Butterworth band-pass.
- HID가 32행 묶음으로 와도 display buffer와 browser animation으로 부드럽게 그린다.
- 보간은 화면에만 적용하고 CSV 원본 표본을 만들거나 변조하지 않는다.
- Hann 256-point one-sided PSD, 고정 -80..40 dB 축.
- Delta/Theta/Alpha/Beta/Gamma power 비율.
- 최근 2초 RMS, peak-to-peak, DC offset, rail clipping.
- 측정 시작/정지, pause/view, event marker, CSV 기록.

품질 수치는 전극 임피던스나 임상 판정이 아니다.

## 6. ErrP와 인지 부하 정의

### 6.1 ErrP

- 사용 채널: 현재 CH1..CH8 전부 (`ERRP_CHANNELS = range(8)`).
- 행동 onset 전 0.2초 baseline + onset 후 0.8초 epoch.
- 1..10 Hz band-pass.
- 약 250..450 ms 부근의 지속적인 음의 deflection을 본다.
- baseline backend는 세션 휴식 노이즈 대비 z-score로 scale invariant하게 동작한다.
- 최종 배포는 참가자별 correct/error labeled trial을 모아 model backend를 쓰는 것이 목표다.

```bash
python3 laptop/record_errp.py data/errp --trials 40
python3 laptop/errp_train.py data/errp
```

그 후 `ERRP_BACKEND="model"`로 바꾼다. 현재 default는 `baseline`이다.

### 6.2 인지 부하 TAR

사용자가 정한 정의:

- CH1..CH4의 theta(4..8 Hz) power를 각각 구해 평균한다.
- CH8의 alpha(8..13 Hz) power를 구한다.
- `TAR = mean(theta CH1..CH4) / alpha CH8`.
- 세션 시작 시 8초 휴식 baseline을 측정한다.
- 최종 부하 값은 `(current TAR - resting TAR) / resting TAR`.
- 2초 window, 1초마다 갱신, Welch PSD, EMA 0.30.

자율성 배분:

- base robot weight 0.50, clamp 0.20..0.80.
- 휴식 대비 ±10%는 deadband.
- TAR 상승: robot weight 증가, ErrP threshold 증가, ErrP 적용 stride 증가.
- TAR 하락: human weight 증가, ErrP를 매 checkpoint에 더 자주 적용.
- ErrP 확률 계산은 행동 사건마다 계속한다.
- `p_error >= 0.90`은 load와 무관하게 override.
- flat/invalid PSD는 최소 robot authority 쪽으로 fail-safe한다.

중요: 사용자가 처음 표현한 “TAR가 높을수록 오류 전위 계산 빈도 감소”는 코드에서
“계산은 계속하되 최종 의사결정에 적용하는 checkpoint 빈도 감소”로 구현했다.

`EEG_CONFIG_VERIFIED=False`는 아직 유지한다. USB transport와 decoder는 실제로
검증됐지만, 실제 전극 장착 상태의 montage/reference/ground, 채널 반응,
참가자별 ErrP 정확도는 별도 검증이 필요하기 때문이다.

## 7. 로봇팔 firmware와 serial protocol

Firmware: `firmware/arm_controller/arm_controller.ino`.

```text
A a1 a2 a3 a4 a5 a6   six target angles; -1 means hold
P                     ping -> PONG
S                     status -> C a1..a6
H                     compiled HOME -> H a1..a6
F                     optional A0 feedback -> F -1 (현재 센서 없음)
```

- firmware가 각 joint별 safe range로 clamp한다.
- 15 ms tick마다 최대 1.5°씩 slew한다.
- software target에 도달하면 `DONE`을 보낸다.
- 중요한 함정: `DONE`은 servo software slew 완료이지 실제 hobby servo shaft와
  링크가 완전히 정착했다는 뜻이 아니다. 카메라 측정 전 약 1.8..2.0초 실제
  기계 정착 시간을 두고 새 프레임 여러 개를 버려야 한다.

## 8. Uno 자동 reset 문제와 상주 세션

Uno serial port를 열 때 DTR이 토글되어 보드가 HOME으로 reset된다. 과거 진단을
짧은 Python script 여러 개로 실행하면서 매번 다음 비효율이 발생했다.

```text
현재 자세 -> 전원/serial reset -> HOME -> 명령한 자세
```

이를 해결한 것이 `laptop/arm_session.py`다. 한 프로세스가 serial을 단 한 번 열고
Unix socket `data/runtime/arm_session.sock`을 통해 이후 명령을 받는다.
`ArmSerial`은 `exclusive=True`로 열려 두 번째 프로세스의 실수로 인한 reset도 막는다.

시작:

```bash
# 딱 한 번 reset 후 안전한 floor hover로 이동하고 계속 살아 있음
python3 laptop/arm_session.py serve --floor hover --elbow 90
```

다른 terminal/client:

```bash
python3 laptop/arm_session.py status
python3 laptop/arm_session.py floor hover 90
python3 laptop/arm_session.py floor grasp 90
python3 laptop/arm_session.py move 90 124 90 180 90 170
python3 laptop/arm_session.py shutdown
```

규칙:

- 상주 session 실행 중 `ArmSerial()`을 직접 여는 다른 script를 실행하지 않는다.
- `arm_jog.py`, 기존 `visual_contact.py --calibrate-empty`, one-shot diagnostic는
  직접 Uno를 열므로 session과 동시에 쓰면 안 된다.
- 향후 자동 집기 controller도 반드시 `ArmSessionClient`를 사용하도록 만든다.
- 죽은 process가 남긴 socket 파일은 server 시작 시 ping 실패를 확인한 후 자동 삭제한다.

## 9. 손목 카메라 실시간 인식

실행:

```bash
python3 laptop/wrist_vision.py --live
```

조작:

- left click: 클릭한 물체의 hue를 runtime target으로 lock.
- right click: target lock을 해제하고 자동 색 모드로 복귀.
- `S`: annotated snapshot 저장.
- `Q` 또는 Escape: 종료.

경로:

- machine perception용 raw: `data/vision/wrist_camera_latest_raw.jpg`
- 사용자용 annotation: `data/vision/wrist_camera_latest.jpg`

반드시 raw와 preview를 분리한다. 과거 controller가 annotated preview를 다시 읽어
노란 gripper-center cross를 실제 노란 물체로 오인한 버그가 있었다.

집게 인식:

- 파란 tape/빨간 tape를 독립 검출한다.
- rigid mount 때문에 marker 탐색은 lower ROI
  `(x=0.22..0.75, y=0.72..1.0)`로 제한한다.
- blue-left/red-right 순서, 크기, fill, aspect, 분리거리, open/closed profile,
  하단 고정 위치를 모두 확인한다.
- 장면의 다른 빨강/파랑 물체가 집게로 바뀌는 것을 막는다.

현재 기본 target preset은 실제 노란 물체를 흰 종이 위 mounted PW315로 측정한
OpenCV HSV `H=19..27, S>=45, V>=200`이다. `config.py`가 최신 원본이다.
오래된 README에 `H=22..38, S>=140`가 남아 있으면 그 문장은 stale이다.

하지만 최종 시스템은 노란색을 가정하면 안 된다. runtime click 또는 FastSAM으로
그 실행의 target을 선택하고 추적해야 한다. 빨강/파랑은 finger marker와 충돌하므로
target으로 쓸 때 별도 identity 처리가 필요하다.

## 10. 카메라로 깊이를 잘못 추정하지 않는 원칙

한 장의 uncalibrated RGB frame에서 다음은 구분해야 한다.

- 물체가 두 손가락 사이에 보임: 방향/투영 정렬 정보.
- 물체가 실제 손가락이 닿는 깊이에 있음: 별개의 3D/접촉 문제.

따라서 “화면상 집게 중앙에 들어왔다”만으로 닫거나 lift하면 안 된다. 물체 크기는
매 실행마다 바뀔 수 있어 절대 거리 threshold로 쓰면 안 된다. Depth Anything 같은
monocular model도 상대 depth 보조 정보일 뿐 안전한 metric contact gate가 아니다.

현재 채택한 방식:

1. 물체가 놓이는 **바닥은 로봇 기준으로 고정**돼 있다.
2. 로봇 관절의 실제로 성공한 floor reference로 Z를 결정한다.
3. 바닥 높이를 유지하는 shoulder/elbow 동시 벡터로 전후 X를 이동한다.
4. runtime target은 이 바닥 경로에서 수평 위치를 찾는 데만 쓴다.
5. 마지막에는 시각 jaw obstruction과 lift 후 target 동행으로 집기를 검증한다.

## 11. 검증된 바닥 자세와 수평 벡터

과거 측면 카메라로 실제 집기/들기/놓기까지 성공한 기준:

```text
base=90, elbow=90, shoulder=140,
wrist_pitch=180, wrist_roll 당시 180, gripper open90 -> close180
```

이후 사용자 확인으로 wrist level은 180이 아니라 **170**으로 수정했다. 바닥에
더 확실히 닿는 실제 grasp shoulder offset은 +2°여서 현재 floor reference는:

```text
floor hover: [90, 124, 90, 180, 90, 170]
floor grasp: [90, 142, 90, 180, 90, 170]
```

과거 실제 측면 영상으로 얻은 local Jacobian:

```text
shoulder +1° -> vertical 약 +11 px
elbow +1°    -> horizontal 약 -7 px, vertical 약 +6 px
```

따라서 높이를 상쇄하는 ground-plane tangent는:

```text
d(shoulder) / d(elbow) = -6 / 11
shoulder(elbow, level) = shoulder_ref(level) - (6/11) * (elbow - 90)
```

`laptop/floor_motion.py`가 이 식을 구현한다.

- calibrated elbow 범위: 78..110°.
- 기본 waypoint step: 4°.
- hover와 grasp는 shoulder가 18° 차이 나는 평행 곡선이다.
- 모든 waypoint는 여섯 servo 전체 명령으로 만들어진다.

2026-07-25 실물 재현:

- `[90,124,90,180,90,170]`에서 카메라가 바닥을 수직으로 보고 노란 target 중심이
  x 약 600 px, jaw 기준 x 약 606 px로 거의 맞았다.
- `[90,142,90,180,90,170]`으로 reset 없이 내려가자 target이 화면을 크게 채웠다.
- 이는 바닥 근접점 재현에는 성공했다는 뜻이다.
- 하지만 바로 다음 gripper close 직전에 작업이 중단됐다. **손목 카메라 기반
  floor grasp/close/lift 성공은 아직 선언하면 안 된다.**

## 12. 시각 jaw contact 보정

servo에 torque/current/position feedback이 없으므로 카메라의 두 finger tape 간격을
사용한다. `laptop/visual_contact.py`에 빈 집게 baseline과 판정이 있다.

로컬 파일(ignored): `data/calibration/wrist_jaw_baseline.json`

PW315 1280x720에서 실제 빈 집게 보정값:

| gripper command | marker separation median | MAD |
|---:|---:|---:|
| 90° | 288.0 px | 0.4 px |
| 110° | 279.5 px | 0.3 px |
| 130° | 256.8 px | 0.4 px |
| 150° | 201.1 px | 0.1 px |
| 170° | 125.1 px | 0.7 px |
| 180° | 89.2 px | 0.2 px |

판정:

- command 140° 미만에서는 contact를 평가하지 않는다.
- observed opening이 empty expected보다 `max(7px, 5*MAD)` 이상 넓으면 `CONTACT`.
- baseline과 같으면 `FREE`.
- marker 누락, 범위 밖, 예상보다 비정상적으로 좁으면 `UNKNOWN`.
- `UNKNOWN`에서는 lift하면 안 된다.

첫 보정 시 firmware `DONE` 직후 실제 linkage가 아직 움직이는 바람에 90° 표본이
닫힌 모습으로 잡혀 non-monotonic으로 거부됐다. 이후 1.8초 settle + 새 frame discard를
추가해 위 monotonic baseline을 얻었다.

실제 테스트:

- 자세 `[90,80,90,180,90,170]`에서 target은 화면상 jaw 중심과 약 10 px 차이였다.
- 180° close 후 observed 약 85.0 px, expected 89.2 px로 `FREE`였다.
- 즉 화면상 정렬은 맞지만 실제 깊이는 아니었다는 사용자의 지적이 실험으로 확인됐다.

주의: floor grasp에서 target이 너무 가까워 tape나 target을 가릴 수 있다. marker가
안 보이면 현재 contact detector는 의도적으로 `UNKNOWN`이다. 이를 무시해 `CONTACT`로
바꾸지 말고, near-field marker 추적 또는 다른 lift verification을 추가해야 한다.

## 13. 이전에 실제로 성공한 것과 아직 미완성인 것

### 실제 하드웨어에서 성공

- Uno 6-servo firmware compile/upload/serial status/home 검증.
- base motor 복구 후 bounded 움직임.
- gripper 90 open / 180 close와 wrist roll 170 level 확인.
- 과거 MacBook 측면 카메라 기반 planar pick:
  물체 검출 -> 접근 -> 닫기 -> 물체가 함께 올라감을 영상 확인 -> 옆으로 이동 ->
  놓기까지 한 번 완전 성공.
- AVerMedia PW315 named capture 1280x720.
- blue/red finger marker 검출.
- raw/annotated frame 분리.
- 빈 jaw curve 실측 및 contact=`FREE` 실험.
- floor hover/grasp 자세를 손목 카메라로 재현.
- PolyG-I native HID 시작/수신/정지, decoder, 256 Hz cadence.
- EEG dashboard 실시간 파형, 고정 scale, 필터, PSD, CSV.

### 코드/테스트로 구현됐지만 물리 검증 gate가 남음

- generic 3D IK와 collision-checked wrist search.
- full shared-autonomy orchestrator.
- TAR-adaptive ErrP 적용.
- runtime wrist target alignment.
- persistent arm session + floor curve.

### 아직 끝나지 않은 핵심

1. 손목 카메라만으로 runtime target을 고르고 floor curve를 따라 이동.
2. near-field에서 target/marker가 커지거나 가려져도 identity 유지.
3. floor grasp 위치에서 close.
4. jaw obstruction `CONTACT` 또는 동등하게 엄격한 물리 증거 확인.
5. floor hover로 lift한 후 target이 집게와 함께 움직이는지 검증.
6. 실패하면 즉시 다시 열고 hover/retreat.
7. 성공 후 가까운 안전 위치에 놓기.
8. 위 전체를 `ArmSessionClient` 한 연결로 수행.
9. 이후 base motor를 포함한 더 넓은 3D workspace로 확장.
10. 참가자별 실제 ErrP dataset과 EEG montage 검증.

## 14. 실패했던 접근과 반복하지 말아야 할 것

- annotated preview를 perception input으로 사용: UI의 노란 cross를 target으로 오인했다.
- target이 jaw midpoint에 있으면 잡을 수 있다고 가정: 깊이가 없어 실제 close는 FREE였다.
- 현재 노란색/흰 종이/배경 사진을 영구 안전 조건으로 사용: 대회 환경에서 깨진다.
- target apparent area만으로 절대 거리를 선언: 물체 크기가 바뀌면 깨진다.
- 짧은 script마다 Uno serial open/close: 매번 HOME reset으로 비효율적이고 위험하다.
- firmware `DONE` 즉시 카메라 측정: 실제 servo/linkage가 아직 정착하지 않았다.
- 모터 6을 180° 수평으로 사용: 실제 수평은 170°다.
- FastSAM/YOLO가 있다고 깊이와 접촉까지 해결된다고 가정: segmentation은 identity/영역
  문제만 해결하며 metric depth나 grasp contact를 보장하지 않는다.
- unverified `WRIST_SEARCH_KINEMATICS_VERIFIED=False` gate를 강제로 켜고 sweeping:
  link length/offset/direction이 완전 검증되지 않아 충돌 위험이 있다.
- sensor가 없는 servo에서 torque/current 값을 읽을 수 있다고 가정: 현재 A0 feedback은
  `F -1`; 압력/전류 센서는 없다.

## 15. 현재 프로세스/장치 상태 (2026-07-27 12:47 KST 스냅샷)

이 항목은 시간이 지나면 달라지므로 반드시 다시 확인한다.

- Uno serial device: `/dev/cu.usbserial-110` 인식됨.
- AVerMedia PW315: AVFoundation에서 인식됨.
- `python3 laptop/wrist_vision.py --live` 실행 중.
- 해당 ffmpeg child가 `AVerMedia PW315:none`을 1280x720/30fps로 읽는 중.
- raw frame mtime은 실시간 갱신 중.
- 최신 관찰: frame quality OK, blue/red markers 발견,
  gripper opening 약 111.7 px, target 없음.
- 카메라는 현재 바닥 target이 아니라 방/블라인드 방향을 보고 있다.
- arm session server는 실행 중이 아니다.
- `data/runtime/arm_session.sock`은 7월 25일 죽은 process가 남긴 stale socket이다.
  `arm_session.py serve`가 시작할 때 자동으로 검증 후 제거한다.
- Uno의 실제 current pose는 포트를 열지 않아 조회하지 않았다. 장치 재연결 직후라면
  HOME일 가능성이 크지만 추측을 상태값으로 사용하지 말 것. 상주 session 시작 시
  한 번 reset된다는 사실을 사용자에게 알리고 workspace를 비운 후 시작한다.

현재 실시간 카메라를 확인하는 안전한 명령:

```bash
stat -f '%Sm %z' data/vision/wrist_camera_latest_raw.jpg
python3 laptop/wrist_vision.py --snapshot
```

두 번째 `--snapshot`은 이미 live process가 카메라를 독점 중이면 실패할 수 있으므로,
그때는 raw 파일의 mtime과 이미지만 본다. 카메라 process를 중복 실행하지 말 것.

## 16. 다음 Claude가 바로 해야 할 권장 순서

### A. 상태 확인

```bash
cd /Users/watson/Desktop/programming/playground/brainToArm
git status --short
git log -5 --oneline
pgrep -af 'wrist_vision.py|arm_session.py|ffmpeg.*avfoundation'
ls /dev/cu.usbserial* /dev/cu.usbmodem* 2>/dev/null
stat -f '%Sm %z' data/vision/wrist_camera_latest_raw.jpg
python3 laptop/test_pipeline.py
```

### B. Uno 상주 세션을 딱 한 번 시작

사용자에게 손/케이블을 치우라고 말한 뒤:

```bash
python3 laptop/arm_session.py serve --floor hover --elbow 90
```

이 process를 종료하지 말고 별도 terminal/client에서:

```bash
python3 laptop/arm_session.py status
```

같은 status를 여러 번 호출해도 pose가 HOME으로 돌아가지 않아야 한다.

### C. 다음 코드 패치

`laptop/floor_grasp.py` 같은 하나의 fail-closed state machine을 만드는 것이 좋다.
반드시 `ArmSessionClient`를 사용하고 direct `ArmSerial`을 열지 않는다.

권장 상태:

```text
IDLE
 -> TARGET_SELECTED_AT_HOVER
 -> FLOOR_X_ALIGN (floor hover curve, elbow/shoulder simultaneous)
 -> DESCEND_TO_KNOWN_FLOOR (same elbow, hover -> grasp)
 -> CLOSE_PROBE
 -> CONTACT or UNKNOWN/FREE
 -> LIFT_TO_HOVER
 -> VERIFY_TARGET_MOVES_WITH_GRIPPER
 -> HOLD / PLACE

모든 오류:
 -> OPEN
 -> HOVER
 -> STOP + 명확한 reason
```

설계 세부:

- target은 live click 또는 FastSAM mask로 runtime 선택한다.
- hover에서 target identity를 저장하고 optical tracking/FastSAM으로 descent 중 유지한다.
- target 색/면적의 절대값은 접촉 gate가 아니다.
- floor X 위치는 `floor_motion.py`의 elbow/shoulder curve만 사용한다.
- base는 우선 90° planar로 고정해 한 방향 집기를 완성한 후 확장한다.
- 마지막 descend는 검증된 shoulder 124 -> 142 parallel level change다.
- near-field target이 frame의 18%를 넘는 현재 일반 필터 때문에 사라진다.
  controller state가 `DESCEND_NEAR_FIELD`일 때 이전 target과의 연속성으로 큰 mask를
  허용하되, 전역 target detector의 제한을 무작정 풀어 배경을 target으로 만들지 않는다.
- close 후 marker가 보이면 `JawBaseline.assess()`를 사용한다.
- marker가 가리면 `UNKNOWN`으로 두고 lift 금지 또는 별도의 엄격한 visual proof를 만든다.
- lift verification은 target의 화면 위치 하나가 아니라, close 전 identity가 hover로
  올라갈 때 gripper와 함께 coherent motion을 보이는지 확인한다.
- 각 move 후 firmware DONE + 1.8..2.0초 settle + fresh frames를 사용한다.
- 모든 physical routine은 하나의 persistent session 안에서 끝낸다.

### D. 물리 시험

가장 마지막으로 관찰했던 조건에서 물체는 elbow=90 floor grasp 시 화면 중앙을 크게
채웠다. 그러나 현재 물체/팔 위치가 바뀌었으므로 이전 이미지를 현재 상태로 간주하지
말고 hover에서 다시 target을 선택한다.

첫 실제 close는 다음 조건을 전부 만족할 때만 한다.

- arm session status가 예상 hover/grasp pose와 일치.
- raw frame이 fresh.
- hover에서 target identity가 안정적으로 여러 frame 유지.
- target이 floor X path로 정렬.
- 사람 손과 cable이 경로 밖.
- 실패 시 즉시 open + hover할 recovery 코드가 이미 실행 가능.

## 17. 구성 gate의 의미

현재 주요 gate:

- `ARM_MOCK=False`: Uno를 실제로 사용한다.
- `ARM_CALIBRATED=False`: generic 3D IK geometry는 아직 완전 검증되지 않음.
- `PLANAR_ARM_CALIBRATED=True`: 과거 base=90 측면 카메라 planar pick은 실제 성공.
- `EEG_SOURCE="hid"`: PolyG-I PID 0x0010 native HID 경로.
- `EEG_CONFIG_VERIFIED=False`: transport는 성공했지만 montage/participant validation 전.
- `CAM_MOCK=True`, `CAM_CALIBRATED=False`: generic overhead-camera orchestrator 경로는
  아직 실제 workspace homography가 없음.
- `WRIST_SEARCH_KINEMATICS_VERIFIED=False`: generic 2/3/4 sweep 물리 실행 잠금.

이 gate들은 서로 다른 subsystem을 뜻한다. 예를 들어 wrist camera가 잘 나온다고
`CAM_CALIBRATED=True` 또는 `WRIST_SEARCH_KINEMATICS_VERIFIED=True`를 자동으로 켜면 안 된다.

## 18. 주요 파일 안내

| 파일 | 역할 |
|---|---|
| `laptop/config.py` | 현재 수치와 gate의 source of truth |
| `firmware/arm_controller/home_pose.h` | HOME 단일 원본 |
| `firmware/arm_controller/arm_controller.ino` | 6-servo serial firmware |
| `laptop/arm_serial.py` | exclusive Uno protocol client |
| `laptop/arm_session.py` | reset 없는 persistent serial owner/socket server |
| `laptop/floor_motion.py` | hover/grasp floor curve와 `-6/11` 벡터 |
| `laptop/arm_jog.py` | 수동 bring-up/upload; session과 동시 사용 금지 |
| `laptop/wrist_vision.py` | PW315 live capture, marker/target, raw/preview |
| `laptop/visual_contact.py` | empty jaw curve와 CONTACT/FREE/UNKNOWN |
| `laptop/wrist_search.py` | collision model 기반 search; physical gate는 아직 false |
| `laptop/vision_segment.py` | FastSAM target/gripper segmentation |
| `laptop/planar_pick.py` | 과거 실제 성공한 측면 카메라 planar picker |
| `laptop/polyg_hid.py` | 실제 PolyG-I HID protocol/decoder |
| `laptop/eeg_dashboard.py` | HID owner + localhost API/UI launcher |
| `dashboard/` | React EEG UI |
| `laptop/errp.py` | baseline/model ErrP detector |
| `laptop/record_errp.py` | correct/error 자동 라벨 epoch 수집 |
| `laptop/errp_train.py` | 참가자별 classifier 훈련 |
| `laptop/cognitive_load.py` | TAR와 자율성 allocator |
| `laptop/orchestrator.py` | 최종 shared-autonomy loop |
| `laptop/test_pipeline.py` | hardware-free 통합 회귀 테스트 |
| `laptop/validate.py` | config/gate 검증 |
| `docs/EEG_DEVICE_COMMS.md` | TeleScan/DLL/실기기 통신 근거 |
| `PATCH_NOTES.md` | 모든 패치와 물리 실험 역사 |

## 19. 테스트와 완료 기준

모든 코드 패치 후 최소:

```bash
python3 -m py_compile laptop/*.py
python3 laptop/test_pipeline.py
git diff --check
```

Dashboard를 바꾸면:

```bash
cd dashboard
npm run build
npm test
```

실제 최종 완료 기준은 unit test가 아니라 다음 영상/상태 증거다.

1. 새 배경/새 runtime target에서 hover 탐색.
2. persistent session 동안 HOME reset 없음.
3. floor curve로 X 정렬.
4. known floor level로 descend.
5. close 시 CONTACT 또는 엄격한 대체 증거.
6. lift 시 같은 target이 gripper와 동행.
7. 가까운 곳에 놓고 release 후 물체가 남음.
8. 실패 case에서 open/hover/stop.
9. 위 과정이 특정 노란 물체나 흰 종이 없이 반복 가능.
10. 그 후 EEG ErrP/TAR를 행동 의사결정에 연결.

## 20. 마지막 전달 요약

이 프로젝트는 EEG 통신과 UI는 상당히 완성됐고, 로봇팔도 과거 측면 카메라로 실제
pick/place까지 성공했다. 현재 핵심 과제는 PW315 손목 카메라 하나로 장면에 과적합하지
않는 floor pick을 끝내는 것이다. 가장 큰 구조적 문제였던 Uno 반복 reset은 persistent
session으로 해결했고, 바닥의 절대 reference와 수평 이동 vector도 코드화했다.

다음 작업자는 새 인식기를 처음부터 다시 만들지 말고 다음 세 요소를 연결해야 한다.

```text
runtime target identity
        +
floor_motion의 절대 바닥/수평 경로
        +
visual_contact 및 lift verification의 fail-closed 물리 증거
```

그리고 반드시 한 `ArmSessionClient` lifecycle 안에서 실행해야 한다.

## 21. 2026-07-27 시뮬레이션 전달 — Claude가 바로 사용할 내용

실물 Uno/PW315를 전혀 열지 않고 `simul/`에 MuJoCo 경로를 추가했다. 공급된 3MF/ZIP은
조립 좌표가 아니라 출력판 배치이므로 STL은 visual identity로만 사용하고, collision과
joint frame은 측정값 기반 명시적 모델로 만들었다. 현재 과제 조건대로 base motor는
90도로 고정되어 simulated joint 자체가 없다.

핵심 파일:

- `simul/mujoco_robot.py`: 실제 6값 servo convention, limit, floor curve, wrist RGB.
- `simul/alignment_env.py`: RGB + commanded angles + previous action만 actor에게 주는 환경.
- `simul/alignment_policy.py`: 하드웨어를 열지 않는 TorchScript runner.
- `simul/models/alignment_policy_v1.ts`: 학습된 2.1 MB 모델.
- `simul/models/alignment_policy_v1.metrics.json`: hash와 정확한 지표.
- `simul/TRAINING_REPORT.md`: 제한, 재현법, Claude 통합 계약.

v1은 20,000 randomized frame으로 학습했고, 별도 500-seed simulation에서 499회 정렬
성공했다. 실제 성공률이라고 주장하면 안 된다. scope는 한 visible candidate가 있는
상태의 `ALIGN`, elbow 78..110 local floor path뿐이다.

Claude가 `floor_grasp.py`에 연결할 때:

1. `WristSceneDetector`가 portable candidate를 하나만 선택한 경우에만 v1을 호출한다.
2. 입력 frame은 RGB이고 현재 complete 6-servo command와 previous action을 같이 준다.
3. 추천 elbow delta는 반드시 `floor_pose()`로 clamp/compensate한다.
4. learned `aligned_probability` 단독으로 정지/descend하지 않는다. 기존 marker/candidate
   geometry도 aligned일 때만 두 신호의 AND로 정렬 완료 처리한다.
5. learned action이 deadband라 0인데 geometry가 안 맞으면 기존 centroid-sign bounded
   step을 fallback으로 쓴다. 0을 grasp permission으로 해석하지 않는다.
6. 여러 candidate가 동시에 보이면 goal-conditioned input이 없는 v1 대신 기존 선택된
   candidate centroid controller를 계속 쓴다.
7. `FLOOR_GRASP_EXECUTE_VERIFIED=False`를 유지하고 먼저 real frame shadow log로 기존
   direction과 비교한다. 새 serial owner를 만들지 말고 기존 `ArmSessionClient`만 쓴다.

테스트는 `python3 simul/test_mujoco_robot.py`의 8개와
`python3 laptop/test_pipeline.py` 전체가 통과했다. 모델 SHA-256은
`becbb150a282299707b0b4f7c122ad4091cf259bc60d8ea4b8f29fc36fc1d7d6`다.

## 22. 2026-07-27 complete-task 시뮬레이션 전달 (21절 대체)

21절의 local alignment v1은 재현용으로 남겼지만 통합 대상은 아니다. 현재 통합 대상은
`simul/models/full_task_policy_v1.ts`이며 다음 전체 macro를 학습했다.

```text
SEARCH_NEXT -> ALIGN_ELBOW_DOWN/UP -> DESCEND -> CLOSE -> LIFT
                                              \-> RECOVER -> 재탐색
```

중요 파일:

- `simul/full_task_env.py`: 실제 입력만 쓰는 15-feature complete-task 환경.
- `simul/train_full_task.py`: 24만 DAgger-style state 훈련/평가/내보내기.
- `simul/full_task_policy.py`: TorchScript runner + camera/pose safety shield +
  descend/close/lift 2-frame vote.
- `simul/evaluate_full_task_physics.py`: symbolic state가 아니라 free body의
  floor-contact 해제와 gripper 추종으로 성공을 판정.
- `simul/models/full_task_policy_v1.ts`: 배포 artifact. SHA-256
  `e4451c8bc64399a8b7382d50874a262a0c205eddccc263fe33c9c379abf40323`.
- `laptop/full_task_adapter.py`: `WristScene`/`WristObservation`/commanded pose를
  학습 때와 동일한 15-value로 변환하는 hardware-free shadow adapter.

정확한 결과:

- raw randomized full task: 95.15%.
- deterministic shield 적용: 99.65%.
- 두 프레임 vote 포함 별도 10,000 seed: 9,997/10,000 (99.97%).
- MuJoCo 실제 contact/free-body 2,000 seed: 1,960/2,000 (98.0%).
- simulated object 폭 <=40 mm: 1,488/1,488 (100%).
- 폭 40--44 mm edge stress: 472/512 (92.1875%).

마지막 40건 실패는 전부 큰 직육면체가 접촉면 끝에서 밀려난 경우다. 정책 실패와 집게
물리 용량을 분리하기 위해 성공 조건을 낮추지 않았다. 따라서 현재 simulated 정상 규격은
폭 40 mm 이하이며 실제 규격은 실물 손가락을 재서 확정해야 한다.

Claude 통합 순서:

1. `FullTaskShadowController`를 한 번 만들고 task 시작/종료 때 `reset()`한다.
2. 매 새 frame의 `scene`, `wrist_observation`, 현재 commanded 6-servo pose, 선택된
   `target`을 `decide()`에 넣는다. 이 호출은 camera/Uno를 열지 않는다.
3. 먼저 returned macro/next_pose를 기존 deterministic controller 옆에 log만 한다.
4. `WAIT`는 반드시 새 frame을 받는다. `SEARCH_NEXT`는 기존 collision-checked search
   planner가 처리한다. 나머지 `next_pose`도 기존 단일 `ArmSessionClient`에서만 실행한다.
5. model confidence만으로 실행하지 않는다. adapter 내부 shield, marker/quality,
   visual-contact baseline, coherent-lift 검증을 유지한다.
6. `FLOOR_GRASP_EXECUTE_VERIFIED=False`는 shadow 결과와 cleared-workspace 실물 검증 전까지
   그대로 둔다. 이 gate를 시뮬레이션 숫자만으로 자동 변경하면 안 된다.

재현:

```bash
python3 -m unittest simul.test_mujoco_robot simul.test_full_task -v
python3 simul/evaluate_full_task_physics.py \
  --policy simul/models/full_task_policy_v1.ts --episodes 2000 --seed 20260729
PYTHONPATH=laptop python3 laptop/test_pipeline.py
```

현재 결과는 local alignment 데모가 아니라 탐색부터 물리적 lift/recovery까지의 전체
시뮬레이션 전달물이다. 다만 실제 성공률 주장은 실물 shadow/접촉/상승 검증 뒤에만 한다.

## 23. 2026-07-27 실제 수직 pinch 승격

22절 이후 실제 장난감차를 대상으로 한 번의 완전한 close/lift가 성공했다.
`arm_fk.py`의 오래된 joint map이 손끝 높이를 크게 잘못 계산해 물체 위에서 닫던 것이
직접 원인이었다. 현재 runtime FK는 `simul/mujoco_robot.py`와 대표 자세 전체에서
수치 오차 없이 일치한다.

실행된 실제 경로:

- open hover `[90,115,90,143,90,170]`, 선택 edge/marker 오차 `du=21,dv=21`.
- reach 37을 잠근 채 shoulder 높이 보상으로 30→24→18→12→6mm 수직 하강.
- 최종 오차 `du=10,dv=23`; occlusion 중 다른 FastSAM mask는 무시.
- gripper 180 close 후 empty-jaw 대비 residual 203px `CONTACT`.
- closed hover lift 후 residual 약 196px `CONTACT`, 실제 hold 확인.
- 같은 floor point에 closed로 내려간 뒤 open, 다시 open hover 복귀.

따라서 `FLOOR_GRASP_EXECUTE_VERIFIED=True`로 승격했다. 이 gate를 False로 바꾸면
모든 실제 descend/close/lift가 다시 차단된다. 상세 증거는
`docs/PHYSICAL_GRASP_VALIDATION.md`에 있다.

FastSAM nested mask 병합, bottom-edge object의 양 finger 사이 제한, 선택한 candidate
identity 전달도 추가했다. 이제 `floor_grasp.py --live --arm`의 `n` reject와 `y`
confirm이 legacy 옆쓸기 controller가 아니라 검증된 `FloorServo`를 호출한다.
headless 실행은 `floor_servo.py --candidate-index N` 또는 `--reject-count N`이다.

시뮬레이션 정책도 grasp-pose target occlusion을 포함해 다시 학습했다.

- SHA-256: `b4a5cf2b976b7571bf38b2b7e96d30d5159d00186e12d93569fe91cdfa4772b7`.
- guarded randomized: 9,998/10,000 (99.98%).
- deterministic: 1,000/1,000.
- MuJoCo free-body contact: 1,959/2,000 (97.95%).
- <=40mm nominal objects: 1,488/1,488 (100%).

후속 HOME-start 무개입 시험에서 slipped-object false positive를 발견했으므로 목표 1도
엄격한 80mm retention 기준으로 재검증 중이다. 목표 2/3은 동일한 실물 controller까지
연결됐지만, 각각의 실물 완료 판정에는 카메라에 별도 도달 가능한 물체가 두 개 놓인
시험 장면이 필요하다. 그 전에는 테스트 성공을 실물 성공이라고 부르지 않는다.
