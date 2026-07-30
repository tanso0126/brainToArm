# brainToArm Windows 실물 로봇팔 실행 가이드

이 폴더는 기존 macOS 실험 코드를 건드리지 않고 만든 **Windows 전용
배포판**입니다. 친구의 새 Windows PC에서 아래 두 파일만 순서대로
실행하면 됩니다.

1. 최초 한 번: `SETUP_WINDOWS.bat`
2. 매 실험: `RUN_AUTONOMOUS.bat`

자동 실행 순서는 현재 실물에서 사용한 것과 같습니다.

`HOME → 카메라 탐색 → 물체 추적 → 2·3·4번 연속 접근 → 즉시 닫기
→ 적재 리프트 → HOME`

## 0. 전달할 하드웨어

- 로봇팔과 Arduino Uno
- 손목 웹캠
- 외부 서보 전원과 충전된 배터리
- USB 케이블
- 초음파 센서(HC-SR04)

현재 배선은 다음과 같습니다.

| 기능 | Uno 핀 |
|---|---:|
| 1번: 베이스 회전 | 13 |
| 2번: 어깨 | 12 |
| 3번: 팔꿈치 | 11 |
| 4번: 손목 피치 | 10 |
| 5번: 집게 열기/닫기 | 9 |
| 6번: 집게 회전 | 8 |
| 초음파 TRIG | 7 |
| 초음파 ECHO | 6 |

집게는 `90°=열림`, `180°=닫힘`입니다. 왼쪽 집게에는 파란 테이프,
오른쪽 집게에는 빨간 테이프가 붙어 있어야 합니다.

> 서보 전원은 Uno 5V가 아니라 외부 전원을 사용합니다. 외부 전원의
> GND와 Uno GND는 반드시 공통으로 연결합니다. 방전된 배터리를 쓰면
> Uno는 명령 완료를 보내도 실제 서보가 움직이지 않을 수 있습니다.

## 1. Arduino 펌웨어 업로드

1. `OPEN_FIRMWARE.bat`을 더블클릭합니다.
2. Arduino IDE에서 보드를 `Arduino Uno`로 선택합니다.
3. `도구 → 포트`에서 Uno의 COM 포트를 고릅니다.
4. 업로드 버튼을 누릅니다.
5. 업로드 후 Arduino IDE의 `Serial Monitor`는 닫습니다. 다른 프로그램이
   COM 포트를 잡고 있으면 자동 실행기가 연결할 수 없습니다.

업로드할 스케치는 다음 파일입니다.

`firmware/arm_controller/arm_controller.ino`

## 2. Windows 최초 설치

인터넷에 연결한 상태에서 `SETUP_WINDOWS.bat`을 더블클릭합니다.

설치기가 자동으로 수행하는 작업:

- Python 3.11 확인(없으면 `winget`으로 설치)
- 저장소 전용 `.venv-windows` 가상환경 생성
- OpenCV, PySerial, NumPy, Ultralytics 설치
- 함께 제공된 FastSAM 모델 파일 검사
- 로봇 없이 Python 실행환경 검사

다른 프로젝트의 Python 환경은 변경하지 않습니다. 설치가 중단되면
창에 나온 마지막 오류를 사진으로 남기면 됩니다.

## 3. 카메라 확인

로봇팔과 웹캠을 설치한 뒤 `CHECK_CAMERA.bat`을 실행합니다.

- 프로그램은 Windows DirectShow 카메라를 0번부터 검사합니다.
- 화면 하단에서 **파란 집게와 빨간 집게가 동시에 검출되는 카메라**를
  자동 선택합니다.
- 양쪽 테이프가 보이지 않으면 카메라 각도를 고칩니다.
- `Q` 또는 `Esc`를 누르면 종료합니다.

자동 선택이 틀리면 명령 프롬프트에서 다음처럼 지정합니다.

```bat
windows_release\CHECK_CAMERA.bat --camera 1
```

## 4. 자동 실험

실행 전 확인:

- 충전된 외부 서보 배터리 연결
- Uno USB 연결
- 웹캠 연결
- 물체를 로봇 앞의 평평한 바닥에 놓기
- 사람 손, 헐거운 케이블, 가방 등을 팔의 이동 공간에서 치우기
- Arduino Serial Monitor 닫기

그다음 `RUN_AUTONOMOUS.bat`을 더블클릭합니다.

프로그램이 자동으로:

1. 파란/빨간 집게가 보이는 웹캠을 찾습니다.
2. CP210x ESP32를 제외하고 Uno/CH340 COM 포트를 찾습니다.
3. 펌웨어의 HOME 값이 저장소의 값과 같은지 검사합니다.
4. 카메라 창을 계속 보여줍니다.
5. 물체를 찾아 실시간으로 접근합니다.
6. 접근이 끝나면 추가 확인 없이 즉시 잡습니다.
7. 잡은 상태로 올린 뒤 HOME으로 돌아갑니다.

카메라나 COM 포트가 여러 개라 자동 선택이 어려우면 다음처럼 지정할
수 있습니다.

```bat
windows_release\RUN_AUTONOMOUS.bat --camera 1 --port COM5
```

## 5. 문제 해결

### `Arduino Uno/CH340 COM port was not found`

- 장치 관리자에서 `포트(COM 및 LPT)`를 확인합니다.
- CH340 드라이버가 필요한 Uno 호환 보드일 수 있습니다.
- Arduino Serial Monitor와 다른 시리얼 프로그램을 닫습니다.
- `DIAGNOSE.bat`을 실행해 감지된 포트를 확인합니다.

### 카메라를 못 찾음

- Windows `설정 → 개인정보 및 보안 → 카메라`에서 데스크톱 앱 카메라
  권한을 켭니다.
- Zoom, Teams, OBS 등 카메라를 점유한 앱을 종료합니다.
- 양쪽 테이프가 프레임 하단에 보이도록 웹캠 각도를 맞춥니다.
- `CHECK_CAMERA.bat --camera 0`, `--camera 1` 순서로 확인합니다.

### Uno는 연결되지만 서보가 움직이지 않음

이 프로젝트에서 실제로 발생했던 원인은 외부 배터리 방전이었습니다.

- 외부 배터리 전압과 충전 상태 확인
- 서보 전원 커넥터 확인
- 외부 전원 GND와 Uno GND 공통 연결 확인
- Uno USB만 연결된 상태를 서보 전원 연결 상태로 착각하지 않기

### `firmware HOME_POSE does not match`

다른 버전의 펌웨어가 Uno에 들어 있습니다. `OPEN_FIRMWARE.bat`으로 현재
스케치를 다시 업로드합니다.

### AI 설치가 오래 걸림

Windows에서 PyTorch/Ultralytics를 처음 설치하면 용량이 크고 시간이
걸릴 수 있습니다. 설치 창을 닫지 마세요. FastSAM 가중치 자체는
`windows_release/assets/FastSAM-s.pt`에 포함되어 별도 다운로드하지
않습니다.

## 6. 수리 후 꼭 확인할 것

친구가 모터, 기어, 링크, 카메라 브래킷을 수리하면서 기구 위치가
달라지면 기존 보정값도 달라질 수 있습니다. 특히 다음은 동일해야
현재 자동 제어가 그대로 재현됩니다.

- 카메라가 집게와 함께 움직이는 고정 방식
- 화면 하단의 파란색(왼쪽)/빨간색(오른쪽) 순서
- 2·3·4번 모터 방향
- 집게 `90°/180°` 방향
- 초음파 센서 방향과 TRIG/ECHO 핀

먼저 `CHECK_CAMERA.bat`으로 영상을 확인하고, HOME에서 케이블이나 몸체가
걸리지 않는지 육안 점검한 뒤 자동 실행을 시작하세요.

## 7. 보존 기준

Windows 이식 전 실물/macOS 전체 상태는 Git 태그
`physical-macos-baseline-2026-07-30`에 고정되어 있습니다. Windows용
파일은 모두 `windows_release/` 안에 있으므로 기존 실험 코드는 그대로
남아 있습니다.
