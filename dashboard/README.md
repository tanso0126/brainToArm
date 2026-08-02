# brainToArm 뇌파 대시보드

PolyG-I 뇌파 측정 서비스와 연결되는 로컬 웹 화면입니다. 브라우저는
그래프와 상태를 표시하고, `../laptop/eeg_dashboard.py`가 USB HID 장치를
단독으로 연결해 `http://127.0.0.1:8765`의 로컬 API로 전달합니다.

## Windows에서 가장 쉬운 실행 방법

최초 한 번 `../windows_release/SETUP_WINDOWS.bat`을 실행한 뒤에는
`../windows_release/START_CONTROL_CENTER.bat`만 더블클릭합니다. 통합 화면에서
EEG, 3D 시뮬레이션, 손목 카메라, Arduino, 다중 후보 파지를 모두
버튼으로 관리합니다.

## macOS 개발 실행

저장소 루트에서 다음 명령 하나를 실행합니다.

```bash
python3 laptop/eeg_dashboard.py
```

정상 실행되면 브라우저에서 `http://localhost:3000`이 열립니다.

## 개발용 실행

Node.js `22.13.0` 이상이 필요합니다.

```bash
npm install
npm run dev
```

API와 화면을 별도로 확인할 때만 위 방법을 사용하고, 평소에는 저장소
루트의 `python3 laptop/eeg_dashboard.py`를 권장합니다.

## 검사 명령

```bash
npm run lint
npm test
```

`npm test`는 실제 배포용 화면을 만들고, 임시 시작 화면이 아니라 최종
뇌파 모니터가 포함되어 있는지 확인합니다.

## 신호와 개인정보 범위

뇌파 표본은 외부 서버로 전송되지 않습니다. CSV 기록은 저장소의
`recordings/` 폴더에 로컬로 저장되며 Git에는 포함되지 않습니다.

화면은 D1WD10 규칙으로 환산한 ADC 입력 mV를 표시합니다. 신호에는
상태를 유지하는 0.5–45Hz 대역 통과 필터와 60Hz 노치 필터를 적용합니다.
CSV에는 원본 카운트, 원본 ADC mV, 필터 적용값이 함께 저장됩니다.

고정 아날로그 전단 이득이 공개되거나 독립적으로 보정되지 않았으므로
전극 입력 µV, 전극 임피던스, 임상적 해석은 표시하거나 주장하지 않습니다.
