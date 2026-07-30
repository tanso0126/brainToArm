# PolyG-I 뇌파 장치 통신 조사 결과

LAXTHA PolyG-I를 macOS에서 TeleScan 없이 직접 구동하기 위해 확인한 USB,
HID 명령, 표본 형식, 전압 환산 및 표시 규칙이다. 이전 조사에서
`LXSM-D1WD6.dll`을 적용한 내용은 잘못이었으며, 2026-07-19에 공식 D1WD10
문서, 설치된 TeleScan 바이너리, 실제 장치 A/B 측정을 함께 사용해 바로잡았다.

**현재 상태:** macOS에서 네이티브로 연결·시작·수신·정지된다. PolyG-I는
16개 물리 입력 중 1–8번이 EEG이고, 현재 대시보드는 이 8개 채널을 정확히
256 Hz로 표시한다.

## 1. 확인된 장치 정체

| 항목 | 값 |
|---|---|
| 제품 | `PolyG-I LAXTHA Inc.` |
| VID / PID | `0x0F1F` / `0x0010` (16) |
| USB 인터페이스 | 제조사 정의 HID, 사용 페이지 `0xFF00` |
| HID 출력 | 8바이트(명령) |
| HID 입력 | 1,024바이트(ADC 워드 512개) |
| 물리 채널 | 16 |
| EEG 채널 | 물리 채널 1–8 |
| 보고서당 시간행 | 512 words / 16 = 32 rows |
| 표본률 | selector 8 = `2^8` = 256 Hz |

공식 `LXSM-D1WD10` 개발자 문서의 product-ID 표는 PolyG-I를 16으로
명시한다. 같은 문서에서 표본률은 `2^selector`, PolyG-I 최대 물리 채널은
16, 첫 8개 source는 EEG로 정의한다. 따라서 PID 숫자와 이름이 비슷하다는
이유로 D1WD6 모듈을 고른 이전 추론은 성립하지 않는다.

물리 연결은 다음과 같다.

```text
MacBook → USB-C hub → PolyG-I (USB HID, VID 0x0F1F / PID 0x0010)
```

별도 동글이나 가상 COM 포트는 없다. macOS 기본 HID 드라이버와 `hidapi`로
직접 연다.

```bash
ioreg -p IOUSB -w0 -l -r -n "PolyG-I LAXTHA Inc."
python3 -m pip install hidapi
python3 laptop/eeg_detect.py --seconds 5
```

## 2. HID 보고서 서술자

장치 descriptor의 핵심은 다음과 같다.

```text
06 00 FF 09 01 A1 01
19 01 29 01 15 00 26 FF 00 95 08 75 08 91 02
19 01 29 01 15 00 26 FF 00 95 80 75 40 81 02 C0
```

- OUTPUT: `8 × 8 bit` = 8 bytes
- INPUT: `128 × 64 bit` = 1,024 bytes
- report ID는 descriptor에 없다. `hidapi.write()` 버퍼에는 API 규칙에 따라
  선두 `0x00` report-ID 자리를 붙이므로 총 9 bytes를 넘긴다.
- macOS `hidapi.read()`는 1,024-byte payload만 돌려준다. Windows 캡처처럼
  선두 0이 포함된 1,025-byte 입력도 디코더가 허용한다.

장치는 열기만 하면 데이터를 보내지 않는다. 아래 초기화 명령이 필요하다.

## 3. D1WD10 명령 형식과 시작 순서

모든 vendor payload는 다음 8 bytes이다.

```text
command, arg1, arg2, 00, 00, 00, 00, 00
```

현재 PolyG-I의 정확한 초기화 순서는 다음과 같다.

| 순서 | payload 앞 3 bytes | 의미 |
|---:|---|---|
| 1 | `01 00 00` | STOP / 이전 stream 정리 |
| 2 | `05 10 00` | 최대 물리 채널 16 설정 |
| 3 | `04 08 00` | selector 8, 256 Hz 설정 |
| 4 | `0B 02 00` | source group 0(EEG 1–8)의 PGA index 2 설정 |
| 5 | `01 01 00` | START |

정지/예외 cleanup은 항상 `01 00 00`을 보낸다. `0x0B`의 인자는
`arg1=gain index`, `arg2=source group`이다. 기본 index 2는 공식 표에서
PGA ×0.40이다. 대시보드에서 측정 정지 중 0–15를 선택할 수 있다.
공식 DLL 매뉴얼의 호출 규칙에 따라 STOP, 최대 채널, 샘플링, PGA 명령
사이에 각각 0.12초를 둔다. START 직후 실측된 공통 전이 rail 표본은
1.0초 동안 읽어서 버린 뒤 세션 시간과 품질 계산을 시작한다.

### 실제 장치 A/B 검증

- 잘못된 D1WD6 명령 및 8채널 해석: 약 225 rows/s로 보이는 비정상 cadence와
  맞지 않는 값이 관측됐다.
- 위 D1WD10 순서 및 16채널 해석: 반복 측정에서 약 255.93–256.22 rows/s,
  보고서 간격 약 0.125 s(32 rows / 256 Hz)를 얻었다.
- 명령 간 공식 정착 시간을 적용하고 START 후 1초를 버린 물리 sweep에서
  index 0(×0.1), 1(×0.2), 2(×0.4)는 8채널 모두 rail 0.00%였다.
  index 3(×0.7)은 약 35–50%, index 4(×1.0)는 약 38–56%가 정확한
  `-32768/+32766` rail이었다. 따라서 현재 장착 상태의 가장 높은
  rail-clean 단계인 index 2를 기본값으로 사용한다.

따라서 애플리케이션 시간축은 실측값을 임의로 225 Hz에 맞추지 않고 명세의
정확한 256 Hz를 사용한다. 화면의 measured rate는 별도로 보여 전송 상태를
감시한다.

## 4. 입력 데이터 해석기

설치된 다음 파일을 정적으로 분석했다.

```text
~/Library/Application Support/CrossOver/Bottles/Steam/drive_c/
Program Files (x86)/TeleScan/LXSM-D1WD10.dll
```

`0x10001950..0x1000199e`의 변환 루프는 2 bytes를 다음처럼 복원한다.

```python
count = (high_byte - 0x80) * 256 + (low_byte & 0xFE)
```

- `0x80`은 offset-binary 중심값이다.
- low byte의 bit 0은 ADC가 아니라 marking bit이므로 `& 0xFE`로 제거한다.
- 1,024 bytes = 512 words이고, 16 물리 채널로 나누면 보고서당 32시간행이다.
- 각 행에서 앞 8개 값만 EEG 대시보드로 보낸다. 나머지는 ECG/EMG/EOG 등
  PolyG-I의 다른 source group이다.

이 구현은 [`laptop/polyg_hid.py`](../laptop/polyg_hid.py)에 있다.

## 5. 전압 환산: 가능한 것과 불가능한 것

D1WD10 DLL의 `.rdata:0x1000A180`에는 다음 float 계수가 들어 있다.

```text
-3.814813680946827e-05 V/count ≈ -1.25 / 32768 V/count
```

공식 문서 역시 DLL float output을 PGA 설정과 무관한 ADC 입력 범위
`-1.25..+1.25 V`로 설명한다. 따라서 화면과 CSV의 ADC 입력 전압은 다음처럼
고정 환산한다.

```python
adc_mv = count * (-1.25 / 32768) * 1000
```

low bit을 marking에 사용하므로 실질 전압 간격은 약 0.0762939 mV이다.

**중요한 한계:** 이것은 전극 입력 µV가 아니라 **ADC 입력 mV**이다. 전극과
ADC 사이의 고정 전치증폭 이득이 공식 D1WD10 자료에 수치로 제공되지 않았고
독립 교정도 하지 않았으므로, 임의 계수를 곱해 µV라고 표시하지 않는다.
PGA ×1.70만으로 전체 system gain을 역산하는 것도 잘못이다.

## 6. 실시간 처리 및 그래프 규칙

`laptop/eeg_dashboard.py`는 보고서 경계를 넘어 상태를 유지하는 causal IIR
필터를 사용한다.

1. 60 Hz notch, 2차, Q=30
2. Butterworth 0.5–45 Hz band-pass: edge당 2차, 전체 4차

UI/CSV는 다음 값을 구분한다.

- 원시 D1WD10 ADC count
- 환산된 원시 ADC mV
- 위 필터를 통과한 ADC mV

파형 Y축은 사용자가 두 방식 중 명시적으로 고른다. **공통 고정**은 모든
채널에 같은 범위와 0 mV 기준선을 적용하여 시간·채널 사이 진폭을 직접
비교한다. **채널별 자동**은 각 표시창의 98백분위 절대 진폭에 여유를 더해
두 자리 유효 정밀도로 올림하고 채널마다 별도 ±mV 범위를 표시한다. 큰
신호에는 즉시 확장하고 신호가 해당 채널 범위의 절반보다 작을 때만
독립적으로 축소하여 축 떨림을 줄인다. 어느 방식도 채널 중심을 0 mV 밖으로
옮기지 않는다. 표본은 256 Hz 시간축으로 연속 재생하며,
HID 보고서가 0.125초마다 묶여 와도 canvas가 보고서 단위로 점프하지 않도록
짧은 display buffer와 브라우저 refresh animation을 쓴다. 이 보간은 표시
위치에만 적용되고 CSV 표본을 생성하거나 바꾸지 않는다.

스펙트럼은 선택 채널의 최근 데이터에 256-point Hann window를 적용한
one-sided PSD(`mV²/Hz`)다. 축은 항상 -80..40 dB로 고정하고 Delta/Theta/
Alpha/Beta/Gamma는 0.5–45 Hz 총 power 대비 비율로 표시한다.

품질 카드의 RMS, peak-to-peak, DC offset, rail clipping %는 최근 2초 표본에서
직접 계산한다. 이는 접촉 임피던스나 임상 판정이 아니다. ADC rail에 닿으면
그래프와 카드에 포화를 명확히 표시한다.

## 7. TeleScan과 CrossOver

TeleScan UI가 장치를 표시하거나 그래프를 띄울 필요는 없다. 앱 실행 대신
설치 DLL의 코드·상수·설정 파일을 분석했으며, 복원한 동작을 실제 장치에서
검증했다.

CrossOver/Wine에서 TeleScan이 `VID_0F1F&PID_0010`을 열지 못하는 것은 현재
bottle의 HID 열거에 vendor-defined usage page `0xFF00` 장치가 들어오지 않기
때문이다. 이 프로젝트는 화면 자동화나 Wine 설정에 의존하지 않는다.

또한 PID `0x002A`용 LAXTHA USB CDC 드라이버는 이 PID `0x0010` 장치와 다른
변형이다. `/dev/cu.*`가 생기지 않는 것은 정상이며 `EEG_SOURCE="serial"`은
이 장치에 사용하지 않는다. LXSDF parser도 mock/serial/TCP 호환 경로용일 뿐,
PID `0x0010`의 1,024-byte HID report에는 적용하지 않는다.

## 8. 재현 및 검증

```bash
# 장치 식별
ioreg -p IOUSB -w0 -l -r -n "PolyG-I LAXTHA Inc."

# 제한 시간 동안 실제 시작/수신/정지
python3 laptop/eeg_detect.py --seconds 5

# 전체 단위·통합 테스트
python3 laptop/test_pipeline.py

# 로컬 대시보드
python3 laptop/eeg_dashboard.py
```

USB transport와 ADC 입력 환산은 해결됐다. 로봇팔 ErrP에 쓰기 전에는 실제
전극 장착 상태에서 Fz/FCz/Cz mapping, reference/ground 배치, 각 채널 포화,
피험자별 labeled ErrP 데이터를 별도로 검증해야 한다. 이 조건이 확인될
때까지 `EEG_CONFIG_VERIFIED=False` 안전 gate는 유지한다.

`OVER` 표시와 대시보드의 ADC rail은 같은 값을 가정하지 않는다. rail은
HID 원시 signed count가 정확히 `-32768` 또는 `+32766`인 비율이다. 설치된
공식 DLL의 `0x10001950..0x1000199e` 역시 동일한 바이트 순서와
`(high-0x80)*256 + (low&0xFE)` 복원을 사용한다. ADC span 판정은 원시
ADC p-p에만 적용한다. 필터 출력은 IIR 전이/링잉으로 원시 입력 범위를
넘을 수 있으므로 필터 p-p를 ADC 포화 기준으로 비교하지 않는다.

대시보드의 통합 보정 진행 숫자는 `최근 8초 수집 샘플/최소 필요 샘플`이다.
예를 들어 `2030/1638`은 이미 수량이 충분하다는 뜻이며 안정도 점수가
아니다. 이 상태에서 보정이 막히면 화면에 표시되는 채널별 `포화/불안정`과
`clip %`가 원인이다. 정상 보정이 채택되면 CH8 휴식 잡음과 TAR 파워 기준을
`data/eeg_baselines/latest.json`에 원자적으로 저장한다. 이후 같은 피험자,
같은 전극/REF/GND 배치에서 장비·PGA·샘플링·필터 설정이 일치할 때만
`POST /api/baseline/load`로 불러온다. 원시 EEG는 이 파일에 저장하지 않는다.

## 9. 출처

- LAXTHA, `LXSM-D1WD10` developer manual:
  <https://www.laxtha.com/DB_Files/ReleaseFile/LXSM-D1WD10-DRV3.pdf>
- PolyG-I product page:
  <https://www.laxtha.com/ProductView.asp?Model=PolyG-I>
- LAXTHA LXSDF compatibility reference:
  <https://github.com/LAXTHA/LXSDF>
- LAXTHA CDC driver (different PID `0x002A`):
  <https://github.com/LAXTHA/DeviceDriver>

로컬 바이너리 주소와 물리 A/B 수치는 이 저장소를 작업한 Mac의 설치본 및
연결된 PolyG-I에서 직접 얻었다.
