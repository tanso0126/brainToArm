# 2자유도 축소 로봇팔 모드

## 현재 기구 상태

현재 로봇팔은 다음 세 출력만 사용합니다.

| 논리 번호 | 부위 | Uno 핀 | 상태 |
|---:|---|---:|---|
| 1 | 베이스 좌우 회전 | 13 | 사용 안 함, PWM 출력 안 함 |
| 2 | 어깨 | 12 | 사용 |
| 3 | 팔꿈치 | 11 | 사용 |
| 4 | 손목 상하 각도 | 10 | 모터 없음, 3번 링크에 기계적으로 고정 |
| 5 | 집게 열기/닫기 | 9 | 사용 |
| 6 | 집게 방향 회전 | 8 | 사용 안 함, PWM 출력 안 함 |
| - | 초음파 TRIG | 7 | 사용 |
| - | 초음파 ECHO | 6 | 사용 |

따라서 팔의 위치를 바꾸는 자유도는 **2번과 3번의 두 개**입니다. 5번은
위치를 정하지 않고 집게만 여닫습니다. 4번 이후의 손목·카메라·초음파·집게
전체는 3번 링크에 붙은 하나의 단단한 링크로 계산합니다.

기존 전체 자유도 코드는 삭제하거나 수정하지 않았습니다. 모터를 수리하면
기존 `firmware/arm_controller`, `laptop/arm_session.py`,
`laptop/realtime_visual_servo.py`를 다시 사용할 수 있습니다.

## 새 코드가 기존 코드와 분리되는 방식

- `firmware/arm_controller_reduced`: 2·3·5번만 attach하는 전용 펌웨어
- `laptop/reduced_dof.py`: 2관절 순기구학, 2×2 자코비안, 충돌 검사
- `laptop/reduced_dof_session.py`: 별도 소켓을 사용하는 Uno 연결 서버
- `laptop/reduced_dof_jog.py`: 2·3번과 집게만 허용하는 수동 점검
- `laptop/reduced_dof_visual_servo.py`: 카메라·초음파 자동 접근과 파지
- `laptop/reduced_dof_firmware.py`: 전용 펌웨어 컴파일·업로드

축소 펌웨어도 통신 호환을 위해 각도 여섯 개를 주고받지만, 1·4·6번 값은
고정 표기값일 뿐입니다. 해당 핀에 `Servo.attach()`를 하지 않으므로 실제
PWM 신호가 나가지 않습니다. 노트북 코드도 이 세 값을 바꾸려는 요청을
즉시 거부합니다.

## 고정된 손목 각도 한 곳에서 보정하기

실제 고정 브래킷이 팔꿈치 링크와 일직선인 현재 가정은 다음 한 줄입니다.

```python
# laptop/reduced_dof.py
FIXED_WRIST_GEOMETRY_DEG = 180.0
```

이 값은 **4번 모터에 보내는 명령이 아닙니다.** 기계적으로 고정된 실제
손목 방향을 기구학 모델에 알려주는 값입니다. 브래킷 고정 방향이 일직선이
아니라면 이 숫자 하나만 실제 방향에 맞게 바꿉니다. 6번도 다른 방향으로
고정했다면 바로 아래 `FIXED_ROLL_GEOMETRY_DEG`만 바꿉니다.

## 처음 실행 순서

### 1. 펌웨어 컴파일 확인

```bash
python3 laptop/reduced_dof_firmware.py compile
```

### 2. Uno에 축소 펌웨어 업로드

업로드 순간 Uno가 재시작하고 2·3·5번이 축소 HOME으로 움직일 수 있습니다.
사람과 케이블을 치우고 외부 서보 전원을 확인한 뒤 실행합니다.

```bash
python3 laptop/reduced_dof_firmware.py upload
```

Uno 후보가 여러 개면 포트를 직접 지정합니다.

```bash
python3 laptop/reduced_dof_firmware.py upload --port /dev/cu.usbserial-110
```

### 3. Uno 연결 서버를 한 번만 열기

```bash
PYTHONPATH=laptop python3 laptop/reduced_dof_session.py serve
```

서버가 시리얼 포트를 계속 소유하므로 각 명령마다 Uno가 리셋되어 HOME으로
튀는 현상을 막습니다. 기존 서버와 다른 `arm_reduced.sock`을 사용합니다.

### 4. 수동으로 2·3번과 집게 확인

새 터미널에서 실행합니다.

```bash
PYTHONPATH=laptop python3 laptop/reduced_dof_jog.py
```

예:

```text
j 2 80
j 3 100
g open
g close
d
h
q
```

`j 1`, `j 4`, `j 6`은 코드가 거부합니다.

### 5. 카메라 게시기 실행

다른 카메라 프로그램이 웹캠을 점유하지 않은 상태에서 실행합니다.

```bash
PYTHONPATH=laptop python3 laptop/wrist_publish.py
```

### 6. 움직이지 않고 다음 한 스텝만 계산

```bash
PYTHONPATH=laptop python3 laptop/reduced_dof_visual_servo.py
```

`--run`이 없으면 로봇팔을 움직이지 않습니다.

### 7. 실제 자동 접근 또는 파지

접근만 실행:

```bash
PYTHONPATH=laptop python3 laptop/reduced_dof_visual_servo.py --run
```

접근 후 즉시 닫고, 짧게 들어 올린 다음, 집게 180°를 유지한 채 축소 HOME
복귀까지 실행:

```bash
PYTHONPATH=laptop python3 laptop/reduced_dof_visual_servo.py --run --grasp
```

미리보기는 `data/vision/reduced_dof_latest.jpg`에 계속 갱신됩니다.

## 제어 원리

2번과 3번을 조금 바꿨을 때 카메라의 전후 위치와 높이가 얼마나 바뀌는지
수치 자코비안으로 매번 다시 계산합니다. 목표가 화면의 목표선에서 벗어난
양과 초음파 거리를 합쳐 다음 2관절 명령을 구합니다. 한 번에 최대 5°만
명령하며 중간 전체 경로의 몸체·테이블 충돌도 검사합니다.

4번이 고정되어 있으므로 카메라 방향을 다른 위치 제어와 독립적으로 정할
수는 없습니다. 대신 2번과 3번을 함께 움직여 고정 손목의 위치와 바라보는
방향을 동시에 바꿉니다. 베이스도 고정이므로 화면 좌우 오차가 큰 물체에는
갈 수 없습니다. 이것은 소프트웨어 오류가 아니라 현재 두 자유도의 물리적
한계이며, 물체가 로봇의 앞뒤 작업 평면 안에 있어야 합니다.

## 안전 범위와 현재 가정

- 2번: 65~145°
- 3번: 35~165°
- 집게: 90~180°
- 한 번의 실시간 명령: 최대 5°
- 집게 끝의 계산상 바닥 여유가 10mm 미만이면 더 내려가지 않음
- 초음파 정지 기준은 기존 실측값 46mm 사용
- 파지 후 집게는 외부 전원을 전제로 180° 유지

고정 브래킷 각도가 모델과 다르면 충돌 계산도 틀립니다. 자동 실행 전 수동
JOG로 낮은 속도에서 2번과 3번의 방향, 손목 고정 방향, 카메라 화면,
초음파 값을 확인해야 합니다.

## 기존 모터가 복구되었을 때

축소 파일을 삭제할 필요가 없습니다. 기존 전체 펌웨어를 다시 업로드하고
기존 세션과 제어기를 실행하면 됩니다. 두 모드를 동시에 실행해 같은 Uno를
열면 안 됩니다.

