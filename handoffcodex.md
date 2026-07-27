# brainToArm — Claude → Codex 인수인계 (2026-07-27)

이 문서 하나로 Claude가 이번 세션에 한 작업을 전부 이어받을 수 있게 작성했다.
대화 맥락 없이 이 글 + 코드 + git log만 보면 된다.

> 원칙(핸드오프 원문 유지): **실제 검증됨 / 코드만 구현됨 / 미완성**을 구분한다.
> 화면 정렬을 실제 접촉으로 과장하지 않는다. 물리 grasp 성공은 아직 선언 못 한다.

---

## 0. 목표 (사용자 요구)

1. 로봇팔이 물체 인식하고 원하면 집기.
2. 여러 물체 인식 → 그중 하나 선택해 집기.
3. "아니다" 신호 주면 다른 물체로 자동 전환해 집기.
4. 이후 그 "아니다"를 뇌파(ErrP)에 연결.

**사용자 강조:** 1,2,3은 **실제 물리로 집기까지 성공**해야 완료. 소프트웨어/mock 성공은 완료 아님.
카메라는 손목 위 1대(PW315), base 고정 planar. 물체는 바닥평면 위(핸드오프 §40 허용).

---

## 1. 이번 세션 내 커밋 (모두 `main`)

```
19451bf feat: reach servo + tracked descent for floor grasp
2428de0 feat: floor-plane visual servo grasp (calibrated look-then-move)
a8e0e3e feat: look-then-move floor calibration (pixel->floor homography)
3c08014 fix: measured wrist floor Jacobian and depth alignment
4b469d5 feat: multi-object floor grasp with ErrP-ready reject
```

(그 사이 `b091796`,`c14f2dd`,`6f2532e`,`bf708cd`,`2e410b7`,`81f68e7`,`620a85a`는
Codex의 sim 작업 — 내가 안 건드림. `simul/`은 읽기만 참고, 수정 안 함.)

새로 만든 파일:
| 파일 | 역할 |
|---|---|
| `laptop/floor_grasp.py` | 다물체 FastSAM 검출 + 이미지위치 기반 reject 순환 + fail-closed 상태기계 (1,2,3 소프트) |
| `laptop/floor_calibrate.py` | 픽셀→바닥(x,y) homography 캘리브 (Phase 1) |
| `laptop/floor_teach.py` | 물체 검출+바닥좌표 probe, 성공=물체 사라짐 판정 grasp |
| `laptop/floor_servo.py` | **메인**. 바닥평면 시각 서보 grasp (정렬+추종하강+close+검증) |
| `laptop/arm_fk.py` | 자립 해석적 FK (sim tool_center와 <1mm 일치 검증) |
| `laptop/depth_perceive.py` | 단안 metric depth 프로브 (근거리 부정확, 참고용) |
| `data/calibration/wrist_floor_homography.json` | 캘리브 결과 (아래 §3) |
| `data/calibration/checkerboard_9x6_opencv.png` | 인쇄용 체커보드 (10x7칸=내부코너 9x6) |

수정: `laptop/config.py`(FLOOR_* 상수 추가), `laptop/arm_session.py`(broken-pipe 방어),
`laptop/test_pipeline.py`(회귀 테스트 추가), `PATCH_NOTES.md`.

---

## 2. 핵심 물리/기하 사실 (이번에 실측으로 알아낸 것)

- **base 90 고정** → 집게가 바닥 닿는 점은 **팔 시상면(중앙선) 한 줄**뿐. 좌우(y) 도달 불가.
  물체는 반드시 화면 **중앙선(픽셀 x≈626)** 위에 놔야 한다. 좌우 확장은 base 풀어야(미래).
- **floor 곡선(shoulder/elbow, wp=180 고정)만으론 도달 부족.** 앞으로 더 뻗으려면
  **wrist_pitch를 내려(집게 끝을 물체로 겨눔) + shoulder로 높이 보정**해야 한다. (사용자 핵심 지적)
- 손목캠 floor Jacobian 실측(관측 hover): `d(물체 image y)/d(elbow) ≈ -12.9 px/deg`,
  `d(image x)/d(elbow) ≈ 0`. wrist_pitch 내리면 물체가 이미지서 집게(하단)로 강하게 내려온다.
- **집게 마커는 카메라와 rigid** → 팔 자세 바꿔도 마커 픽셀 거의 고정(≈626,700).
  따라서 "관측 hover에서 마커 바닥좌표로 서보"는 무의미(J≈0). 서보는 **물체 픽셀 vs 마커 픽셀**로 한다.
- `arm_fk` / `simul` 모델의 floor 도달 상한은 ~0.44m로 나오지만 **실제 팔은 더 뻗는다**(모델 과소평가).
  그러니 FK는 **높이(z) 유지용**으로만 쓰고, 도달·정렬은 **카메라 피드백**으로 한다.

---

## 3. Phase 1 캘리브 — 픽셀→바닥 homography (완료·검증)

**깊이 문제의 해결책.** 물체가 바닥평면(z=0)에 있다는 조건에서, 고정 **관측 자세**
`[90,112,90,158,90,170]`에서 카메라 픽셀 ↔ 바닥(x,y)를 정확한 사영 homography로 매핑.
카메라 내부파라미터·FK·단안깊이 전부 불필요.

- 결과: `data/calibration/wrist_floor_homography.json`, **재투영 RMS 0.47mm** (체커보드 9x6@22.5mm 54코너).
- 검증: 체커보드 코너들이 기대 바닥좌표(칸22.5mm 배수)에 ~2mm로 매핑됨.
- 좌표계: 체커보드 프레임(원점=보드 좌상단 코너, 관측자세 기준). **절대 원점 위치는 무관** —
  같은 프레임을 Phase 2가 재사용하므로.

**재캘리브 필요 시점(중요):**
- 팔/카메라/테이블 위치가 바뀌면(예: 대회장) **재캘리브 필수**. 1분이면 됨.
- 물체 위치만 바뀌는 건 재캘리브 불필요(그게 핵심).
- 물체가 바닥 아닌 선반 등 다른 높이면 이 방식 깨짐(별도 확장 필요).

**재캘리브 절차:**
```bash
# 1) 카메라 발행기 + arm session 켜기 (§6)
# 2) 관측 자세로 이동 + 프레임 저장
python3 laptop/floor_calibrate.py observe
# 3) 체커보드(data/calibration/checkerboard_9x6_opencv.png 인쇄, 평평하게)를
#    바닥 작업영역에 놓고 (전체가 화면 안), 인쇄된 한 칸 자로 재서:
python3 laptop/floor_calibrate.py homography --inner-cols 9 --inner-rows 6 --square-mm <실측mm>
# 체커보드 없으면 자로 잰 점 4개+ 로 대체:
python3 laptop/floor_calibrate.py add-point --x <mm> --y <mm>   # 여러 위치 반복
python3 laptop/floor_calibrate.py solve-points
```

---

## 4. 소프트웨어 1,2,3 (실물서 작동 확인)

`laptop/floor_grasp.py` + `laptop/floor_servo.py`:
- **① 인식**: FastSAM(`vision_segment.FastSAMDetector`)이 모든 인스턴스 제안 →
  집게 테이프(marker-IoU)와 바닥/팔(테두리·전체프레임) 제외 → 색·배경 무관.
  실물 프레임서 물체 4개 인식 확인. 색 무관(흰/빨강 장난감도 검출).
- **② 여러개 중 선택**: `rank_wrist_candidates`(집게 가까운 순) → `CandidateSelector.choose`.
- **③ reject 순환**: `CandidateSelector.reject(cand)` = **이미지 위치**로 veto(FastSAM id는
  매프레임 바뀌니 위치로), 다음 물체로. `confirm()`으로 초기화. **오늘 키보드 `n`/`y`,
  내일 ErrP가 같은 `reject()` 호출** — 로봇쪽 코드 안 바뀜.
- 라이브 데모: `python3 laptop/floor_grasp.py --live` (raw 프레임 읽어 표시, n/y/q).
- 회귀 테스트: `test_pipeline.py::test_floor_grasp_selection_and_reject`,
  `test_floor_homography` 통과. 전체 `python3 laptop/test_pipeline.py` PASS.

---

## 5. Phase 2 — 실제 grasp 서보 (`floor_servo.py`) : 정렬 성공, close 미완

**동작 원리(올바른 형태):**
- 관측 hover(≈35mm) 높이서 카메라가 물체 + 집게 마커 둘 다 봄.
- **집게가 바닥 근처일 때 손가락(마커)과 바닥물체가 같은 평면** → 이미지 정렬 = 실제 정렬(깊이 애매성 없음).
  그래서 관측 hover에서 물체 픽셀을 **집게 마커 중점 픽셀(≈626,700)**로 끌어옴.
- 전진 도달 = 단일 스칼라 `reach`: 먼저 wrist_pitch 180→140, 그다음 elbow 90→78, 마지막 wp 140→130.
  각 reach마다 shoulder는 FK로 hover 높이 유지(`_level_pose`/`_reach_pose`). **바닥 안 긁음.**
- 정렬되면(물체가 집게 벌림≈288px 안, dv≤110) 하강.

**실측 성공 부분:**
- 서보가 물체를 집게끝 **정중앙까지 정렬**: `dv −470 → −12 px`, `du −12`. **완벽.** 바닥 안 긁음.
- 캘리브·인식·정렬·fail-closed 전부 실물 동작.

**미완(마지막 벽) — 반드시 알아야 함:**
- 정렬 후 하강해서 **~2cm 근접하면 집게가 물체를 가려(occlusion) FastSAM이 물체를 놓침**
  (엉뚱한 blob 잡아 픽셀 356px 튐). 그 순간부터 눈 감고 전진.
- **옆으로 접근하는 집게가 물체를 straddle(감싸기) 전에 앞면으로 밀어냄**
  (물체 floor x 124→199mm로 밀림). 2회 시도 다 이 패턴으로 실패.
- 즉 **접촉 순간 물체가 안 보이고 + 옆쓸기 접근이 물체를 bulldoze** = 손목캠 1대의 근본 한계.

**실행:**
```bash
python3 laptop/floor_servo.py --align-only   # hover 정렬만 (안전, 하강 안 함)
python3 laptop/floor_servo.py                # 정렬 → 추종하강 → close → 들기 → 검증
```
성공 판정: 관측 자세 복귀 후 물체가 바닥서 사라짐(들림). 밀림은 "여전히 바닥"으로 안 속게 전체 프레임 검사.

---

## 6. 런타임 프로세스 / 장치 (이 스냅샷은 휘발성 — 반드시 재확인)

- **Uno**: `/dev/cu.usbserial-110` (CH340). ESP32 `usbserial-0001`은 자동탐색 제외.
- **카메라**: AVerMedia PW315, ffmpeg AVFoundation, 1280x720. GUI 없는 **헤드리스 발행기**가
  raw/annotated를 `data/vision/wrist_camera_latest_raw.jpg`에 계속 씀.
  발행기 스크립트는 세션 scratchpad에 있었음(휘발). 아래 §8에 전체 코드 첨부 — `laptop/`에 넣어 쓰면 됨.
  (또는 GUI 되면 `python3 laptop/wrist_vision.py --live`.)
- **arm session**: `python3 laptop/arm_session.py serve --floor hover --elbow 90` (상주, socket).
  **한 번만 실행**, 다른 client는 socket으로. `ArmSessionClient` 사용, 직접 `ArmSerial` 열지 말 것.
- **USB 허브 주의**: 프린터 등 USB 재열거 시 Uno/카메라 핸들이 죽는 것 관찰됨
  (BrokenPipe/`Device not configured`). 그럼 발행기·arm session 재시작. arm session은
  broken-pipe에 안 죽게 방어 추가함(client 중단해도 서버 생존).

현재 살아있는 것(스냅샷): 발행기 pid 25966(+ffmpeg 26059), arm session 실행 중. 팔은 안전 hover
`[90,124,90,180,90,170]`, gripper open. 물체(장난감차) 바닥 floor≈(199,63)mm, 중앙선 근처.

---

## 7. 반드시 지킬 규칙 (사용자 피드백, 어기면 화냄)

- **팔은 항상 천천히·부드럽게.** 큰 자세 점프 금지. 관절당 ≤3° 작은 waypoint로 나눠 보내고
  중간은 짧게(0.25s), 마지막만 full settle(1.9s). `floor_teach._slow_move` /
  `floor_servo.slow_move` 참고. (메모리 `arm-move-slowly.md`)
- **바닥 긁기 금지.** 정렬은 hover(바닥 위)서. 바닥 목표 z로 서보 돌리면 손가락이 바닥 박고 긁음(발생함).
- **과잉주장 금지.** "화면상 사이"≠"실제 잡힘". 반드시 물체가 실제로 들려 사라졌는지 검증.
- **base 90 고정**, wrist_roll 170=수평, gripper 90=열림/180=닫힘.
- 매 move 후 firmware DONE + ~1.9s settle + 새 프레임 discard 후 측정.
- sudo 비번은 문서/저장소에 절대 기록 금지(HP 드라이버 설치 때 1회 사용, 저장 안 함).

---

## 8. 다음 사람이 할 것 (마지막 벽 넘기)

정렬은 됐다. 남은 건 **접촉 순간 occlusion + bulldoze**. 셋 중:

**A. 수직 하강 pinch (추천, 소프트웨어로 시도 가능).**
지금은 옆으로 쓸며 접근(bulldoze 원인). 물체를 집게 사이 x-중앙(du≈0)에 맞춘 뒤
**앞전진 최소·수직 하강**으로 손가락이 물체 양옆으로 내려오게 → close. `floor_servo.grasp`의
하강 스케줄을 "전진 고정 + shoulder로만 높이 하강"으로 바꿔 실험.

**B. 근접 identity 유지.** occlusion 구간에서 FastSAM 대신 마지막 정렬 픽셀 유지 +
`visual_contact.JawBaseline.assess`(빈 집게 대비 벌림)로 CONTACT 판정, 또는 근접 광학추적 추가.

**C. 근본 = 접촉 시 안 보임.** 깊이센서/둘째 카메라 추가, 또는 **Codex의 sim 학습정책**
(접촉을 정책이 내재화) — `simul/`의 `full_task_*`/`alignment_*`가 이 방향. 이게 진짜 일반해.

목표 완료 기준(사용자): 새 배경/새 물체에서 hover 탐색 → 정렬 → 하강 → CONTACT/들림 검증 →
가까운 곳 놓기 → 실패 시 open/hover/stop. **실제 영상으로 집기 성공** 후 EEG ErrP/TAR 연결.

### 헤드리스 카메라 발행기 (scratchpad에 있던 것, `laptop/wrist_publish.py`로 저장해 쓰면 됨)
```python
"""Headless continuous wrist-frame publisher (no GUI)."""
import sys, time
sys.path.insert(0, "laptop")
import config
from wrist_vision import (
    NamedAVFoundationCamera, WristDetector, annotate,
    _atomic_write_jpeg, LATEST_RAW_PATH, LATEST_PREVIEW_PATH)

def main():
    cam = NamedAVFoundationCamera(); det = WristDetector()
    for _ in range(config.WRIST_CAMERA_WARMUP_FRAMES):
        ok, _f = cam.read()
        if not ok: raise RuntimeError("camera stopped during warmup")
    print("[publish] READY", flush=True)
    last = 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok: raise RuntimeError("camera read failed")
            obs, _m = det.detect(frame)
            now = time.monotonic()
            if now - last >= 0.15:
                _atomic_write_jpeg(LATEST_RAW_PATH, frame)
                _atomic_write_jpeg(LATEST_PREVIEW_PATH, annotate(frame, obs))
                last = now
    finally:
        cam.release()

if __name__ == "__main__":
    main()
```

---

## 9. 상태 확인 명령 (시작 시)
```bash
cd /Users/watson/Desktop/programming/playground/brainToArm
git log -6 --oneline
python3 laptop/test_pipeline.py
ls /dev/cu.usbserial*
ps aux | grep -E 'arm_session serve|wrist_publish|ffmpeg.*avfoundation' | grep -v grep
python3 laptop/arm_session.py status         # 상주 세션 살아있나
python3 laptop/floor_calibrate.py test --u 626 --v 500   # homography 로드 확인
```

핵심 요약: **캘리브·인식·선택·거부·정렬은 실물서 됨. 마지막 물리 close(집기)만 occlusion+bulldoze로 미완.
다음 = A(수직 하강 pinch) 또는 C(sim 정책)로 그 한 걸음 완성.**
