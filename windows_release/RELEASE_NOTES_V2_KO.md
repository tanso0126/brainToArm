# brainToArm Windows 통합 운영실 v2.0.1

이 버전은 개발용 배치 파일 묶음이 아니라 Windows용 설치 앱입니다.

1. `brainToArm-Windows-Setup-v2.0.1.exe` 하나를 받습니다.
2. 더블클릭해 설치합니다. Python, Node.js 또는 명령 프롬프트 설정은 필요 없습니다.
3. 설치가 끝나면 전용 GUI가 자동으로 열립니다.
4. 이후에는 바탕화면의 **brainToArm 통합 운영실** 아이콘만 실행합니다.

GUI에서 PolyG-I EEG, ErrP/TAR, 3D 시뮬레이션, 손목 카메라, Arduino Uno,
다중 물체 인식, 자동 접근·파지·HOME 복귀와 모든 주요 설정을 다룹니다.

실물 서보 전원과 Arduino 펌웨어 업로드는 하드웨어 작업이므로 최초 실험 전에
설명서의 배선 및 펌웨어 절차를 확인해야 합니다.

## v2.0.1 수정

- 패키지의 `_internal` 실행환경에 Arduino HOME 설정과 펌웨어 전체를 포함했습니다.
- 시뮬레이터가 `firmware/arm_controller/home_pose.h`를 찾지 못하던 v2.0.0
  패키징 오류를 수정했습니다.
- Windows 빌드에서 GUI/API 확인뿐 아니라 MuJoCo 시뮬레이션 시작과 실제 JPEG
  렌더 프레임 생성까지 통과해야 릴리스하도록 검사를 강화했습니다.
