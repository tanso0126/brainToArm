"""축소 자유도 Arduino 펌웨어를 컴파일하거나 Uno에 업로드합니다.

기존 ``arm_firmware.py``와 전체 6서보 펌웨어는 그대로 보존합니다. 이
도구가 다루는 스케치는 2·3·5번 출력만 attach하므로 1·4·6번에 PWM을
보내지 않습니다.
"""

from pathlib import Path
import argparse
import subprocess

from arm_firmware import FQBN, find_arduino_cli, resolve_upload_port


ROOT = Path(__file__).resolve().parents[1]
SKETCH_DIR = ROOT / "firmware" / "arm_controller_reduced"


def compile_firmware():
    cli = find_arduino_cli()
    print(f"[축소 펌웨어] 컴파일: {SKETCH_DIR}", flush=True)
    subprocess.run(
        [str(cli), "compile", "--fqbn", FQBN, str(SKETCH_DIR)],
        check=True,
    )
    return cli


def upload_firmware(port=None):
    cli = compile_firmware()
    selected = resolve_upload_port(port)
    print(f"[축소 펌웨어] Uno 업로드: {selected}", flush=True)
    subprocess.run(
        [
            str(cli), "upload", "--port", selected,
            "--fqbn", FQBN, str(SKETCH_DIR),
        ],
        check=True,
    )
    print(
        "[완료] 2·3·5번만 활성화했습니다. 1·4·6번은 PWM 비활성입니다.",
        flush=True,
    )
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("compile", "upload"),
        help="컴파일만 하거나 실제 Uno에 업로드")
    parser.add_argument(
        "--port", default=None,
        help="자동 선택이 안 될 때 Uno 포트 직접 지정")
    args = parser.parse_args()
    if args.action == "compile":
        compile_firmware()
    else:
        upload_firmware(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
