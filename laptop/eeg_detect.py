"""Bench test #1 — the single most important unknown.

Question: does the LAXTHA PolyG-I expose a plain USB virtual COM port (Path A,
clean, macOS-native) or does it require the Windows LXSMWD12.dll (Path B)?

Plug the device in (USB, powered on) and run:
    python eeg_detect.py

It lists serial ports before/after and dumps raw bytes from any new port so you
can eyeball whether LXSDF packets are streaming. If a new /dev/cu.usbserial-*
appears and spews bytes -> Path A works, no Windows needed.
"""
import sys
import time
import argparse

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


def list_serial():
    return {p.device: (p.description, p.hwid) for p in list_ports.comports()}


def dump(port, baud, seconds=3.0):
    print(f"\n--- raw dump {port} @ {baud} for {seconds}s ---")
    try:
        s = serial.Serial(port, baud, timeout=0.2)
    except Exception as e:
        print(f"  open failed: {e}")
        return
    t0 = time.time()
    total = 0
    while time.time() - t0 < seconds:
        data = s.read(64)
        if data:
            total += len(data)
            print("  " + " ".join(f"{b:02X}" for b in data[:32]))
    s.close()
    print(f"  got {total} bytes in {seconds}s "
          f"({'STREAMING — Path A viable' if total else 'silent — check baud / try Path B'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="probe this exact port instead of waiting for a new one")
    ap.add_argument("--baud", type=int, action="append",
                    help="baud to try (repeatable; defaults to common LAXTHA rates)")
    args = ap.parse_args()
    if serial is None:
        print("pyserial not installed. Run: pip install pyserial")
        return 1
    bauds = args.baud or [115200, 921600, 460800, 256000, 57600]

    before = list_serial()
    print("Ports BEFORE (unplug device, note these):")
    for d, (desc, hw) in before.items():
        print(f"  {d}  {desc}  [{hw}]")

    if args.port:
        cands = [args.port]
    else:
        input("\nNow plug in + power on the PolyG-I, then press Enter...")
    after = list_serial()
    print("\nPorts AFTER:")
    for d, (desc, hw) in after.items():
        print(f"  {d}  {desc}  [{hw}]")

    if not args.port:
        # Probe only ports that appeared after the prompt. Opening every serial
        # device could reset or command the Arduino arm.
        cands = sorted(set(after) - set(before))
    if not cands:
        print("\nNo new virtual COM port appeared. If the EEG was already plugged in,")
        print("rerun with --port /dev/...; otherwise investigate Windows Path B.")
        return 1

    for port in cands:
        for baud in bauds:
            dump(port, baud, seconds=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
