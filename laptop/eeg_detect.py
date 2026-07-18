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
import glob

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


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


if __name__ == "__main__":
    print("Ports BEFORE (unplug device, note these):")
    for d, (desc, hw) in list_serial().items():
        print(f"  {d}  {desc}  [{hw}]")

    input("\nNow plug in + power on the PolyG-I, then press Enter...")
    after = list_serial()
    print("\nPorts AFTER:")
    for d, (desc, hw) in after.items():
        print(f"  {d}  {desc}  [{hw}]")

    # Try common LAXTHA baud rates on every candidate port.
    cands = glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*") \
        + glob.glob("/dev/tty.usbserial*")
    if not cands:
        print("\nNo virtual COM port appeared -> device likely needs the Windows")
        print("LXSMWD12.dll. Use Path B (EEG_SOURCE='tcp' + windows bridge).")
        sys.exit(0)

    for port in cands:
        for baud in (115200, 921600, 460800, 256000, 57600):
            dump(port, baud, seconds=2.0)
