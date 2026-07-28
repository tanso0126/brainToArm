"""Bench diagnostic for the connected LAXTHA PolyG-I.

The verified PID 0x0010 unit is USB HID.  With no arguments this tool finds it,
sends the recovered TeleScan initialization sequence, captures real samples,
and always sends STOP before closing.  ``--port`` retains a conservative serial
probe for other LAXTHA variants such as the PID 0x002A CDC model.
"""
import argparse
import statistics
import sys
import time

from polyg_hid import PGA_GAINS, PolyGIHID, enumerate_devices

try:
    import serial
except ImportError:
    serial = None


def probe_hid(seconds):
    matches = enumerate_devices()
    if not matches:
        print("PolyG-I HID not found (expected VID=0x0F1F PID=0x0010).")
        return 1
    item = matches[0]
    print("PolyG-I HID found:")
    print(f"  product={item.get('product_string')!r}")
    print(f"  manufacturer={item.get('manufacturer_string')!r}")
    print(f"  path={item.get('path')!r}")

    reports = 0
    rows = []
    arrivals = []
    with PolyGIHID() as device:
        device.start()
        print(
            "  stream=STARTED "
            f"(16 physical channels, gain index {device.gain_index} "
            f"×{PGA_GAINS[device.gain_index]:.2f}, "
            f"sample selector {device.sample_selector})")
        started = time.monotonic()
        deadline = started + seconds
        while time.monotonic() < deadline:
            block = device.read_rows()
            if block:
                reports += 1
                rows.extend(block)
                arrivals.append(time.monotonic() - started)
            else:
                time.sleep(0.001)

    if not rows:
        print(f"FAIL: no HID INPUT report arrived in {seconds:.1f}s.")
        return 1
    elapsed = max(seconds, arrivals[-1] if arrivals else seconds)
    print(f"PASS: {reports} reports, {len(rows)} 8-channel samples in {elapsed:.2f}s")
    if len(arrivals) >= 2:
        intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
        median_interval = statistics.median(intervals)
        measured_fs = 64.0 / median_interval
        print(f"  median report interval={median_interval:.4f}s "
              f"(~{measured_fs:.1f} sample rows/s)")
    print("  channel ranges (raw signed counts):")
    for channel in range(8):
        values = [row[channel] for row in rows]
        print(f"    ch{channel + 1}: {min(values):6d} .. {max(values):6d}")
    print("  stream=STOPPED cleanly")
    return 0


def probe_serial(port, baud, seconds):
    if serial is None:
        print("pyserial not installed. Run: pip install pyserial")
        return 1
    print(f"Serial compatibility probe: {port} @ {baud} for {seconds:.1f}s")
    try:
        with serial.Serial(port, baud, timeout=0.2) as device:
            started = time.monotonic()
            total = 0
            while time.monotonic() - started < seconds:
                data = device.read(256)
                if data:
                    total += len(data)
                    print("  " + " ".join(f"{byte:02X}" for byte in data[:32]))
    except Exception as exc:
        print(f"FAIL: serial open/read failed: {exc}")
        return 1
    print(f"{'PASS' if total else 'FAIL'}: received {total} serial bytes")
    return 0 if total else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="capture duration (default: 4 seconds)")
    parser.add_argument("--port", help="probe a CDC/serial variant instead of HID")
    parser.add_argument("--baud", type=int, default=115200,
                        help="serial baud used with --port (default: 115200)")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be > 0")
    if args.port:
        return probe_serial(args.port, args.baud, args.seconds)
    try:
        return probe_hid(args.seconds)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
