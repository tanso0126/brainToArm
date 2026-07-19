# PolyG-I EEG — device communication findings

Everything established about how the LAXTHA PolyG-I actually talks to the
laptop, what works, what is ruled out, and what is still unknown. Written so
another engineer (or agent) can continue without repeating the investigation.

**Status in one line:** **resolved and working natively on macOS.** TeleScan's
installed `LXSM-D1WD6.dll` revealed the initialization commands and report
decoder; repeated live captures now receive continuous 8-channel data and stop
the device cleanly.

The earlier conclusion that Cmd0's MSB must be 1 was the decisive error. That
rule belongs to LXSDF serial framing, while this PID `0x0010` HID protocol uses
low command bytes such as `0x01`, `0x02`, `0x04`, and `0x0A`.

---

## 1. Physical topology

```
MacBook (Apple silicon, macOS)
  └─ USB-C port
       └─ Belkin USB-C 7-in-1 Multiport Adapter   (idVendor 0x050D)
            └─ VIA Labs USB2.0 Hub                (idVendor 0x2109)
                 └─ PolyG-I LAXTHA Inc.           (idVendor 0x0F1F, idProduct 0x0010)
```

The PolyG-I connects by a **plain USB cable from the device body** to the hub.
There is **no wireless dongle** — the "dongle" is just the USB-C hub adapter the
MacBook needs because it has no USB-A port.

The device is **bus powered and enumerates successfully**, so power/cabling are
not the problem.

---

## 2. USB identity (verified with `ioreg`)

| Field | Value |
|---|---|
| USB Product Name | `PolyG-I LAXTHA Inc.` |
| USB Vendor Name | `LAXTHA Inc.` |
| idVendor | **0x0F1F** (3871) |
| idProduct | **0x0010** (16) |
| bDeviceClass | 0 (class defined at interface level) |
| bcdUSB | 0x0110 (USB 1.1) |
| Speed | Full Speed, 12 Mbps |
| bNumConfigurations | 1 |
| iSerialNumber | 0 (no serial string) |
| **bInterfaceClass** | **3 = HID** |
| bInterfaceSubClass / Protocol | 0 / 0 (not a boot device) |
| bNumEndpoints | 2 |
| Bound driver (macOS) | `AppleUserUSBHostHIDDevice` (OS built-in) |

Reproduce:

```bash
ioreg -p IOUSB -w0 -l -r -n "PolyG-I LAXTHA Inc."
ioreg -p IOService -w0 -l -r -c IOUSBHostInterface | grep -A20 -i laxtha
```

---

## 3. HID report descriptor (decoded)

Raw descriptor from IOKit:

```
06 00 FF 09 01 A1 01 19 01 29 01 15 00 26 FF 00 95 08 75 08 91 02
19 01 29 01 15 00 26 FF 00 95 80 75 40 81 02 C0
```

Decoded:

```
06 00 FF   Usage Page (Vendor-Defined 0xFF00)
09 01      Usage (0x01)
A1 01      Collection (Application)
  19 01 29 01 15 00 26 FF 00     Usage 1..1, Logical 0..255
  95 08 75 08 91 02              OUTPUT: ReportCount 8  x ReportSize 8 bit  = 8 bytes
  19 01 29 01 15 00 26 FF 00     Usage 1..1, Logical 0..255
  95 80 75 40 81 02              INPUT : ReportCount 128 x ReportSize 64 bit = 1024 bytes
C0         End Collection
```

Consequences:

- **OUTPUT report = 8 bytes** → host → device commands.
- **INPUT report = 1024 bytes** → device → host stream (this is where EEG data
  will arrive). Matches IOKit `MaxInputReportSize = 1024`, `MaxOutputReportSize = 8`.
- No Report IDs are declared, so hidapi writes need a leading `0x00` byte.
- `ReportInterval = 1000` µs (1 ms polling).
- Vendor-defined usage page `0xFF00` — a raw data pipe, not a keyboard/mouse.

---

## 4. What works on macOS

The device **opens natively with no driver install and no special permission**:

```bash
pip install hidapi
```

```python
import hid
d = hid.device()
d.open(0x0F1F, 0x0010)          # succeeds
print(d.get_manufacturer_string())   # 'LAXTHA Inc.'
print(d.get_product_string())        # 'PolyG-I LAXTHA Inc.'
```

`hid.enumerate()` reports:

```
vendor_id: 3871, product_id: 16, manufacturer_string: 'LAXTHA Inc.',
product_string: 'PolyG-I LAXTHA Inc.', usage_page: 65280, usage: 1,
interface_number: 0, path: b'DevSrvsID:...'
```

Writes are accepted by the OS (`write()` returns the byte count).

**This means Windows is not required.** The earlier plan's "Path B" (a Windows
box running `LXSMWD12.dll` and forwarding over TCP) is unnecessary.

---

## 5. Passive behavior versus initialized streaming

- 6 s non-blocking read: **0 reports**.
- 10 s blocking read (`timeout_ms=10000`): **0 bytes**.
- `get_feature_report(0)` and `(1)`: **read error** (no feature reports).

So the device powers up, enumerates, and accepts an open, but correctly transmits
nothing until initialized. IOKit later showed one INPUT report per earlier probe,
which also disproved the original "completely silent" interpretation.

With the recovered initialization sequence, a 3-second native capture produced:

```
PASS: 10 reports, 640 8-channel samples in 3.00s
median report interval=0.2843s (~225.1 sample rows/s)
```

Every INPUT report is 1,024 bytes. Repeated starts and stops are deterministic.

---

## 6. PolyG-I HID command and input formats — recovered

`LXSM-D1WD6.dll` constructs every command as an 8-byte HID OUTPUT payload:

```
command, arg1, arg2, 00, 00, 00, 00, 00
```

hidapi requires a leading report-ID byte, so the actual `write()` buffer is nine
bytes: `00 | command arg1 arg2 00 00 00 00 00`.

Verified initialization for this PolyG-I:

| purpose | 3 meaningful bytes | evidence |
|---|---|---|
| stop/reset prior stream | `01 00 00` | `Stop_Stream` export |
| mode 0 | `0A 00 00` | `Set_ModeChange(0)` |
| physical PID 16, gain index 6 | `02 10 09` | `Set_PGAs(16, 6)` stores `15-6` |
| working sample timer | `04 01 D3` | `Set_SampleFreq(9)` |
| start | `01 01 00` | `Start_Stream` export |

The full order is STOP → mode → PID/gain → timer → START. Sending START alone
can produce only one response block instead of the continuous configured stream.

The DLL also exposes timer selectors 7 (`04 07 48`) and 8 (`04 03 A4`). On this
unit selector 9 is the stable high-rate mode. Although TeleScan's `polyg-i.txt`
labels the calibration 256 Hz, sustained USB delivery measured 225.1 rows/s, so
the application uses the measured 225 Hz clock for epoch timing.

### INPUT report decoder

`LXSM-D1WD6.dll` reads 1,025 bytes on Windows (report ID plus 1,024-byte payload),
skips the ID, and decodes 512 consecutive words:

```python
word = (((high_byte - 8) & 0xFF) << 8) | low_byte
sample = word - 65536 if word & 0x8000 else word
```

The DLL treats the result as 64 time rows × 8 interleaved acquisition channels.
macOS hidapi already removes the report ID and returns exactly 1,024 bytes.

### Why the original probe failed

The original investigation incorrectly applied the LXSDF Cmd0 MSB rule and only
tried `0x80–0xFF`. It used `Cmd1 = Cmd2 = 0x00`, in both placements
inside the 8-byte OUTPUT report:

| layout | report bytes | result |
|---|---|---|
| head | `00 | c0 c1 c2 00 00 00 00 00` | no response (128 tries) |
| tail | `00 | 00 00 00 00 00 c0 c1 c2` | no response (128 tries) |

(leading `00` = hidapi report-ID byte). The real commands begin with `01`, `02`,
`04`, and `0A`, so none was present in that search space. Blind brute force was
neither necessary nor appropriate; static analysis supplied the exact bytes.

---

## 7. Ruled out — and why

### 7.1 Virtual COM port (the old "Path A") — WRONG for this unit

No `/dev/cu.*` or `/dev/tty.usb*` node ever appears. That is **correct
behaviour**, not a fault: this unit is the HID variant.

LAXTHA ships **two USB variants**. Their own driver package
(`github.com/LAXTHA/DeviceDriver` → `LXUSBCDC.zip` → `LXUSBCDC.inf`) binds:

```
[DeviceList.NTx86]
%DESCRIPTION%=DriverInstall, USB\VID_0F1F&PID_002A     ; -> usbser.sys
DESCRIPTION="LX USBCDC"
SERVICE="USB UART"
```

- **PID 0x002A** = CDC variant → appears as a virtual COM port (usbser.sys).
- **PID 0x0010** = our unit → **HID**, never a COM port on any OS.

So `EEG_SOURCE="serial"` can never work for this hardware.

### 7.2 TeleScan under CrossOver/Wine — IMPOSSIBLE

TeleScan is installed in the CrossOver bottle named `Steam`:

```
~/Library/Application Support/CrossOver/Bottles/Steam/drive_c/Program Files (x86)/TeleScan
```

It launches and runs, but reports **"device connection failed"**. Reason —
Wine's HID enumeration in that bottle contains only:

```
HKLM\System\CurrentControlSet\Enum\HID\VID_2563&PID_0575
HKLM\System\CurrentControlSet\Enum\HID\VID_845E&PID_0001
HKLM\System\CurrentControlSet\Enum\HID\VID_845E&PID_0002
```

`VID_0F1F&PID_0010` is **absent**. Wine's macOS HID backend only exposes
Generic-Desktop usage pages (gamepads/joysticks) and filters out
vendor-defined page `0xFF00`. **This is not fixable by configuration.**

Verify:

```bash
CX=/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine
"$CX" --bottle "Steam" reg query 'HKLM\System\CurrentControlSet\Enum\HID'
```

#### Side fix that was needed to get TeleScan to launch at all

TeleScan prompts *"<Visual C++ 2015 Redistributable(x86)> will be installed!"*
and the install is then blocked as a duplicate. Cause: the bottle already has a
**newer** VC++ runtime (v14.51 / 2022), so the 2015 (14.0) installer refuses;
the actual runtime DLLs (`msvcp140.dll`, `vcruntime140.dll`, `ucrtbase.dll`,
`concrt140.dll`) are already present in both `system32` and `syswow64`. The
install is therefore unnecessary. Fix applied:

```bash
CX=/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine
K='HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86'
"$CX" --bottle "Steam" reg add "$K" /v Installed /t REG_DWORD /d 1 /f
"$CX" --bottle "Steam" reg add "$K" /v Major /t REG_DWORD /d 14 /f
"$CX" --bottle "Steam" reg add "$K" /v Minor /t REG_DWORD /d 51 /f
"$CX" --bottle "Steam" reg add "$K" /v Bld /t REG_DWORD /d 36247 /f
"$CX" --bottle "Steam" reg add "$K" /v Version /t REG_SZ /d "v14.51.36247.00" /f

# then launch the exe directly, bypassing the prerequisite wrapper:
"$CX" --bottle "Steam" --wait-children "C:/Program Files (x86)/TeleScan/TeleScan.exe"
```

(Only worth doing on a machine where Wine *can* see the device — i.e. not this
one. Recorded for completeness.)

### 7.3 Vendor Windows APIs don't cover PolyG

- **LXDeviceAPI** (`LXE64_LXDeviceAPI_DeveloperManual_en.pdf`) has exactly the
  functions we'd want (`StartStream`, `StopStream`, `GetStreamData`,
  `SetSampleFrequency`, …) but its *Supporting Devices* appendix lists only
  QEEG-32FX and QEEG-64FX(8…64ch). **PolyG-I is not supported.**
- **LXSMWD12** (`LXE33_ubpulse_USBHID_Developer_Manual_LXSMWD12_en.pdf`) is
  "ubpulse HRV API … USB HID communication" and covers PIDs 33 / 58 / 56 only.
  It does confirm the architecture: *"USB HID Class Driver (OS default driver)"* —
  i.e. the vendor DLL is a thin wrapper over ordinary HID, which is exactly what
  we already do natively on macOS.

### 7.4 Static analysis of TeleScan — completed enough to solve acquisition

Installed TeleScan components relevant to devices:

| File | Role |
|---|---|
| `TeleScan.exe` | contains the device table; the only `LXSM*` name string in it is `LXSMWD4` |
| `LXEXDLL_DEVICESELECT.dll` | holds the device-name list (`PolyG-U`, **`PolyG-I`**, `PolyG-E`, `PolyG-A`, QEEG…, ubpulse…) |
| `LXSM-D1WD5/6/7/8/10.dll`, `LXSMWD2/5/6/7/8.dll` | **HID** device modules — all import `hid.dll`, `HidD_*`, `HidP_*`, `SetupDiGetClassDevs` |
| `LXSMWD4.dll` | **serial** module — uses `SetCommState`, `SetCommTimeouts`, `SetupComm` (serves the CDC/PID-0x002A models, not ours) |
| `LXDeviceAPI.dll` | modern API, pushes PIDs 0x04 / 0x06 / 0x42 |
| `SLABHIDDevice.dll`, `ftd2xx.dll` | Silicon Labs HID and FTDI helpers |

`deviceIDinfo.txt` maps PolyG-I to TeleScan device 28, and
`device/device28config.sys` identifies physical PID 16, a 16-signal polygraph,
initial gain index 6, and the first eight signals as EEG. `LXEXDLL_D1WD6.dll` is
the only extension in this family with `SetDeviceID` and `SetModeChange`; it
imports `LXSM-D1WD6.dll` directly.

Disassembling the small exported functions in `LXSM-D1WD6.dll` recovered:

- command buffer construction at RVA `0x1370`;
- PID/gain command at `Set_PGAs` RVA `0x1550`;
- sample timer table at `Set_SampleFreq` RVA `0x13D0`;
- stop at RVA `0x1690` and start at RVA `0x1870`;
- ReadFile size `0x401`, 512-word conversion loop, and the eight-channel ×
  64-row memory layout.

Those results were then verified against the live device before being added to
the codebase. No modification of TeleScan or Windows execution was required.

Note: the installer `TeleScan_Setup.exe` is a **Setup Factory** package
(`irsetup.exe` + `lua5.1.dll`); 7-Zip cannot open it and only installer-infra PEs
are stored uncompressed in its overlay. Extraction is unnecessary anyway — the
app is already installed in the bottle.

---

## 8. LXSDF protocol reference (not used by this HID stream)

The stream format is **LXSDF T2A** ("LX Serial Data Format"), documented at
<http://laxtha.net/packet-lxsdf-t2a/> and in `github.com/LAXTHA/LXSDF`.

Stream-mode Tx packet, one byte per index:

| index | value | meaning |
|---|---|---|
| 0 | 255 (0xFF) | SyncByte0 |
| 1 | 254 (0xFE) | SyncByte1 |
| 2 | 0–15 | PPD — Packet Property Data (0–15 = stream mode; 16–254 = non-stream) |
| 3 | 0–254 | PUD0 |
| 4 | 0–255 | **PC — Packet Count** (+1 per packet; use for drop detection) |
| 5 | 0–253 | PUD1 |
| 6 | 0–255 | PCD — Packet Cyclic Data |
| 7 | 0–253 | CRD (bit 6) / PUD2 (bits 5–3) / PCDT (bits 2–0) |
| 8 | 0–253 | PSD1 — channel 0 **high** byte |
| 9 | 0–255 | PSD0 — channel 0 **low** byte |
| 10, 11 | | channel 1 high, low |
| … | | 2 bytes per channel, high then low |

Sync bytes `FF FE` occur only at a packet start (high bytes are constrained to
≤253), so a receiver resynchronises by scanning for that pair.

Other useful details from the spec:

- `PCD[28]` = **number of channels** in the packet's stream area.
- `PCD[27]` = **number of samples** in the packet's stream area.
- When `PCDT = 0`, **PC wraps at 31**, not 255. ⚠️ `laptop/lxsdf.py` currently
  assumes 8-bit wraparound for drop counting — revisit once the real PCDT is known.
- `ComPath` values: 0 = UART, 1 = USB CDC, 2 = Bluetooth SPP, 3 = BLE SPS.

`laptop/lxsdf.py` implements this framing for mock, CDC/serial, and TCP
compatibility sources. It must **not** parse PID `0x0010` HID reports: live blocks
contain no `FF FE` headers, and the vendor DLL confirms the separate raw format.

---

## 9. Implemented codebase changes

1. `EEG_SOURCE="hid"` selects native PID `0x0010` acquisition.
2. `laptop/polyg_hid.py` owns exact discovery, command writes, report decoding,
   and deterministic STOP cleanup.
3. `EEGBridge` timestamps the 64 decoded rows per report and feeds the existing
   scale-invariant ErrP pipeline without using `LXSDFParser`.
4. `eeg_detect.py` performs a bounded live HID capture by default and preserves
   an explicit `--port` serial compatibility mode.
5. `requirements.txt` includes `hidapi`; validation understands the HID settings.
6. Unit tests cover command length, high-byte bias, signed conversion, optional
   Windows report ID, and malformed report rejection.

---

## 10. Remaining signal-level validation

USB communication is no longer blocked. The remaining work is physiological
calibration, not transport reverse engineering:

1. Mount the real electrodes and confirm which acquisition indices correspond
   to Fz/FCz/Cz for `ERRP_FRONTOCENTRAL`.
2. Observe known signal changes/artifacts per electrode and verify no channel is
   flat, duplicated by cabling, or saturated.
3. Re-measure sustained row rate during a longer mounted session; keep `EEG_FS`
   aligned to the physical clock.
4. Record participant-specific labeled ErrP trials and train/validate the model.
5. Only then set `EEG_CONFIG_VERIFIED=True`; the guard is intentionally still
   false even though raw input now works.

---

## 11. Reproduction cheat-sheet

```bash
# 1. Is the device present and what is it?
ioreg -p IOUSB -w0 -l | grep -iE '"USB Product Name"|idVendor|idProduct'

# 2. Interface class (3 = HID) and report descriptor
ioreg -p IOService -w0 -l -r -c IOUSBHostInterface | grep -A25 -i laxtha

# 3. Any serial port? (expected: none for PID 0x0010)
ls /dev/cu.* /dev/tty.usb* 2>/dev/null

# 4. Start, capture, decode, report cadence/ranges, and STOP safely
python3 laptop/eeg_detect.py --seconds 5

# 5. Exercise the exact background bridge used by the application
python3 laptop/eeg_bridge.py

# 6. Confirm Wine cannot see it (historical diagnosis only)
CX=/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine
"$CX" --bottle "Steam" reg query 'HKLM\System\CurrentControlSet\Enum\HID'
```

---

## 12. Sources

- PolyG-I product page — <http://www.laxtha.com/ProductView.asp?Model=PolyG-I>
- LXSDF spec repo — <https://github.com/LAXTHA/LXSDF>
  (`LXD12_LXSDFT2_CommunicationStandard.pdf`, `LXD10_LXSDFT2A_CommunicationStandard.pdf`)
- LXSDF T2A packet layout — <http://laxtha.net/packet-lxsdf-t2a/>
- LXDeviceAPI + developer manual — <http://laxtha.net/lxdeviceapi/>,
  <https://github.com/LAXTHA/LXDeviceAPI>
- LXSMWD12 USB-HID developer manual —
  <https://github.com/ubpulse/ubpulse-H3> (`LXE33_ubpulse_USBHID_Developer_Manual_LXSMWD12_en.pdf`)
- LAXTHA CDC driver (PID 0x002A) — <https://github.com/LAXTHA/DeviceDriver> (`LXUSBCDC.zip`)
- TeleScan — <https://github.com/LAXTHA/TeleScan>
