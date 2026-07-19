# brainToArm EEG dashboard

Local React/Vinext interface for the PolyG-I acquisition service. The browser
renders the data; `../laptop/eeg_dashboard.py` exclusively owns the USB HID
device and exposes a localhost API at `http://127.0.0.1:8765`.

## Development

Requires Node.js `>=22.13.0`.

```bash
npm install
npm run dev
```

Normally start both halves from the repository root instead:

```bash
python3 laptop/eeg_dashboard.py
```

## Checks

```bash
npm run lint
npm test
```

`npm test` creates the production bundle and verifies that the rendered shell
contains the final monitor rather than the disposable Sites starter.

## Data boundary

No EEG samples are sent to a hosted service. CSV sessions are written by the
Python API to the repository's ignored `recordings/` directory. The UI reports
raw signed counts and a signal-presence proxy; it does not claim calibrated μV,
electrode impedance, or clinical interpretation.
