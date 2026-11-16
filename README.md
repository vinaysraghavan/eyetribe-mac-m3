# Eye Tribe tracker on an Apple Silicon Mac (M3, macOS Tahoe)

A userspace driver and bridge that makes the discontinued Eye Tribe tracker (USB `29ab:00fb`) work again.
It serves gaze data over the original Eye Tribe TCP/JSON protocol on `127.0.0.1:6555`.

```
tracker --USB--> eyetribe_stream (C++, libusb) --raw frames, pipe--> eyetribe_server.py --TCP :6555--> your app
```

Measured on one person, one session (M3 MacBook, 1512x982 screen, about 60 cm): about **0.5 to 0.7 degrees accuracy
on held-out points, about 1 degree precision, about 30 Hz**. See "Limits" before trusting that for anything else.

## Setup (once)

```bash
brew install libusb pkg-config
python3 -m venv venv && source venv/bin/activate
pip install opencv-python numpy
clang++ -std=c++17 -O2 eyetribe_stream.cpp -o eyetribe_stream $(pkg-config --cflags --libs libusb-1.0)
chmod +x run_tracker.sh
```

## Run

```bash
./run_tracker.sh --show
```

In the debug window: **c** calibrate (look at each dot, head still), **v** validate (held-out accuracy test),
**g** live gaze dot, **s** snapshot, **q** quit. `calibration.json` is saved and reloaded automatically.
Check the protocol from another terminal: `python eyetribe_client_test.py 20`.

Useful options: `--calib-grid 3..5` (default 4 = 16 points), `--save-misses N` (save frames where a glint pair was
seen but no pupil), `--snap-after S`, `--screen WxH`, `--fix-deg 1.5` / `--fix-ms 150` (fixation radius and minimum duration).

**Fixations.** The `fix` flag in each frame is true when 85% of the last 250 ms of gaze samples lie within `--fix-deg`
degrees of their median (at least `--fix-ms` of data). The radius is generous on purpose: this tracker's precision is about
1 degree, so a tighter test would almost never fire. Gaze is also limited to the calibrated range plus a 25% margin
(the quadratic fit otherwise extrapolates wildly), and to 15% beyond the screen edges.

## What was found out about this device

- It enumerates as a UVC camera but streams **8-bit grayscale packed in a "YUY2" container** (mode `WxH` is really a
  `2W x H` grayscale image). macOS decodes that as a solid green image, so frames are read over libusb instead.
- macOS owns the camera's interfaces; libusb must auto-detach the kernel driver (needs `sudo`).
- **The LEDs strobe on alternate frames.** Half the frames are LED-off and contain no glints; the server skips them.
  The driver numbers every camera frame and dropped frames leave gaps, so frame parity (even/odd) identifies the lit frames
  exactly, whatever is dropped. (Comparing each frame with the previous one Python saw passed about 17% of unlit frames
  as lit, and discarded about 17% of lit ones, at the driver's real 25-30% drop rate.) Rebuild `eyetribe_stream` after
  updating so the numbering also counts incomplete frames.
- Extension unit 3, control `0x03` is the **LED bank mask** (2 bytes; observed: 0 near-right, 1 far-right, 2 near-left,
  4 far-left, 0xff all). The tracker stays dark until control `0x04` is **written** (any value, e.g. its own current
  value). `--leds 0xff` does both. The state is lost on unplug / power cycle.
- Controls `0x07`, `0x08`, `0x09` hold identifier strings and a signature. Nothing here writes to them.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Could not open 29ab:00fb" | Device not on the bus: replug, check **System Settings > Privacy & Security > Allow accessories to connect**, try a direct port. Wait 10 s after plugging in. |
| Image very dark, no eyes | Illumination not started: run via `--leds 0xff` (the launcher does). |
| "none" is high in validation | Use the current server and a rebuilt `eyetribe_stream` (parity gating). Remaining misses are saved by `--save-misses N` as `miss_*.png`: look for pupils merged into a long dark eyelid band (half-closed eyes), which the shape test rejects on purpose. |
| Accuracy drifts | You moved your head. Head position is not compensated; recalibrate. |

## Files

- `eyetribe_stream.cpp` : the driver (negotiation, async bulk streaming, LED control, `--xu-scan`, `--uvc-scan`, sweeps)
- `eyetribe_server.py` : detection, calibration, validation, gaze demo, Eye Tribe protocol server
- `eyetribe_client_test.py`, `run_tracker.sh` : protocol check and launcher
- Development tools: `eyetribe_uvc_driver.cpp` (descriptor probe), `eyetribe_rawusb.cpp`, `eyetribe_capture.cpp`,
  `analyze_raw.py` (the steps used to reverse-engineer the format)

## Limits (please read)

- Tested by one person, without glasses, in one office, with one calibration at a time. Other faces, glasses, lighting
  and sitting positions are untested.
- No head-movement compensation. Large head shifts degrade accuracy until recalibration.
- Writes to undocumented device controls were used during reverse engineering; it behaved, but there is no guarantee
  for other units or firmware.
- The protocol implementation covers `tracker get/set`, `push`, `heartbeat` and the `calibration` category. It was tested
  with the included client, not with the original Eye Tribe SDK or other third-party clients.
