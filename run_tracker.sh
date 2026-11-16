#!/usr/bin/env bash
# Start the Eye Tribe pipeline:
#   eyetribe_stream (libusb driver, needs sudo)  --pipe-->  eyetribe_server.py (TCP bridge on 127.0.0.1:6555)
#
#   ./run_tracker.sh                  server only
#   ./run_tracker.sh --show           with the debug window (keys: c calibrate, v validate, g gaze dot, s snapshot, q quit)
#   ./run_tracker.sh --show --save-misses 10
#
# Environment: LEDS=0xff (LED banks), PYTHON=python (interpreter with opencv-python and numpy)
set -euo pipefail
cd "$(dirname "$0")"

LEDS="${LEDS:-0xff}"
PY="${PYTHON:-python}"

if [ ! -x ./eyetribe_stream ]; then
    echo "eyetribe_stream is not built. Run:" >&2
    echo "  clang++ -std=c++17 -O2 eyetribe_stream.cpp -o eyetribe_stream \$(pkg-config --cflags --libs libusb-1.0)" >&2
    exit 1
fi

# The tracker only illuminates after --leds has set the LED mask and rewritten control 0x04,
# and it forgets that when it is unplugged, so this runs on every start.
sudo ./eyetribe_stream --leds "$LEDS" --stdout | "$PY" eyetribe_server.py --source stdin "$@"
