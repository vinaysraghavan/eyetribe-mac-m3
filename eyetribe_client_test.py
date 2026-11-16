#!/usr/bin/env python3
"""
Minimal Eye Tribe API client: connects to the bridge server and prints gaze data.

    python eyetribe_client_test.py [seconds]      (default 20)

Prints server info, then ~5 gaze samples per second:
    state flags (gaze/eyes/presence), smoothed gaze x,y in screen pixels, fixation flag.
"""
import json
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 6555


def send(sock, obj):
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())


def describe_state(state):
    names = [(0x01, "gaze"), (0x02, "eyes"), (0x04, "presence"), (0x08, "FAIL"), (0x10, "lost")]
    return "+".join(n for bit, n in names if state & bit) or "none"


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    try:
        sock = socket.create_connection((HOST, PORT), timeout=3)
    except OSError as e:
        sys.exit("Could not connect to %s:%d (%s). Is eyetribe_server.py running?" % (HOST, PORT, e))
    sock.settimeout(1.0)

    send(sock, {"category": "tracker", "request": "set", "values": {"push": True, "version": 1}})
    send(sock, {"category": "tracker", "request": "get",
                "values": ["iscalibrated", "screenresw", "screenresh", "framerate", "trackerstate"]})

    dec = json.JSONDecoder()
    buf, t0, last_print, last_hb, frames = "", time.time(), 0.0, time.time(), 0
    while time.time() - t0 < seconds:
        if time.time() - last_hb > 3:  # keep the connection alive
            send(sock, {"category": "heartbeat"})
            last_hb = time.time()
        try:
            data = sock.recv(65536)
        except socket.timeout:
            continue
        if not data:
            print("server closed the connection")
            break
        buf += data.decode("utf-8", "replace")
        while True:
            buf = buf.lstrip()
            if not buf:
                break
            try:
                msg, end = dec.raw_decode(buf)
            except json.JSONDecodeError:
                break
            buf = buf[end:]
            values = msg.get("values") if isinstance(msg, dict) else None
            if not isinstance(values, dict):
                continue
            if "frame" in values:
                frames += 1
                f = values["frame"]
                if f and time.time() - last_print >= 0.2:
                    last_print = time.time()
                    avg = f["avg"]
                    print("%-22s gaze (%7.1f, %7.1f)  fix=%s" %
                          (describe_state(f["state"]), avg["x"], avg["y"], f["fix"]))
            else:
                print("server info:", values)
    print("received %d frames in %.1f s" % (frames, time.time() - t0))


if __name__ == "__main__":
    main()
