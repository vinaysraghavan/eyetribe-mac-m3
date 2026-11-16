#!/usr/bin/env python3
"""
Eye Tribe bridge server for macOS (Apple Silicon).

Reads the tracker's UVC camera through OpenCV/AVFoundation, finds pupil + corneal
glint (PCCR), maps the pupil-glint vector to screen pixels with a calibration, and
serves gaze data over the Eye Tribe TCP/JSON protocol on 127.0.0.1:6555.

Setup:
    pip install opencv-python numpy
    # System Settings > Privacy & Security > Camera: allow your Terminal/IDE.

Typical use:
    python eyetribe_server.py --probe-cameras
    python eyetribe_server.py --camera 1 --show            # check detection visually
    python eyetribe_server.py --camera 1 --calibrate       # local 9-point calibration
    python eyetribe_server.py --camera 1                   # serve (loads calibration.json)

Clients using the standard Eye Tribe API can also run their own calibration
(calibration start / pointstart / pointend); the result is saved too.

Limitations: no head-pose compensation, so recalibrate if you move; detection
thresholds assume a dark-pupil IR image and may need tuning (--pupil-thr etc.).
"""

import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 6555

# Eye Tribe frame state bit flags
STATE_GAZE, STATE_EYES, STATE_PRESENCE, STATE_FAIL, STATE_LOST = 0x01, 0x02, 0x04, 0x08, 0x10

# Rough physical assumptions, only used to report calibration error in degrees.
MM_PER_PX = 0.2
VIEW_DIST_MM = 600.0


# --------------------------------------------------------------------------- #
# Screen size
# --------------------------------------------------------------------------- #
def detect_screen_size():
    """Main-screen size in points (what OS cursor coordinates use)."""
    try:
        from AppKit import NSScreen  # pyobjc, optional
        f = NSScreen.mainScreen().frame()
        return int(f.size.width), int(f.size.height)
    except Exception:
        pass
    try:
        import tkinter
        r = tkinter.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return int(w), int(h)
    except Exception:
        return 1512, 982  # 14" MacBook Pro default


# --------------------------------------------------------------------------- #
# Pupil / glint detection
# --------------------------------------------------------------------------- #
class Detector:
    def __init__(self, pupil_thr=0, glint_thr=0, scale=0.5,
                 min_frac=1.5e-4, max_frac=0.02):
        self.pupil_thr = pupil_thr
        self.glint_thr = glint_thr
        self.scale = scale
        self.min_frac = min_frac  # pupil area limits as a fraction of frame area
        self.max_frac = max_frac
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.normalize_on = True
        self.pupil_sigma = 2.5   # smoothing (px) for the pupil search; tuned on real grainy IR frames
        self.min_circ = 0.78       # circularity of the convex hull
        self.min_solidity = 0.82
        self.min_aspect = 0.7
        self.prior = []          # pupil centres from recent frames (a partly covered pupil is less round)
        self.prior_age = 0

    def normalize(self, gray, min_range=24):
        """Stretch contrast (1st..99.9th percentile -> 0..255) so a dark IR frame still
        works with fixed thresholds. Percentiles come from a subsample for speed."""
        sample = gray[::8, ::8]
        lo, hi = np.percentile(sample, (1.0, 99.9))
        if hi - lo < min_range:  # don't amplify noise more than 255/min_range times
            hi = lo + min_range
        k = 255.0 / (hi - lo)
        return cv2.convertScaleAbs(gray, alpha=k, beta=-lo * k)

    def glint_peaks(self, raw):
        """Small bright spots via difference-of-Gaussians. The threshold comes from a robust noise
        estimate (MAD), so it adapts to grainy low-light frames. Returns (peaks, noise_sigma, thr)
        with peaks = [(strength, x, y), ...] strongest first."""
        f = raw.astype(np.float32)
        dog = cv2.GaussianBlur(f, (0, 0), 1.0) - cv2.blur(f, (15, 15))
        sub = dog[::6, ::6]
        sigma = 1.4826 * float(np.median(np.abs(sub - np.median(sub))))
        thr = float(self.glint_thr) if self.glint_thr > 0 else max(5.0, 8.0 * sigma)
        ys, xs = np.nonzero(dog >= thr)          # sparse: only pixels above the threshold
        vals = dog[ys, xs]
        peaks = []
        for i in np.argsort(-vals)[:400]:        # greedy non-maximum suppression, strongest first
            x, y = int(xs[i]), int(ys[i])
            if any(abs(x - px) <= 3 and abs(y - py) <= 3 for _, px, py in peaks):
                continue
            # a real glint is genuinely bright: well above the local background (skin level).
            # This rejects faint rings that appear just outside a dark pupil edge.
            local = float(np.median(raw[max(y - 20, 0):y + 21, max(x - 20, 0):x + 21]))
            if float(raw[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].max()) < 1.5 * local + 5:
                continue
            peaks.append((float(vals[i]), x, y))
            if len(peaks) >= 40:
                break
        return peaks, sigma, thr

    def diagnose(self, raw):
        """Describe what the glint stage sees (for snapshots). Returns text lines."""
        peaks, sigma, thr = self.glint_peaks(raw)
        lines = ["glint filter: noise sigma %.2f -> threshold %.1f; %d peaks above it" % (sigma, thr, len(peaks))]
        h, w = raw.shape[:2]
        self.dbg_peaks = []
        for i, (v, x, y) in enumerate(peaks[:10]):
            peak = int(raw[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].max())
            bg = int(np.median(raw[max(y - 25, 0):y + 26, max(x - 25, 0):x + 26]))
            lines.append("peak %2d at (%4d,%4d): strength %.1f, raw peak %3d, local background %3d" %
                         (i + 1, x, y, v, peak, bg))
            self.dbg_peaks.append((x, y))
        return lines

    def find_pupil(self, raw, members, center, relaxed=False):
        """Look for a dark round blob around one glint cluster. Tries several darkness levels
        (the pupil is a compact dark spot inside a larger dark eye region) and returns every
        candidate with its roundness; the caller keeps the good ones."""
        h, w = raw.shape[:2]
        R = max(40, int(0.04 * w))
        x0, y0 = max(int(center[0]) - R, 0), max(int(center[1]) - R, 0)
        x1, y1 = min(int(center[0]) + R, w), min(int(center[1]) + R, h)
        if x1 - x0 < 30 or y1 - y0 < 30:
            return []
        roi = raw[y0:y1, x0:x1].copy()
        m = np.zeros(roi.shape, np.uint8)
        for gx, gy in members:                      # hide the glints so they don't punch holes
            cv2.circle(m, (int(gx - x0), int(gy - y0)), 4, 255, -1)
        roi = cv2.inpaint(roi, m, 3, cv2.INPAINT_TELEA)
        sm = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), self.pupil_sigma)
        dmin, dmax = 0.009 * w, 0.045 * w            # plausible pupil diameters in px
        amin, amax = math.pi / 4 * dmin ** 2, math.pi / 4 * dmax ** 2
        kk = max(5, int(round(0.0045 * w))) | 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
        cands = []
        levels = np.percentile(sm, (4, 8, 12, 16, 20, 24, 28, 32))
        for lev in levels:
            mask = cv2.morphologyEx((sm <= lev).astype(np.uint8) * 255, cv2.MORPH_OPEN, k)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cn in contours:
                self.dbg["blobs"] += 1
                a = cv2.contourArea(cn)
                if len(cn) < 5 or not (amin <= a <= amax):
                    continue
                self.dbg["size"] += 1
                # Shape is judged on the convex hull: a noisy edge inflates the raw contour length and made
                # round pupils look "not circular" (circularity 0.55 for blobs that are clearly discs).
                hull = cv2.convexHull(cn)
                hull_area, hull_per = cv2.contourArea(hull), cv2.arcLength(hull, True)
                circ = 4 * math.pi * a / (hull_per * hull_per) if hull_per else 0.0
                solidity = a / hull_area if hull_area else 0.0
                (cx, cy), (ma, mi), _ = cv2.fitEllipse(cn)
                diam, aspect = (ma + mi) / 2, min(ma, mi) / max(ma, mi)
                ctr = (x0 + cx, y0 + cy)
                near = math.hypot(ctr[0] - center[0], ctr[1] - center[1]) <= 0.6 * diam + 6
                if relaxed:   # we saw an eye here a moment ago: tolerate eyelid-distorted shapes
                    rnd = circ >= 0.70 and solidity >= 0.78 and aspect >= 0.6
                else:
                    rnd = circ >= self.min_circ and solidity >= self.min_solidity and aspect >= self.min_aspect
                if rnd:
                    self.dbg["round"] += 1
                if rnd and near:
                    self.dbg["near"] += 1
                stage = 3 if (rnd and near) else (2 if rnd else 1)
                if len(self.dbg_blobs) < 60:
                    self.dbg_blobs.append([ctr[0], ctr[1], diam, stage])
                cands.append({"center": ctr, "diam": diam, "circ": circ, "aspect": aspect,
                              "good": rnd and near})
        return cands

    def detect(self, gray):
        """Glint-first detection for grainy IR frames. Finds the LED reflections (a pair per eye),
        groups them, then finds the dark round pupil around each group. Returns up to two eyes:
        dicts with pupil, glint (mean of this eye's glints), diam, pccr (px), sorted by image x."""
        self.dbg = {"glints": 0, "blobs": 0, "size": 0, "round": 0, "near": 0, "pairs": 0}
        self.dbg_blobs = []   # [cx, cy, diameter, stage]; stage 1=size ok, 2=round, 3=round and next to glints
        self.dbg_glints = []
        raw = gray
        h, w = raw.shape[:2]
        peaks, _, _ = self.glint_peaks(raw)
        self.dbg["glints"] = len(peaks)
        self.dbg_glints = [(x, y) for _, x, y in peaks]
        if not peaks:
            return []

        rc = 0.016 * w   # glints closer than this belong to the same eye
        clusters = []
        for v, x, y in peaks:
            for cl in clusters:
                if math.hypot(x - cl["cx"], y - cl["cy"]) <= rc:
                    cl["pts"].append((x, y))
                    cl["w"] += v
                    cl["cx"], cl["cy"] = np.mean(cl["pts"], axis=0)
                    break
            else:
                clusters.append({"pts": [(x, y)], "w": v, "cx": float(x), "cy": float(y)})
        clusters.sort(key=lambda c: -c["w"] * (1 + 0.5 * (len(c["pts"]) - 1)))  # glint pairs first
        self.dbg["pairs"] = sum(1 for c in clusters if len(c["pts"]) >= 2)

        cands = []
        for cl in clusters[:4]:
            relaxed = any(math.hypot(cl["cx"] - px, cl["cy"] - py) < 70 for px, py in self.prior)
            found = [c for c in self.find_pupil(raw, cl["pts"], (cl["cx"], cl["cy"]), relaxed) if c["good"]]
            if not found:
                continue
            # average over the darkness levels that gave a good pupil: steadier than any single level
            pupil = np.mean([c["center"] for c in found], axis=0)
            diam = float(np.median([c["diam"] for c in found]))
            glint = np.mean(cl["pts"], axis=0)
            cands.append({"score": cl["w"] * float(np.mean([c["circ"] for c in found])),
                          "pupil": pupil, "glint": glint, "diam": diam, "pccr": pupil - glint})

        cands.sort(key=lambda e: -e["score"])
        eyes = []
        for c in cands:
            if all(np.linalg.norm(c["pupil"] - e["pupil"]) > 2 * max(c["diam"], e["diam"]) for e in eyes):
                eyes.append(c)
            if len(eyes) == 2:
                break
        if eyes:
            self.prior, self.prior_age = [e["pupil"] for e in eyes], 0
        else:
            self.prior_age += 1
            if self.prior_age > 30:
                self.prior = []
        return sorted(eyes, key=lambda e: e["pupil"][0])


# --------------------------------------------------------------------------- #
# Calibration (pupil-glint vector -> screen px)
# --------------------------------------------------------------------------- #
def _design(f, mode, mean, std):
    z = (f - mean) / std
    x, y = z[:, 0], z[:, 1]
    cols = [np.ones_like(x), x, y]
    if mode == "poly2":
        cols += [x * y, x * x, y * y]
    return np.stack(cols, axis=1)


class Calibrator:
    MIN_SAMPLES = 10
    CLIP_MARGIN = 0.25   # how far past the calibrated range the mapping is trusted (fraction of the range)

    def __init__(self):
        self.reset()

    def reset(self):
        self.features, self.targets = [], []
        self.samples = []
        self.collecting = False
        self.coef = self.mean = self.std = None
        self.zmin = self.zmax = None   # calibrated feature range, in standardised units
        self.mode = None
        self.avg_err_px = None
        self.point_errs = []

    @property
    def ready(self):
        return self.coef is not None

    def begin_point(self):
        self.samples = []
        self.collecting = True

    def feed(self, feat):
        if self.collecting:
            self.samples.append(np.asarray(feat, dtype=np.float64))

    def end_point(self, x, y):
        self.collecting = False
        if len(self.samples) < self.MIN_SAMPLES:
            return False
        s = np.array(self.samples)
        s = s[len(s) // 4:]  # drop the first quarter (saccade/settling)
        self.features.append(np.median(s, axis=0))
        self.targets.append([x, y])
        return True

    def fit(self):
        F, T = np.array(self.features), np.array(self.targets)
        n = len(F)
        if n < 3:
            return False
        self.mode = "poly2" if n >= 6 else "affine"
        self.mean, self.std = F.mean(axis=0), F.std(axis=0) + 1e-6
        A = _design(F, self.mode, self.mean, self.std)
        self.coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ T)
        z = (F - self.mean) / self.std
        self.zmin, self.zmax = z.min(axis=0), z.max(axis=0)
        err = np.linalg.norm(A @ self.coef - T, axis=1)
        self.point_errs = err.tolist()
        self.avg_err_px = float(err.mean())
        return True

    def _clip(self, f):
        """A quadratic fit extrapolates wildly outside the points it was trained on (we saw gaze 450 px past
        the screen edge), so features are limited to the calibrated range plus a margin."""
        if self.zmin is not None:
            zmin, zmax = self.zmin, self.zmax
        else:
            zmin, zmax = np.full(2, -1.3), np.full(2, 1.3)   # calibration files written before this was stored
        span = zmax - zmin
        lo = self.mean + (zmin - self.CLIP_MARGIN * span) * self.std
        hi = self.mean + (zmax + self.CLIP_MARGIN * span) * self.std
        return np.clip(f, lo, hi)

    def map(self, feat):
        f = self._clip(np.asarray(feat, dtype=np.float64))
        A = _design(f[None, :], self.mode, self.mean, self.std)
        return (A @ self.coef)[0]

    def deg(self, px):
        return math.degrees(math.atan2(px * MM_PER_PX, VIEW_DIST_MM))

    def save(self, path, screen):
        with open(path, "w") as fh:
            d = {"mode": self.mode, "coef": self.coef.tolist(),
                 "mean": self.mean.tolist(), "std": self.std.tolist(),
                 "screen": list(screen), "avg_err_px": self.avg_err_px}
            if self.zmin is not None:
                d["zmin"], d["zmax"] = self.zmin.tolist(), self.zmax.tolist()
            json.dump(d, fh)

    def load(self, path, screen):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return False
        if tuple(d.get("screen", ())) != tuple(screen):
            print(f"[calib] {path} was made for screen {d.get('screen')}, ignoring")
            return False
        self.mode = d["mode"]
        self.coef, self.mean, self.std = (np.array(d[k]) for k in ("coef", "mean", "std"))
        self.avg_err_px = d.get("avg_err_px")
        if "zmin" in d and "zmax" in d:
            self.zmin, self.zmax = np.array(d["zmin"]), np.array(d["zmax"])
        return True


def validation_report(points, deg):
    """points: list of (target_xy, [gaze samples]). Returns (per_point_rows, summary dict).
    Accuracy = distance of the median gaze from the target; precision = RMS spread of the samples."""
    rows, accs, precs, used = [], [], [], 0
    for target, samples in points:
        target = np.asarray(target, dtype=np.float64)
        if len(samples) < 10:
            rows.append((target, None, None, len(samples)))
            continue
        arr = np.array(samples, dtype=np.float64)
        med = np.median(arr, axis=0)
        acc = float(np.linalg.norm(med - target))
        prec = float(np.sqrt(np.mean(np.sum((arr - med) ** 2, axis=1))))
        rows.append((target, acc, prec, len(samples)))
        accs.append(acc)
        precs.append(prec)
        used += 1
    summary = {"points_used": used, "points_total": len(points)}
    if accs:
        summary.update(acc_mean=float(np.mean(accs)), acc_median=float(np.median(accs)), acc_max=float(np.max(accs)),
                       prec_mean=float(np.mean(precs)), acc_mean_deg=deg(float(np.mean(accs))),
                       acc_max_deg=deg(float(np.max(accs))), prec_mean_deg=deg(float(np.mean(precs))))
    return rows, summary


class LitFrameGate:
    """The tracker strobes its LEDs on alternate frames; LED-off frames show no glints, so there is
    nothing to detect in them.

    The driver numbers every camera frame, and frames it drops leave gaps in that numbering, so frame
    parity (even/odd) IS the lit/unlit pattern however many frames get dropped. Brightness is only used to
    learn which parity is the lit one. (Comparing each frame with the previous frame Python happened to see
    gets 17-18% wrong at the driver's real drop rate of 25-30%.)

    Without sequence numbers it falls back to comparing consecutive frames. If no alternation is present at
    all (e.g. a single steady LED bank), every frame is used."""

    def __init__(self, min_delta=0.3, window=60):
        self.min_delta = min_delta
        self.deltas = deque(maxlen=window)
        self.prev = None
        self.prev_lit = True
        self.par_mean = [None, None]
        self.mismatch = deque(maxlen=8)

    def is_lit(self, gray, seq=None):
        m = float(gray[::8, ::8].mean())
        if seq is None:
            return self._lit_by_delta(m)
        p = int(seq) & 1
        pm = self.par_mean
        pm[p] = m if pm[p] is None else 0.97 * pm[p] + 0.03 * m
        if pm[0] is None or pm[1] is None or abs(pm[0] - pm[1]) < self.min_delta:
            return True                      # not enough information, or no alternation: use every frame
        lit_par = 0 if pm[0] > pm[1] else 1
        # Heal a parity slip (a camera frame the driver did not count): if most recent frames look like the
        # *other* level, the parity-to-level assignment is reversed, so swap it.
        near_lit = abs(m - pm[lit_par]) < abs(m - pm[1 - lit_par])
        self.mismatch.append(near_lit != (p == lit_par))
        if len(self.mismatch) == 8 and sum(self.mismatch) >= 6:
            pm[0], pm[1] = pm[1], pm[0]
            self.mismatch.clear()
            lit_par = 1 - lit_par
        return p == lit_par

    def _lit_by_delta(self, m):
        if self.prev is None:
            self.prev = m
            return True
        d = m - self.prev
        self.prev = m
        self.deltas.append(abs(d))
        if len(self.deltas) >= 20 and float(np.median(self.deltas)) < self.min_delta:
            self.prev_lit = True
            return True
        if d > self.min_delta:
            lit = True
        elif d < -self.min_delta:
            lit = False
        else:
            lit = not self.prev_lit
        self.prev_lit = lit
        return lit


def deg_to_px(deg):
    return math.tan(math.radians(deg)) * VIEW_DIST_MM / MM_PER_PX


class FixationDetector:
    """Dispersion-style fixation test that tolerates the ~1 degree of noise this tracker has.
    A fixation is flagged when, over the last window_ms (and at least min_ms of data), 85% of the
    gaze samples lie within radius_px of their median. The old 'eight samples within 35 px' rule
    needed less noise than the hardware has and almost never fired."""

    def __init__(self, radius_px, min_ms=150, window_ms=250):
        self.r = radius_px
        self.min_s = min_ms / 1000.0
        self.win_s = window_ms / 1000.0
        self.buf = deque()

    def reset(self):
        self.buf.clear()

    def update(self, t, xy):
        self.buf.append((t, np.asarray(xy, dtype=np.float64)))
        while self.buf and t - self.buf[0][0] > self.win_s:
            self.buf.popleft()
        if len(self.buf) < 4 or self.buf[-1][0] - self.buf[0][0] < self.min_s:
            return False
        pts = np.array([p for _, p in self.buf])
        med = np.median(pts, axis=0)
        return bool(np.percentile(np.linalg.norm(pts - med, axis=1), 85) <= self.r)


# --------------------------------------------------------------------------- #
# One Euro filter for gaze smoothing
# --------------------------------------------------------------------------- #
class OneEuro:
    def __init__(self, mincutoff=1.5, beta=0.004, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.reset()

    def reset(self):
        self.t = self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.t is None:
            self.t, self.x, self.dx = t, x, np.zeros_like(x)
            return x
        dt = max(t - self.t, 1e-3)
        ad = self._alpha(self.dcutoff, dt)
        self.dx = ad * (x - self.x) / dt + (1 - ad) * self.dx
        a = self._alpha(self.mincutoff + self.beta * np.linalg.norm(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


# --------------------------------------------------------------------------- #
# TCP client wrapper
# --------------------------------------------------------------------------- #
class Client:
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
        self.push = False
        self.alive = True

    def send(self, obj):
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        with self.lock:
            try:
                self.sock.sendall(data)
            except OSError:  # includes timeouts: stream may be corrupt, drop client
                self.alive = False


# --------------------------------------------------------------------------- #
# Frame source: raw grayscale frames piped from eyetribe_stream
# --------------------------------------------------------------------------- #
class StdinSource:
    """Reads frames written by `eyetribe_stream --stdout`.
    Per frame: b"ETF1", u32 width, u32 height, u32 seq, then width*height gray bytes."""
    HEADER = struct.Struct("<4sIII")

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin.buffer
        self.seq = None
        self.last_seq = None
        self.lost = 0

    def _read_exact(self, n):
        data = self.stream.read(n)
        if data is None or len(data) < n:
            sys.exit("[source] capture stream ended")
        return data

    def isOpened(self):
        return True

    def read(self):
        magic, w, h, seq = self.HEADER.unpack(self._read_exact(self.HEADER.size))
        if magic != b"ETF1" or not (0 < w <= 8192 and 0 < h <= 8192):
            sys.exit("[source] bad frame header; is the input really eyetribe_stream --stdout output?")
        if self.seq is not None and seq != self.seq + 1:
            self.lost += seq - self.seq - 1
        self.seq = self.last_seq = seq
        n = w * h
        frame = np.empty(n, dtype=np.uint8)
        view = memoryview(frame)
        got = 0
        readinto = getattr(self.stream, "readinto", None)
        while got < n:                       # fill the frame buffer directly: no intermediate bytes object
            if readinto is not None:
                k = readinto(view[got:])
            else:
                chunk = self.stream.read(n - got)
                k = len(chunk) if chunk else 0
                if k:
                    view[got:got + k] = chunk
            if not k:
                sys.exit("[source] capture stream ended")
            got += k
        return True, frame.reshape(h, w)

    def release(self):
        pass


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
class EyeTribeBridgeServer:
    def __init__(self, args):
        self.args = args
        self.screen = args.screen
        self.running = True
        self.lock = threading.RLock()  # guards calib, latest, calibration session state
        self.clients = []
        self.calib = Calibrator()
        self.detector = Detector(args.pupil_thr, args.glint_thr, args.proc_scale)
        self.detector.normalize_on = not args.no_normalize
        self.filter = OneEuro()
        self.eye_off = [None, None]   # per-eye offset from the two-eye mean feature
        self.eye_x = [None, None]     # tracked image x of each eye, to tell them apart when only one is found
        self.det_counts = [0, 0, 0]   # frames with 0 / 1 / 2 eyes
        self.gaze_hist = deque(maxlen=5)
        self.gate = LitFrameGate()
        self.fixation = FixationDetector(deg_to_px(getattr(args, "fix_deg", 1.5)), getattr(args, "fix_ms", 150))
        self.measured = True          # was the last frame a usable (LED-lit) measurement frame?
        self.last_eyes = []
        self.last_lit = None          # most recent lit frame (for snapshots)
        self.lost = 0                 # consecutive lit frames without a usable eye feature
        self.last_good = None
        self.miss_saved = 0
        self.miss_last = 0.0
        self.recent = deque(maxlen=8)
        self.latest = None
        self.fps = 0.0
        self.last_t = None
        self.t0 = time.time()
        self.calibrating = False
        self.calib_pointcount = 9
        self.calib_pending = None
        self.calib_result = None

        if args.calib_file and self.calib.load(args.calib_file, self.screen):
            print(f"[calib] loaded {args.calib_file} (avg error {self.calib.avg_err_px:.1f}px)")

        if args.source == "stdin":
            self.cap = StdinSource()
            print("[camera] reading raw frames from stdin (eyetribe_stream)")
        else:
            self.cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                sys.exit("Could not open camera %d. Check the index (--probe-cameras) and "
                         "camera permission for your terminal." % args.camera)
            if args.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            if args.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            if args.fps:
                self.cap.set(cv2.CAP_PROP_FPS, args.fps)  # AVFoundation may ignore this
            print("[camera] %dx%d @ %.0f fps (as reported)" % (
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                self.cap.get(cv2.CAP_PROP_FPS)))

    # ---- per-frame processing ------------------------------------------------
    def _frame_feature(self, eyes):
        """One pupil-glint feature per frame that means the same thing whether one or two eyes are
        found. Calibration is trained on the two-eye mean, so with a single eye we add that eye's
        learned offset from the mean; otherwise a one-eye frame would be off by half the eye-to-eye
        difference (about 170 px on screen in testing)."""
        if len(eyes) >= 2:
            a, b = eyes[0], eyes[-1]
            feat = (a["pccr"] + b["pccr"]) / 2
            for i, e in enumerate((a, b)):
                target = feat - e["pccr"]
                self.eye_off[i] = target if self.eye_off[i] is None else 0.98 * self.eye_off[i] + 0.02 * target
                x = float(e["pupil"][0])
                self.eye_x[i] = x if self.eye_x[i] is None else 0.9 * self.eye_x[i] + 0.1 * x
            return feat
        if len(eyes) == 1 and all(o is not None for o in self.eye_off) and all(x is not None for x in self.eye_x):
            e = eyes[0]
            i = int(np.argmin([abs(float(e["pupil"][0]) - x) for x in self.eye_x]))
            return e["pccr"] + self.eye_off[i]
        return None

    def _maybe_save_miss(self, gray, eyes):
        """--save-misses N: keep up to N lit frames where a glint pair was seen but fewer pupils than
        pairs were found, so detection failures can be studied on real data."""
        limit = getattr(self.args, "save_misses", 0)
        d = self.detector.dbg
        now = time.time()
        if not limit or self.miss_saved >= limit or d.get("pairs", 0) <= len(eyes) or now - self.miss_last < 0.7:
            return
        self.miss_saved += 1
        self.miss_last = now
        name = "miss_%03d.png" % self.miss_saved
        cv2.imwrite(name, gray)
        print("[miss] saved %s: glint pairs seen %d, pupils found %d | detector %s" % (name, d["pairs"], len(eyes), d))

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if not self.gate.is_lit(gray, getattr(self.cap, "last_seq", None)):   # LED-off frame: keep the last output
            self.measured = False
            return self.last_eyes
        self.measured = True
        self.last_lit = gray
        eyes = self.detector.detect(gray)
        self.last_eyes = eyes
        if len(eyes) < 2:
            self._maybe_save_miss(gray, eyes)
        now = time.time()
        if self.last_t:
            inst = 1.0 / max(now - self.last_t, 1e-3)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self.last_t = now
        h, w = gray.shape[:2]

        with self.lock:
            feat = self._frame_feature(eyes)
            self.det_counts[min(len(eyes), 2)] += 1
            if feat is not None:
                self.calib.feed(feat)
            calibrated = self.calib.ready
            if feat is not None and calibrated:
                raw = self.calib.map(feat)
                raw = np.clip(raw, [-0.15 * self.screen[0], -0.15 * self.screen[1]],
                              [1.15 * self.screen[0], 1.15 * self.screen[1]])   # safety net beyond the feature clip
                for e in eyes:
                    e["raw"] = raw
                self.gaze_hist.append(raw)
                med = np.median(np.array(self.gaze_hist), axis=0)   # drops single-frame glitches
                avg = self.filter(med, now)
                fix = self.fixation.update(now, avg)
                state = STATE_GAZE | STATE_EYES | STATE_PRESENCE
                self.lost = 0
                self.last_good = (np.array(raw), np.array(avg), fix, state)
            else:
                self.lost += 1
                if calibrated and feat is None and self.lost <= 4 and self.last_good is not None:
                    # a blink or one missed frame: hold the last gaze instead of reporting 'lost'
                    raw, avg, fix, state = self.last_good
                else:
                    raw = avg = np.zeros(2)
                    fix = False
                    self.filter.reset()
                    self.fixation.reset()
                    self.gaze_hist.clear()
                    self.last_good = None
                    state = (STATE_EYES | STATE_PRESENCE) if eyes else STATE_FAIL

            def eye_json(e):
                r = e.get("raw", np.zeros(2))
                return {"avg": {"x": float(r[0]), "y": float(r[1])},
                        "raw": {"x": float(r[0]), "y": float(r[1])},
                        "pcenter": {"x": float(e["pupil"][0] / w), "y": float(e["pupil"][1] / h)},
                        "psize": float(e["diam"])}

            ms = int((now % 1) * 1000)
            frame_json = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + ".%03d" % ms,
                "time": int((now - self.t0) * 1000),
                "fix": fix, "state": state,
                "raw": {"x": float(raw[0]), "y": float(raw[1])},
                "avg": {"x": float(avg[0]), "y": float(avg[1])},
            }
            if eyes:
                # Camera is not mirrored: subject's right eye is on the image's left.
                right, left = eyes[0], eyes[-1]
                frame_json["righteye"], frame_json["lefteye"] = eye_json(right), eye_json(left)
                frame_json["psize"] = float(np.mean([e["diam"] for e in eyes]))
            self.latest = frame_json

        msg = {"category": "tracker", "request": "get", "statuscode": 200,
               "values": {"frame": frame_json}}
        for c in list(self.clients):
            if c.push and c.alive:
                c.send(msg)
        return eyes

    # ---- Eye Tribe protocol --------------------------------------------------
    def tracker_values(self):
        with self.lock:
            return {
                "version": 1,
                "trackerstate": 0,
                "framerate": int(round(self.fps)) if self.fps else 60,
                "iscalibrated": self.calib.ready,
                "iscalibrating": self.calibrating,
                "calibresult": self.calib_result,
                "frame": self.latest,
                "screenindex": 0,
                "screenresw": self.screen[0],
                "screenresh": self.screen[1],
                "screenpsizew": round(self.screen[0] * MM_PER_PX / 1000, 4),
                "screenpsizeh": round(self.screen[1] * MM_PER_PX / 1000, 4),
                "heartbeatinterval": 3000,
            }

    def calibration_result(self):
        c = self.calib
        return {"result": True, "deg": c.deg(c.avg_err_px), "degl": c.deg(c.avg_err_px),
                "degr": c.deg(c.avg_err_px),
                "calibpoints": [{"state": 2, "cp": {"x": t[0], "y": t[1]},
                                 "mecp": {"x": t[0], "y": t[1]}, "acd": {"ad": c.deg(e), "adl": c.deg(e), "adr": c.deg(e)},
                                 "mepix": {"mean": e, "meanl": e, "meanr": e}}
                                for t, e in zip(c.targets, c.point_errs)]}

    def handle_request(self, client, msg):
        cat, req = msg.get("category"), msg.get("request")
        vals = msg.get("values")
        reply = {"category": cat, "request": req, "statuscode": 200}

        if cat == "heartbeat":
            return {"category": "heartbeat", "statuscode": 200}

        if cat == "tracker":
            if req == "set" and isinstance(vals, dict):
                if "push" in vals:
                    client.push = bool(vals["push"])
                return reply
            if req == "get":
                allv = self.tracker_values()
                keys = vals if isinstance(vals, list) else list(allv)
                reply["values"] = {k: allv[k] for k in keys if k in allv}
                return reply

        if cat == "calibration":
            with self.lock:
                if req == "start":
                    self.calib.reset()
                    self.calib_result = None
                    self.calibrating = True
                    self.calib_pointcount = int((vals or {}).get("pointcount", 9))
                    return reply
                if req == "pointstart" and isinstance(vals, dict):
                    self.calib_pending = (float(vals.get("x", 0)), float(vals.get("y", 0)))
                    self.calib.begin_point()
                    return reply
                if req == "pointend":
                    x, y = self.calib_pending or (0.0, 0.0)
                    if not self.calib.end_point(x, y):
                        reply.update(statuscode=800, statusmessage="Not enough eye samples for point")
                        return reply
                    if len(self.calib.targets) >= self.calib_pointcount:
                        self.calibrating = False
                        if self.calib.fit():
                            self.calib_result = self.calibration_result()
                            if self.args.calib_file:
                                self.calib.save(self.args.calib_file, self.screen)
                            reply["values"] = {"calibresult": self.calib_result}
                        else:
                            reply.update(statuscode=800, statusmessage="Calibration failed")
                    return reply
                if req in ("clear", "abort"):
                    self.calib.reset()
                    self.calib_result = None
                    self.calibrating = False
                    return reply

        reply.update(statuscode=501, statusmessage="Unsupported request")
        return reply

    def client_loop(self, c):
        buf, dec = "", json.JSONDecoder()
        while self.running and c.alive:
            try:
                data = c.sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buf += data.decode("utf-8", "replace")
            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                try:
                    msg, end = dec.raw_decode(buf)
                except json.JSONDecodeError:
                    if len(buf) > 65536:
                        buf = ""
                    break
                buf = buf[end:]
                if isinstance(msg, dict):
                    r = self.handle_request(c, msg)
                    if r:
                        c.send(r)
        c.alive = False
        if c in self.clients:
            self.clients.remove(c)
        try:
            c.sock.close()
        except OSError:
            pass

    def socket_listener(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(5)
        print(f"Eye Tribe TCP bridge listening on {HOST}:{PORT}")
        while self.running:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            conn.settimeout(1.0)
            c = Client(conn)
            self.clients.append(c)
            threading.Thread(target=self.client_loop, args=(c,), daemon=True).start()

    # ---- local calibration (--calibrate or 'c' key in --show) -----------------
    def run_local_calibration(self):
        w, h = self.screen
        n = max(3, min(5, int(getattr(self.args, "calib_grid", 3))))
        fracs = np.linspace(0.1, 0.9, n)
        pts = [(fx * w, fy * h) for fy in fracs for fx in fracs]
        settle, collect = 0.8, 1.2
        win = "EyeTribe calibration"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        with self.lock:
            self.calib.reset()
        ok = True
        for (x, y) in pts:
            t0 = time.time()
            started = False
            while time.time() - t0 < settle + collect:
                ret, frame = self.cap.read()
                if ret:
                    self.process(frame)
                el = time.time() - t0
                if el >= settle and not started:
                    with self.lock:
                        self.calib.begin_point()
                    started = True
                canvas = np.zeros((h, w, 3), np.uint8)
                r = int(18 + 10 * math.sin(el * 8)) if el < settle else 10
                cv2.circle(canvas, (int(x), int(y)), r, (255, 255, 255), 2)
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 255), -1)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    ok = False
                    break
            if not ok:
                break
            with self.lock:
                if not self.calib.end_point(x, y):
                    print(f"[calib] no eye data at ({x:.0f},{y:.0f}); point skipped")
        cv2.destroyWindow(win)
        cv2.waitKey(1)
        with self.lock:
            if ok and self.calib.fit():
                print(f"[calib] {self.calib.mode}, {len(self.calib.targets)} points, "
                      f"mean error {self.calib.avg_err_px:.1f}px (~{self.calib.deg(self.calib.avg_err_px):.2f} deg)")
                if self.args.calib_file:
                    self.calib.save(self.args.calib_file, self.screen)
                    print(f"[calib] saved {self.args.calib_file}")
                return True
        print("[calib] calibration aborted or failed")
        return False

    # ---- main loop -----------------------------------------------------------
    def _open_fullscreen(self, title):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def run_local_validation(self):
        """Held-out accuracy test: 8 targets the calibration never saw. Accuracy = how far the median
        gaze is from the target; precision = frame-to-frame spread. Measured on the smoothed output
        that clients receive, and on the unfiltered mapping for comparison."""
        if not self.calib.ready:
            print("[validate] not calibrated yet: run calibration first (press c, or use --calibrate)")
            return None
        w, h = self.screen
        fr = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.2), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5)]
        targets = [(fx * w, fy * h) for fx, fy in fr]
        settle, collect = 0.8, 1.2
        win = "EyeTribe validation"
        self._open_fullscreen(win)
        counts0 = list(self.det_counts)
        data, raw_data, frame_totals, ok = [], [], [], True
        for (x, y) in targets:
            t0, out_s, raw_s, frames = time.time(), [], [], 0
            while time.time() - t0 < settle + collect:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                self.process(frame)
                el = time.time() - t0
                if el >= settle and self.measured:
                    frames += 1
                    with self.lock:
                        f = self.latest
                    if f and (f["state"] & STATE_GAZE):
                        out_s.append(np.array([f["avg"]["x"], f["avg"]["y"]]))
                        raw_s.append(np.array([f["raw"]["x"], f["raw"]["y"]]))
                canvas = np.zeros((h, w, 3), np.uint8)
                r = int(18 + 10 * math.sin(el * 8)) if el < settle else 10
                cv2.circle(canvas, (int(x), int(y)), r, (255, 255, 255), 2)
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 255), -1)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    ok = False
                    break
            if not ok:
                break
            data.append(((x, y), out_s))
            raw_data.append(((x, y), raw_s))
            frame_totals.append(frames)
        cv2.destroyWindow(win)
        cv2.waitKey(1)
        if not ok or not data:
            print("[validate] aborted")
            return None
        rows, summary = validation_report(data, self.calib.deg)
        _, raw_summary = validation_report(raw_data, self.calib.deg)
        print("[validate] held-out targets (screen %dx%d), smoothed output:" % (w, h))
        for (target, acc, prec, n), total in zip(rows, frame_totals):
            if acc is None:
                print("   target (%4.0f,%4.0f): not enough eye data (gaze in %d of %d frames)" % (target[0], target[1], n, total))
            else:
                print("   target (%4.0f,%4.0f): accuracy %5.1f px (%.2f deg)   precision %5.1f px   gaze in %d of %d LED-lit frames" %
                      (target[0], target[1], acc, self.calib.deg(acc), prec, n, total))
        d = [a - b for a, b in zip(self.det_counts, counts0)]
        tot = max(sum(d), 1)
        print("[validate] eyes found per LED-lit frame: two %.0f%%, one %.0f%%, none %.0f%%" % (100 * d[2] / tot, 100 * d[1] / tot, 100 * d[0] / tot))
        if summary.get("acc_mean") is not None:
            print("[validate] SUMMARY (smoothed, what clients receive): accuracy %.1f px (%.2f deg), worst %.1f px (%.2f deg), "
                  "precision %.1f px (%.2f deg), %d/%d targets usable" %
                  (summary["acc_mean"], summary["acc_mean_deg"], summary["acc_max"], summary["acc_max_deg"],
                   summary["prec_mean"], summary["prec_mean_deg"], summary["points_used"], summary["points_total"]))
        if raw_summary.get("acc_mean") is not None:
            print("[validate] unfiltered mapping, for comparison: accuracy %.1f px (%.2f deg), precision %.1f px (%.2f deg)" %
                  (raw_summary["acc_mean"], raw_summary["acc_mean_deg"], raw_summary["prec_mean"], raw_summary["prec_mean_deg"]))
        return summary

    def run_gaze_demo(self):
        """Fullscreen black window with a dot at the smoothed gaze point. ESC or q to leave."""
        if not self.calib.ready:
            print("[gaze demo] not calibrated yet: run calibration first (press c, or use --calibrate)")
            return
        w, h = self.screen
        win = "EyeTribe gaze demo (ESC to exit)"
        self._open_fullscreen(win)
        trail = deque(maxlen=25)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.process(frame)
            with self.lock:
                f = self.latest
            canvas = np.zeros((h, w, 3), np.uint8)
            tracking = bool(f and (f["state"] & STATE_GAZE))
            if tracking:
                trail.append((int(f["avg"]["x"]), int(f["avg"]["y"])))
            else:
                trail.clear()
            for i, (tx, ty) in enumerate(trail):
                cv2.circle(canvas, (tx, ty), 4 + i // 4, (0, 120 + 5 * i, 0), -1)
            if trail:
                cv2.circle(canvas, trail[-1], 22, (0, 255, 0), 2)
            else:
                cv2.putText(canvas, "no eyes detected", (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        cv2.destroyWindow(win)
        cv2.waitKey(1)

    def snapshot(self, frame):
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.detector.detect(g)
        cv2.imwrite("snap_raw.png", g)
        stretched = self.detector.normalize(g, min_range=6)
        cv2.imwrite("snap_stretched.png", stretched)
        p1, p50, p999 = np.percentile(g[::4, ::4], (1, 50, 99.9))
        print("[snapshot] raw p1=%.0f median=%.0f p99.9=%.0f max=%d | eyes found: %d | detector: %s" %
              (p1, p50, p999, g.max(), len(eyes), self.detector.dbg))
        for line in self.detector.diagnose(g):
            print("    " + line)
        vis = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        for i, (x, y) in enumerate(self.detector.dbg_peaks):
            cv2.circle(vis, (int(x), int(y)), 18, (0, 255, 255), 2)
            cv2.putText(vis, str(i + 1), (int(x) + 20, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        for e in eyes:
            cv2.circle(vis, tuple(int(v) for v in e["pupil"]), int(e["diam"] / 2), (0, 255, 0), 2)
        cv2.imwrite("snap_annotated.png", vis)
        print("    saved snap_raw.png, snap_stretched.png, snap_annotated.png (numbered circles = strongest bright spots)")

    def draw_debug(self, frame, eyes):
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = cv2.cvtColor(self.detector.normalize(g, min_range=6), cv2.COLOR_GRAY2BGR)  # display-only stretch
        p1, p50, p999 = np.percentile(g[::8, ::8], (1, 50, 99.9))
        stage_colors = {0: (110, 110, 110), 1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 255, 0)}
        for bx, by, bd, st in self.detector.dbg_blobs:   # gray=too small/big, yellow=size ok, orange=round, green=glint near
            cv2.circle(vis, (int(bx), int(by)), max(3, int(bd / 2)), stage_colors[st], 1)
        for gx, gy in self.detector.dbg_glints:          # cyan = bright glint candidates
            cv2.circle(vis, (int(gx), int(gy)), 6, (255, 255, 0), 1)
        for e in eyes:
            cv2.circle(vis, tuple(int(v) for v in e["pupil"]), int(e["diam"] / 2), (0, 255, 0), 2)
            cv2.circle(vis, tuple(int(v) for v in e["glint"]), 4, (0, 0, 255), -1)
        scale = 900 / max(vis.shape[:2])
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
        cv2.putText(vis, "fps %.0f  eyes %d  calibrated %s   [c]alibrate [v]alidate [g]aze [q]uit" %
                    (self.fps, len(eyes), self.calib.ready), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        d = self.detector.dbg
        cv2.putText(vis, "glints %d | dark blobs %d -> size ok %d -> round %d -> glint near %d   [s]napshot" %
                    (d["glints"], d["blobs"], d["size"], d["round"], d["near"]), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(vis, "RAW brightness p1 %.0f  median %.0f  p99.9 %.0f  max %d   (healthy with you in view: median ~14+)" %
                    (p1, p50, p999, g.max()), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("EyeTribe debug", vis)
        return cv2.waitKey(1) & 0xFF

    def run(self):
        if not self.args.calibrate_only:
            threading.Thread(target=self.socket_listener, daemon=True).start()
        elif not self.args.calibrate:
            self.args.calibrate = True

        if self.args.calibrate:
            self.run_local_calibration()
            if self.args.calibrate_only:
                return
        if self.args.validate:
            self.run_local_validation()
        if self.args.gaze_demo:
            self.run_gaze_demo()

        last_ok, warned = time.time(), False
        t_run0, snap_done = time.time(), False
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    if not warned and time.time() - last_ok > 3:
                        print("[camera] WARNING: no frames for 3s. Try a direct USB port (no hub), "
                              "--width 0 --height 0 --fps 0, and check `eyetribe_uvc probe` formats.")
                        warned = True
                    time.sleep(0.01)
                    continue
                last_ok, warned = time.time(), False
                eyes = self.process(frame)
                if self.args.snap_after and not snap_done and self.measured and time.time() - t_run0 >= self.args.snap_after:
                    snap_done = True
                    print("[snapshot] taking snapshot now")
                    self.snapshot(self.last_lit)
                if self.args.show:
                    key = self.draw_debug(frame, eyes) if self.measured else (cv2.waitKey(1) & 0xFF)
                    if key == ord("q"):
                        break
                    if key == ord("s") and self.last_lit is not None:
                        self.snapshot(self.last_lit)
                    if key == ord("c"):
                        self.run_local_calibration()
                    if key == ord("v"):
                        self.run_local_validation()
                    if key == ord("g"):
                        self.run_gaze_demo()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()


def probe_cameras():
    for i in range(8):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            break
        ok, f = False, None
        for _ in range(30):  # cameras often need a moment before the first frame
            ok, f = cap.read()
            if ok:
                break
            time.sleep(0.1)
        print("camera %d: %s" % (i, "%dx%d" % (f.shape[1], f.shape[0]) if ok else "opened, but no frames after 3s"))
        cap.release()
    print("Names for these indices: run `system_profiler SPCameraDataType`. Use --show to see which is the tracker.")


def fourcc_str(v):
    v = int(v)
    chars = [(v >> (8 * i)) & 0xFF for i in range(4)]
    return "".join(chr(c) if 32 <= c < 127 else "?" for c in chars)


def diagnose(args):
    """Report what the camera really delivers (converted and raw) and save sample images."""
    for convert in (True, False):
        print("--- CONVERT_RGB=%d" % convert)
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("could not open camera %d" % args.camera)
            continue
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1 if convert else 0)
        print("reported: fourcc '%s', %dx%d, %.1f fps" % (
            fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)), cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)))
        frame, got = None, 0
        for _ in range(60):  # let auto-exposure / streaming settle
            ok, f = cap.read()
            if ok:
                frame, got = f, got + 1
            time.sleep(0.03)
        cap.release()
        if frame is None:
            print("no frames received")
            continue
        print("frames received: %d/60, shape %s, dtype %s" % (got, frame.shape, frame.dtype))
        print("min %s  max %s  mean %.2f" % (frame.min(), frame.max(), frame.mean()))
        if frame.ndim == 3:
            print("per-channel mean:", [round(float(frame[..., c].mean()), 2) for c in range(frame.shape[2])])
        if frame.dtype == np.uint8 and frame.ndim in (2, 3):
            name = "diag_convert%d.png" % convert
            cv2.imwrite(name, frame)
            print("saved", name)
            if frame.max() > frame.min():
                stretched = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite("diag_convert%d_stretched.png" % convert, stretched)
                print("saved diag_convert%d_stretched.png (contrast stretched)" % convert)
        else:
            np.save("diag_raw.npy", frame)
            print("saved diag_raw.npy")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["opencv", "stdin"], default="opencv",
                   help="stdin = raw frames from `eyetribe_stream --stdout` (needed on macOS)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1536)
    p.add_argument("--height", type=int, default=1536)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--screen", default=None, help="WxH in points (default: auto-detect)")
    p.add_argument("--calib-file", default="calibration.json")
    p.add_argument("--calib-grid", type=int, default=4, help="calibration grid size N (N x N points, 3-5; more points = steadier fit)")
    p.add_argument("--calibrate", action="store_true", help="run local 9-point calibration, then serve")
    p.add_argument("--calibrate-only", action="store_true", help="calibrate, save, and exit")
    p.add_argument("--show", action="store_true", help="show debug preview window")
    p.add_argument("--validate", action="store_true", help="run a held-out accuracy test at startup (needs calibration)")
    p.add_argument("--gaze-demo", action="store_true", help="show a live gaze dot at startup (needs calibration)")
    p.add_argument("--fix-deg", type=float, default=1.5, help="fixation radius in degrees of gaze (default 1.5)")
    p.add_argument("--fix-ms", type=int, default=150, help="minimum fixation duration in ms (default 150)")
    p.add_argument("--save-misses", type=int, default=0, help="save up to N frames where a glint pair was seen but no pupil was found")
    p.add_argument("--snap-after", type=float, default=0, help="save a diagnostic snapshot after this many seconds")
    p.add_argument("--probe-cameras", action="store_true")
    p.add_argument("--diagnose", action="store_true", help="print raw pixel stats and save sample frames")
    p.add_argument("--pupil-thr", type=int, default=0, help="(unused; pupil threshold is now automatic)")
    p.add_argument("--glint-thr", type=float, default=0, help="min glint strength (difference-of-Gaussians units; 0 = automatic from image noise)")
    p.add_argument("--no-normalize", action="store_true", help="disable automatic contrast stretch before detection")
    p.add_argument("--proc-scale", type=float, default=0.5, help="downscale for pupil search")
    a = p.parse_args()
    if a.screen:
        try:
            a.screen = tuple(int(v) for v in a.screen.lower().split("x"))
        except ValueError:
            p.error("--screen must look like 1512x982")
    else:
        a.screen = detect_screen_size()
    return a


if __name__ == "__main__":
    args = parse_args()
    if args.probe_cameras:
        probe_cameras()
        sys.exit(0)
    if args.diagnose:
        diagnose(args)
        sys.exit(0)
    print("[screen] %dx%d" % args.screen)
    EyeTribeBridgeServer(args).run()
