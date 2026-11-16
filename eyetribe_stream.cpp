// eyetribe_stream.cpp
//
// Streaming driver for the Eye Tribe on macOS (Apple Silicon). Negotiates a UVC
// mode over libusb, keeps several asynchronous bulk transfers in flight (no
// truncated frames), assembles frames from UVC payload headers, and either prints
// brightness stats or writes the frames to stdout for eyetribe_server.py.
//
// The tracker sends 8-bit grayscale packed in a "YUY2" container, so a mode listed
// as WxH is really a (2*W) x H grayscale image. Frames are passed on unconverted.
//
// Build:
//   brew install libusb pkg-config
//   clang++ -std=c++17 -O2 eyetribe_stream.cpp -o eyetribe_stream $(pkg-config --cflags --libs libusb-1.0)
//
// Usage (needs sudo; quit other camera apps first):
//   sudo ./eyetribe_stream                          5 s of stats, nothing written
//   sudo ./eyetribe_stream --xu-scan                read all extension-unit controls
//   sudo ./eyetribe_stream --xu 0x03=0x0f           write an XU control, watch brightness
//   sudo ./eyetribe_stream --leds 0xff --stdout | python eyetribe_server.py --source stdin --show
//
// Options:
//   --frame N      mode index from `eyetribe_uvc probe` (default 3 = 768x1024 -> 1536x1024 gray @ 60 fps)
//   --format N     format index (default 1)
//   --fps N        request a frame rate (default 0 = device default for the mode)
//   --seconds S    run time (default: 5 without --stdout, unlimited with it)
//   --stdout       write frames to stdout (stats go to stderr)
//   --leds MASK    turn the illumination on: sets LED-bank mask (control 0x03; 0xff = all four banks)
//                  and then rewrites control 0x04 with its current value. The tracker stays dark
//                  until 0x04 is written (found experimentally: the value does not matter, any
//                  write starts it), and it keeps that state until power-cycled. Use this at every start.
//   --xu SEL=B[,B...]  write bytes to extension-unit control SEL after streaming starts (repeatable)
//   --xu-scan      read GET_INFO/GET_LEN/GET_CUR for selectors 1..16, then exit
//   --sweep SEL:V1,V2,...  while streaming, write each value to control SEL and report image
//                  brightness; the original value is restored afterwards. Values are integers
//                  (little-endian, sized to the control) or raw bytes written as xHEX (e.g. x901f901f).
//                  Repeatable. On unit 3 it refuses selectors 7, 8, 9 (identifier/signature data).
//   --dwell S      seconds per sweep value (default 1.5)
//   --start-delay S  wait S seconds (with a countdown) before sweeping, so you can get in position
//   --interleave   after each swept value, restore the original and re-measure it, so setting
//                  effects can be told apart from scene changes (e.g. you moving)
//   --uvc-scan     list the camera's standard controls (exposure, gain, brightness...) with
//                  min/max/current values, then exit. Sweep one with --unit ID --sweep SEL:...
//                  (the ID and selector come from the scan; --unit changes the target unit).
//   --unit N       extension unit id (default 3), --vcif N video-control interface (default 0)
//   --vid/--pid    device ids (default 0x29ab / 0x00fb)
//
// Stdout format, per frame (little-endian): "ETF1", u32 width, u32 height, u32 sequence,
// then width*height bytes of 8-bit grayscale.
//
// CAUTION: --xu writes to unknown device controls. Write only single, small values, and
// unplug/replug the tracker if it misbehaves.

#include <libusb.h>
#include <sys/time.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Mode { int w, h, fps; };  // fps = device default for the mode (from `eyetribe_uvc probe`)
const Mode MODES[] = {{1152, 1536, 27}, {1036, 1536, 30}, {768, 1024, 60}, {768, 1536, 42},
                      {2304, 1536, 42}, {2072, 1536, 27}, {1536, 1024, 30}, {1536, 1536, 60}};

constexpr uint8_t UVC_SET_CUR = 0x01, UVC_GET_CUR = 0x81, UVC_GET_LEN = 0x85, UVC_GET_INFO = 0x86;
constexpr uint8_t VS_PROBE = 0x01, VS_COMMIT = 0x02;
constexpr unsigned CTRL_TIMEOUT = 1000;

std::atomic<bool> g_stop{false};
void on_sigint(int) { g_stop = true; }

struct Args {
    unsigned vid = 0x29ab, pid = 0x00fb;
    int format = 1, frame = 3, fps = 0, unit = 3, vcif = 0, ntransfers = 8;
    double seconds = -1;
    bool to_stdout = false, scan = false, uvcscan = false;
    std::vector<std::string> xu, sweeps;
    int leds = -1;  // LED bank mask; -1 = leave the illumination alone
    double dwell = 1.5, start_delay = 0;
    bool interleave = false;
};

uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24); }
void wr32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (v >> (8 * i)) & 0xFF; }

void print_hex(FILE *f, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; ++i) std::fprintf(f, "%02x%s", p[i], i + 1 < n ? " " : "");
}

// ----------------------------------------------------------------- UVC ctrl --

int vs_get(libusb_device_handle *h, uint8_t req, uint8_t sel, int ifnum, uint8_t *buf, uint16_t len) {
    return libusb_control_transfer(h, LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_CLASS | LIBUSB_RECIPIENT_INTERFACE,
                                   req, static_cast<uint16_t>(sel << 8), static_cast<uint16_t>(ifnum), buf, len, CTRL_TIMEOUT);
}
int vs_set(libusb_device_handle *h, uint8_t sel, int ifnum, uint8_t *buf, uint16_t len) {
    return libusb_control_transfer(h, LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_CLASS | LIBUSB_RECIPIENT_INTERFACE,
                                   UVC_SET_CUR, static_cast<uint16_t>(sel << 8), static_cast<uint16_t>(ifnum), buf, len, CTRL_TIMEOUT);
}
uint16_t xu_index(const Args &A) { return static_cast<uint16_t>((A.unit << 8) | A.vcif); }

int xu_get(libusb_device_handle *h, const Args &A, uint8_t req, uint8_t sel, uint8_t *buf, uint16_t len) {
    return libusb_control_transfer(h, LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_CLASS | LIBUSB_RECIPIENT_INTERFACE,
                                   req, static_cast<uint16_t>(sel << 8), xu_index(A), buf, len, CTRL_TIMEOUT);
}
int xu_set(libusb_device_handle *h, const Args &A, uint8_t sel, std::vector<uint8_t> &data) {
    return libusb_control_transfer(h, LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_CLASS | LIBUSB_RECIPIENT_INTERFACE,
                                   UVC_SET_CUR, static_cast<uint16_t>(sel << 8), xu_index(A), data.data(),
                                   static_cast<uint16_t>(data.size()), CTRL_TIMEOUT);
}

void xu_scan(libusb_device_handle *h, const Args &A) {
    std::fprintf(stderr, "Extension unit %d, selectors 1..16 (info bit0=get, bit1=set):\n", A.unit);
    for (int sel = 1; sel <= 16; ++sel) {
        uint8_t info = 0, lenb[2] = {0, 0};
        int ri = xu_get(h, A, UVC_GET_INFO, static_cast<uint8_t>(sel), &info, 1);
        int rl = xu_get(h, A, UVC_GET_LEN, static_cast<uint8_t>(sel), lenb, 2);
        int len = (rl == 2) ? (lenb[0] | (lenb[1] << 8)) : -1;
        std::fprintf(stderr, "  0x%02x: ", sel);
        if (ri == 1) std::fprintf(stderr, "info=0x%02x ", info); else std::fprintf(stderr, "info=%s ", libusb_error_name(ri));
        if (len >= 0) std::fprintf(stderr, "len=%d ", len); else std::fprintf(stderr, "len=%s ", libusb_error_name(rl));
        if (len > 0 && len <= 64) {
            std::vector<uint8_t> cur(len);
            int rc = xu_get(h, A, UVC_GET_CUR, static_cast<uint8_t>(sel), cur.data(), static_cast<uint16_t>(len));
            if (rc > 0) { std::fprintf(stderr, "cur="); print_hex(stderr, cur.data(), rc); }
            else std::fprintf(stderr, "cur=%s", libusb_error_name(rc));
        }
        std::fprintf(stderr, "\n");
    }
}

bool apply_xu(libusb_device_handle *h, const Args &A, const std::string &spec) {
    size_t eq = spec.find('=');
    if (eq == std::string::npos) { std::fprintf(stderr, "--xu needs SEL=BYTE[,BYTE...]: %s\n", spec.c_str()); return false; }
    unsigned long sel = std::strtoul(spec.substr(0, eq).c_str(), nullptr, 0);
    std::vector<uint8_t> data;
    size_t pos = eq + 1;
    while (pos <= spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (tok.empty()) break;
        unsigned long b = std::strtoul(tok.c_str(), nullptr, 0);
        if (b > 255) { std::fprintf(stderr, "byte out of range: %s\n", tok.c_str()); return false; }
        data.push_back(static_cast<uint8_t>(b));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (sel < 1 || sel > 255 || data.empty()) { std::fprintf(stderr, "bad --xu spec: %s\n", spec.c_str()); return false; }

    uint8_t lenb[2] = {0, 0};
    if (xu_get(h, A, UVC_GET_LEN, static_cast<uint8_t>(sel), lenb, 2) == 2) {
        int len = lenb[0] | (lenb[1] << 8);
        if (static_cast<size_t>(len) != data.size())
            std::fprintf(stderr, "warning: selector 0x%02lx expects %d byte(s), sending %zu\n", sel, len, data.size());
    }
    int r = xu_set(h, A, static_cast<uint8_t>(sel), data);
    if (r < 0) { std::fprintf(stderr, "XU set 0x%02lx failed: %s\n", sel, libusb_error_name(r)); return false; }
    std::fprintf(stderr, "XU set 0x%02lx = ", sel);
    print_hex(stderr, data.data(), data.size());
    std::fprintf(stderr, " ok\n");
    return true;
}

bool find_streaming(libusb_device *dev, int &ifnum, int &ep_addr) {
    libusb_config_descriptor *cfg = nullptr;
    if (libusb_get_active_config_descriptor(dev, &cfg) != 0 || !cfg) return false;
    bool found = false;
    for (int i = 0; i < cfg->bNumInterfaces && !found; ++i)
        for (int a = 0; a < cfg->interface[i].num_altsetting && !found; ++a) {
            const libusb_interface_descriptor &alt = cfg->interface[i].altsetting[a];
            if (alt.bInterfaceClass == 14 && alt.bInterfaceSubClass == 2 && alt.bNumEndpoints > 0) {
                ifnum = alt.bInterfaceNumber;
                ep_addr = alt.endpoint[0].bEndpointAddress;
                found = true;
            }
        }
    libusb_free_config_descriptor(cfg);
    return found;
}

// ------------------------------------------------------------------ streaming --

struct Stream {
    size_t expected = 0;
    uint32_t gw = 0, gh = 0, seq = 0;
    std::vector<uint8_t> cur;
    int lastfid = -1;
    bool skip_frame = false;

    std::mutex m;
    std::condition_variable cv;
    std::vector<uint8_t> latest;
    bool has = false, stopping = false;
    uint32_t latest_seq = 0;

    std::atomic<uint64_t> frames{0}, dropped{0}, bad{0}, errors{0}, timeouts{0}, stalls{0};
    bool writing = false;
    std::atomic<int> inflight{0}, last_mean{0}, last_max{0}, peak{0}, last_std100{0};
    std::atomic<uint64_t> mean_sum{0}, mean_n{0}, std_sum100{0};
    std::atomic<bool> running{true};
};

void end_frame(Stream *s) {
    if (s->cur.empty()) return;
    if (s->skip_frame || s->cur.size() != s->expected) {
        s->bad++;
        s->seq++;   // keep the sequence number equal to the camera frame index (parity = lit/unlit)
    } else {
        uint64_t sum = 0, sumsq = 0, n = 0;
        int mx = 0;
        for (size_t i = 0; i < s->cur.size(); i += 97) {
            unsigned v = s->cur[i];
            sum += v;
            sumsq += v * v;
            mx = v > static_cast<unsigned>(mx) ? static_cast<int>(v) : mx;
            ++n;
        }
        if (n == 0) n = 1;
        double mean = double(sum) / n, var = double(sumsq) / n - mean * mean;
        s->last_std100 = static_cast<int>(std::sqrt(var > 0 ? var : 0) * 100);
        s->std_sum100 += static_cast<uint64_t>(s->last_std100.load());
        s->last_mean = static_cast<int>(sum / n);
        s->last_max = mx;
        s->mean_sum += static_cast<uint64_t>(s->last_mean.load());
        s->mean_n++;
        int pk = s->peak.load();
        while (mx > pk && !s->peak.compare_exchange_weak(pk, mx)) {}
        s->frames++;
        std::lock_guard<std::mutex> lk(s->m);
        if (s->has && s->writing) s->dropped++;
        s->latest.swap(s->cur);
        s->latest_seq = s->seq++;
        s->has = true;
        s->cv.notify_one();
    }
    s->cur.clear();
    s->skip_frame = false;
}

void process(Stream *s, const uint8_t *b, int n) {
    if (n < 2) return;
    int hl = b[0];
    if (hl < 2 || hl > 12 || hl > n) { s->bad++; s->skip_frame = true; return; }
    int info = b[1], fid = info & 1;
    if (s->lastfid >= 0 && fid != s->lastfid) end_frame(s);  // FID toggle = new frame
    s->lastfid = fid;
    if (info & 0x40) s->skip_frame = true;                   // device flagged an error
    s->cur.insert(s->cur.end(), b + hl, b + n);
    if (info & 0x02) end_frame(s);                           // EOF bit
}

const char *status_name(int st) {
    switch (st) {
    case LIBUSB_TRANSFER_COMPLETED: return "completed";
    case LIBUSB_TRANSFER_ERROR: return "error";
    case LIBUSB_TRANSFER_TIMED_OUT: return "timed out";
    case LIBUSB_TRANSFER_CANCELLED: return "cancelled";
    case LIBUSB_TRANSFER_STALL: return "stall";
    case LIBUSB_TRANSFER_NO_DEVICE: return "no device";
    case LIBUSB_TRANSFER_OVERFLOW: return "overflow";
    default: return "unknown";
    }
}

void LIBUSB_CALL on_transfer(libusb_transfer *t) {
    Stream *s = static_cast<Stream *>(t->user_data);
    switch (t->status) {
    case LIBUSB_TRANSFER_COMPLETED:
    case LIBUSB_TRANSFER_OVERFLOW:
        process(s, t->buffer, t->actual_length);
        break;
    case LIBUSB_TRANSFER_TIMED_OUT:
        s->timeouts++;
        break;
    case LIBUSB_TRANSFER_CANCELLED:
        s->inflight--;
        return;
    case LIBUSB_TRANSFER_NO_DEVICE:
        std::fprintf(stderr, "device disconnected (USB reports no device)\n");
        s->running = false;
        s->inflight--;
        return;
    case LIBUSB_TRANSFER_STALL:
        s->stalls++;
        if (s->errors++ < 10) std::fprintf(stderr, "transfer stalled; clearing halt\n");
        libusb_clear_halt(t->dev_handle, t->endpoint);
        break;
    default:
        if (s->errors++ < 10) std::fprintf(stderr, "transfer status: %s\n", status_name(t->status));
        if (s->errors > 200) { std::fprintf(stderr, "too many transfer errors, stopping\n"); s->running = false; }
        break;
    }
    if (s->running) {
        int r = libusb_submit_transfer(t);
        if (r == 0) return;
        std::fprintf(stderr, "resubmit failed: %s (transfers still in flight: %d)\n", libusb_error_name(r),
                     s->inflight.load() - 1);
    }
    s->inflight--;
}

void writer_thread(Stream *s) {
    std::vector<uint8_t> buf;
    uint8_t hdr[16] = {'E', 'T', 'F', '1'};
    wr32(hdr + 4, s->gw);
    wr32(hdr + 8, s->gh);
    for (;;) {
        uint32_t seq = 0;
        {
            std::unique_lock<std::mutex> lk(s->m);
            s->cv.wait_for(lk, std::chrono::milliseconds(200), [&] { return s->has || s->stopping; });
            if (!s->has) {
                if (s->stopping) break;
                continue;
            }
            buf.swap(s->latest);
            seq = s->latest_seq;
            s->has = false;
        }
        wr32(hdr + 12, seq);
        if (std::fwrite(hdr, 1, sizeof hdr, stdout) != sizeof hdr ||
            std::fwrite(buf.data(), 1, buf.size(), stdout) != buf.size() || std::fflush(stdout) != 0) {
            std::fprintf(stderr, "stdout closed; stopping\n");
            g_stop = true;
            break;
        }
    }
}

struct Ctl { int bit; uint8_t sel; const char *name; };
const Ctl CT_CTLS[] = {{0, 1, "scanning mode"}, {1, 2, "auto-exposure mode"}, {2, 3, "auto-exposure priority"},
                       {3, 4, "exposure time (abs)"}, {4, 5, "exposure time (rel)"}, {5, 6, "focus (abs)"},
                       {6, 7, "focus (rel)"}, {7, 9, "iris (abs)"}, {8, 10, "iris (rel)"}, {9, 11, "zoom (abs)"},
                       {10, 12, "zoom (rel)"}, {11, 13, "pan/tilt (abs)"}, {12, 14, "pan/tilt (rel)"},
                       {13, 15, "roll (abs)"}, {14, 16, "roll (rel)"}, {17, 8, "focus auto"}, {18, 17, "privacy"}};
const Ctl PU_CTLS[] = {{0, 2, "brightness"}, {1, 3, "contrast"}, {2, 6, "hue"}, {3, 7, "saturation"},
                       {4, 8, "sharpness"}, {5, 9, "gamma"}, {6, 10, "white balance temp"},
                       {7, 11, "white balance component"}, {8, 1, "backlight compensation"}, {9, 4, "gain"},
                       {10, 5, "power line frequency"}, {11, 16, "hue auto"}, {12, 12, "white balance temp auto"},
                       {13, 13, "white balance component auto"}, {14, 14, "digital multiplier"},
                       {15, 15, "digital multiplier limit"}};

struct VCUnit { int id; bool camera; std::vector<uint8_t> bm; };

bool parse_vc_units(libusb_device *dev, std::vector<VCUnit> &out, int &vcif) {
    libusb_config_descriptor *cfg = nullptr;
    if (libusb_get_active_config_descriptor(dev, &cfg) != 0 || !cfg) return false;
    bool found = false;
    for (int i = 0; i < cfg->bNumInterfaces; ++i)
        for (int a = 0; a < cfg->interface[i].num_altsetting; ++a) {
            const libusb_interface_descriptor &alt = cfg->interface[i].altsetting[a];
            if (alt.bInterfaceClass != 14 || alt.bInterfaceSubClass != 1) continue;
            found = true;
            vcif = alt.bInterfaceNumber;
            const uint8_t *p = alt.extra;
            int n = alt.extra_length;
            while (n >= 3) {
                int len = p[0];
                if (len < 3 || len > n) break;
                if (p[1] == 0x24) {
                    if (p[2] == 0x02 && len >= 15 && (p[4] | (p[5] << 8)) == 0x0201) {  // camera input terminal
                        VCUnit u{p[3], true, {}};
                        for (int k = 0; k < p[14] && 15 + k < len; ++k) u.bm.push_back(p[15 + k]);
                        out.push_back(u);
                    } else if (p[2] == 0x05 && len >= 8) {  // processing unit
                        VCUnit u{p[3], false, {}};
                        for (int k = 0; k < p[7] && 8 + k < len; ++k) u.bm.push_back(p[8 + k]);
                        out.push_back(u);
                    }
                }
                p += len;
                n -= len;
            }
        }
    libusb_free_config_descriptor(cfg);
    return found;
}

int unit_get(libusb_device_handle *h, int vcif, int unit, uint8_t req, uint8_t sel, uint8_t *buf, uint16_t len) {
    return libusb_control_transfer(h, LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_CLASS | LIBUSB_RECIPIENT_INTERFACE,
                                   req, static_cast<uint16_t>(sel << 8), static_cast<uint16_t>((unit << 8) | vcif),
                                   buf, len, CTRL_TIMEOUT);
}

std::string fmt_val(const uint8_t *b, size_t n) {
    char t[160];
    if (n <= 8) {
        unsigned long long v = 0;
        for (size_t i = 0; i < n; ++i) v |= static_cast<unsigned long long>(b[i]) << (8 * i);
        std::snprintf(t, sizeof t, "%llu", v);
        return t;
    }
    std::string out;
    for (size_t i = 0; i < n; ++i) { std::snprintf(t, sizeof t, "%02x", b[i]); out += t; }
    return out;
}

void uvc_scan(libusb_device_handle *h) {
    std::vector<VCUnit> units;
    int vcif = 0;
    if (!parse_vc_units(libusb_get_device(h), units, vcif)) { std::fprintf(stderr, "No video-control interface found.\n"); return; }
    for (const auto &u : units) {
        std::fprintf(stderr, "Unit %d (%s), control bitmap: ", u.id, u.camera ? "camera terminal" : "processing unit");
        print_hex(stderr, u.bm.data(), u.bm.size());
        std::fprintf(stderr, "\n");
        const Ctl *tbl = u.camera ? CT_CTLS : PU_CTLS;
        size_t cnt = u.camera ? sizeof CT_CTLS / sizeof CT_CTLS[0] : sizeof PU_CTLS / sizeof PU_CTLS[0];
        for (size_t i = 0; i < cnt; ++i) {
            size_t byte = tbl[i].bit / 8;
            if (byte >= u.bm.size() || !(u.bm[byte] & (1 << (tbl[i].bit % 8)))) continue;
            uint8_t info = 0, lenb[2] = {0, 0};
            int ri = unit_get(h, vcif, u.id, UVC_GET_INFO, tbl[i].sel, &info, 1);
            int rl = unit_get(h, vcif, u.id, UVC_GET_LEN, tbl[i].sel, lenb, 2);
            int len = rl == 2 ? (lenb[0] | (lenb[1] << 8)) : 0;
            std::fprintf(stderr, "  sel 0x%02x %-28s", tbl[i].sel, tbl[i].name);
            if (ri != 1 || len <= 0 || len > 64) { std::fprintf(stderr, " (not readable: %s)\n", libusb_error_name(ri != 1 ? ri : rl)); continue; }
            std::fprintf(stderr, " info=0x%02x len=%d", info, len);
            const struct { const char *tag; uint8_t req; } reqs[] = {{"min", 0x82}, {"max", 0x83}, {"cur", UVC_GET_CUR}, {"def", 0x87}};
            for (const auto &q : reqs) {
                std::vector<uint8_t> b(len);
                int r = unit_get(h, vcif, u.id, q.req, tbl[i].sel, b.data(), static_cast<uint16_t>(len));
                if (r == len) std::fprintf(stderr, " %s=%s", q.tag, fmt_val(b.data(), len).c_str());
            }
            std::fprintf(stderr, "\n");
        }
    }
}

// Turn the illumination on: LED-bank mask (0x03), then rewrite 0x04 to start the sensor/strobe.
bool init_illumination(libusb_device_handle *h, const Args &A) {
    std::vector<uint8_t> mask = {static_cast<uint8_t>(A.leds & 0xFF), static_cast<uint8_t>((A.leds >> 8) & 0xFF)};
    int r = xu_set(h, A, 0x03, mask);
    if (r < 0) { std::fprintf(stderr, "LED mask write failed: %s\n", libusb_error_name(r)); return false; }

    std::vector<uint8_t> cur(8);
    int got = xu_get(h, A, UVC_GET_CUR, 0x04, cur.data(), 8);
    if (got != 8) cur = {0x90, 0x1f, 0x90, 0x1f, 0x90, 0x1f, 0x90, 0x1f};  // factory value seen in the scan
    r = xu_set(h, A, 0x04, cur);
    if (r < 0) { std::fprintf(stderr, "control 0x04 write failed: %s\n", libusb_error_name(r)); return false; }
    std::fprintf(stderr, "Illumination on: LED mask 0x%04x, control 0x04 rewritten (", A.leds);
    print_hex(stderr, cur.data(), cur.size());
    std::fprintf(stderr, ")\n");
    return true;
}

void pump(libusb_context *ctx, double secs) {
    auto t0 = std::chrono::steady_clock::now();
    while (!g_stop && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        timeval tv{0, 50000};
        libusb_handle_events_timeout(ctx, &tv);
    }
}

bool encode_value(const std::string &tok, size_t len, std::vector<uint8_t> &out) {
    out.clear();
    if (!tok.empty() && (tok[0] == 'x' || tok[0] == 'X')) {
        std::string hex = tok.substr(1);
        if (hex.size() % 2 != 0 || hex.empty()) return false;
        for (size_t i = 0; i < hex.size(); i += 2) {
            char *end = nullptr;
            std::string byte = hex.substr(i, 2);
            unsigned long v = std::strtoul(byte.c_str(), &end, 16);
            if (!end || *end) return false;
            out.push_back(static_cast<uint8_t>(v));
        }
        return out.size() == len;
    }
    if (len == 0 || len > 8) return false;
    char *end = nullptr;
    unsigned long long v = std::strtoull(tok.c_str(), &end, 0);
    if (!end || *end) return false;
    if (len < 8 && (v >> (8 * len)) != 0) return false;
    for (size_t i = 0; i < len; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    return true;
}

void run_sweep(libusb_context *ctx, libusb_device_handle *h, const Args &A, Stream &S, const std::string &spec) {
    size_t colon = spec.find(':');
    if (colon == std::string::npos) { std::fprintf(stderr, "--sweep needs SEL:V1,V2,...: %s\n", spec.c_str()); return; }
    unsigned long sel = std::strtoul(spec.substr(0, colon).c_str(), nullptr, 0);
    if (sel < 1 || sel > 16) { std::fprintf(stderr, "sweep selector must be 1..16\n"); return; }
    if (A.unit == 3 && sel >= 7 && sel <= 9) {
        std::fprintf(stderr, "Refusing to write selector 0x%02lx (holds identifier/signature data).\n", sel);
        return;
    }
    uint8_t lenb[2] = {0, 0};
    if (xu_get(h, A, UVC_GET_LEN, static_cast<uint8_t>(sel), lenb, 2) != 2) {
        std::fprintf(stderr, "selector 0x%02lx: cannot read length (unsupported?)\n", sel);
        return;
    }
    size_t len = lenb[0] | (lenb[1] << 8);
    if (len == 0 || len > 64) { std::fprintf(stderr, "selector 0x%02lx: unsupported length %zu\n", sel, len); return; }
    std::vector<uint8_t> orig(len);
    int r = xu_get(h, A, UVC_GET_CUR, static_cast<uint8_t>(sel), orig.data(), static_cast<uint16_t>(len));
    if (r != static_cast<int>(len)) { std::fprintf(stderr, "selector 0x%02lx: cannot read current value\n", sel); return; }

    std::fprintf(stderr, "Sweep selector 0x%02lx (%zu bytes), original value: ", sel, len);
    print_hex(stderr, orig.data(), len);
    std::fprintf(stderr, "\n");

    auto measure = [&](const char *label) {
        S.mean_sum = 0; S.mean_n = 0; S.peak = 0; S.std_sum100 = 0;
        pump(ctx, A.dwell);
        uint64_t n = S.mean_n.load();
        if (n == 0) std::fprintf(stderr, "  %-22s no frames received\n", label);
        else std::fprintf(stderr, "  %-22s mean %6.1f  std %5.2f  peak %3d  (%.0f fps)\n", label,
                          double(S.mean_sum.load()) / n, double(S.std_sum100.load()) / (100.0 * n), S.peak.load(),
                          n / A.dwell);
    };

    measure("original");
    std::string list = spec.substr(colon + 1);
    size_t pos = 0;
    while (pos <= list.size() && !g_stop) {
        size_t comma = list.find(',', pos);
        std::string tok = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            std::vector<uint8_t> val;
            if (!encode_value(tok, len, val)) {
                std::fprintf(stderr, "  %-22s skipped (cannot encode for %zu-byte control)\n", tok.c_str(), len);
            } else {
                int wr = xu_set(h, A, static_cast<uint8_t>(sel), val);
                if (wr < 0) std::fprintf(stderr, "  %-22s write failed: %s\n", tok.c_str(), libusb_error_name(wr));
                else {
                    measure(tok.c_str());
                    if (A.interleave) {
                        xu_set(h, A, static_cast<uint8_t>(sel), orig);
                        measure("  (back to original)");
                    }
                }
            }
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    int rr = xu_set(h, A, static_cast<uint8_t>(sel), orig);
    std::fprintf(stderr, "  restored original value: %s\n", rr < 0 ? libusb_error_name(rr) : "ok");
}

int run(const Args &A, libusb_context *ctx, libusb_device_handle *h) {
    int ifnum = -1, ep = 0;
    if (!find_streaming(libusb_get_device(h), ifnum, ep)) {
        std::fprintf(stderr, "No streaming interface with an endpoint found.\n");
        return 1;
    }

    // Claim every interface (control first), like libuvc; auto-detach captured the device.
    std::vector<int> claimed;
    bool stream_ok = false;
    libusb_config_descriptor *cfg = nullptr;
    if (libusb_get_active_config_descriptor(libusb_get_device(h), &cfg) == 0 && cfg) {
        for (int i = 0; i < cfg->bNumInterfaces; ++i) {
            int num = cfg->interface[i].altsetting[0].bInterfaceNumber;
            int cr = libusb_claim_interface(h, num);
            if (cr == 0) claimed.push_back(num);
            else std::fprintf(stderr, "claim interface %d: %s\n", num, libusb_error_name(cr));
            if (num == ifnum && cr == 0) stream_ok = true;
        }
        libusb_free_config_descriptor(cfg);
    }
    auto release_all = [&]() { for (int n : claimed) libusb_release_interface(h, n); };
    if (!stream_ok) {
        std::fprintf(stderr, "Could not claim the streaming interface (run with sudo; quit other camera apps).\n");
        release_all();
        return 1;
    }

    if (A.scan) {
        xu_scan(h, A);
        release_all();
        return 0;
    }
    if (A.uvcscan) {
        uvc_scan(h);
        release_all();
        return 0;
    }

    // ---- negotiate ----
    const Mode mode = MODES[A.frame - 1];
    Stream S;
    S.expected = size_t(mode.w) * mode.h * 2;
    S.gw = mode.w * 2;
    S.gh = mode.h;

    uint8_t lenb[2] = {0, 0};
    int ctl_len = 26;
    if (vs_get(h, UVC_GET_LEN, VS_PROBE, ifnum, lenb, 2) == 2) ctl_len = lenb[0] | (lenb[1] << 8);
    if (ctl_len < 26 || ctl_len > 48) ctl_len = 26;
    std::vector<uint8_t> ctl(48, 0), neg(48, 0);
    vs_get(h, UVC_GET_CUR, VS_PROBE, ifnum, ctl.data(), static_cast<uint16_t>(ctl_len));
    ctl[0] = 1; ctl[1] = 0;
    ctl[2] = static_cast<uint8_t>(A.format);
    ctl[3] = static_cast<uint8_t>(A.frame);
    wr32(&ctl[4], static_cast<uint32_t>(1e7 / (A.fps > 0 ? A.fps : mode.fps)));
    int r = vs_set(h, VS_PROBE, ifnum, ctl.data(), static_cast<uint16_t>(ctl_len));
    if (r < 0) { std::fprintf(stderr, "SET_CUR probe failed: %s\n", libusb_error_name(r)); release_all(); return 1; }
    r = vs_get(h, UVC_GET_CUR, VS_PROBE, ifnum, neg.data(), static_cast<uint16_t>(ctl_len));
    if (r < 0) { std::fprintf(stderr, "GET_CUR probe failed: %s\n", libusb_error_name(r)); release_all(); return 1; }
    if (neg[3] != A.frame) std::fprintf(stderr, "warning: device chose frame %u instead of %d\n", neg[3], A.frame);
    double fps = rd32(&neg[4]) ? 1e7 / rd32(&neg[4]) : 0;
    r = vs_set(h, VS_COMMIT, ifnum, neg.data(), static_cast<uint16_t>(ctl_len));
    if (r < 0) { std::fprintf(stderr, "SET_CUR commit failed: %s\n", libusb_error_name(r)); release_all(); return 1; }
    libusb_clear_halt(h, static_cast<unsigned char>(ep));
    std::fprintf(stderr, "Streaming mode %d: %ux%u grayscale @ %.1f fps (maxPayload %u)\n",
                 A.frame, S.gw, S.gh, fps, rd32(&neg[22]));

    // ---- start async transfers ----
    size_t bufsize = rd32(&neg[22]);
    if (bufsize < 16384) bufsize = 16384;
    if (bufsize > (1u << 20)) bufsize = 1u << 20;
    std::vector<std::vector<uint8_t>> bufs(A.ntransfers, std::vector<uint8_t>(bufsize));
    std::vector<libusb_transfer *> xfers;
    for (int i = 0; i < A.ntransfers; ++i) {
        libusb_transfer *t = libusb_alloc_transfer(0);
        libusb_fill_bulk_transfer(t, h, static_cast<unsigned char>(ep), bufs[i].data(), static_cast<int>(bufsize),
                                  on_transfer, &S, 1000);
        if (libusb_submit_transfer(t) == 0) { S.inflight++; xfers.push_back(t); }
        else { libusb_free_transfer(t); std::fprintf(stderr, "submit failed for transfer %d\n", i); }
    }
    if (xfers.empty()) { release_all(); return 1; }

    std::thread writer;
    S.writing = A.to_stdout;
    if (A.to_stdout) writer = std::thread(writer_thread, &S);

    if (A.leds >= 0) init_illumination(h, A);
    for (const auto &spec : A.xu) apply_xu(h, A, spec);
    if (!A.sweeps.empty()) {
        std::fprintf(stderr, "NOTE: sweep writes to undocumented controls; values are restored afterwards.\n");
        if (A.start_delay > 0) {
            std::fprintf(stderr, "Get in position (face the tracker, sit still). Sweep starts in %.0f s...\n", A.start_delay);
            pump(ctx, A.start_delay);
        }
        pump(ctx, 1.0);  // let streaming settle
        for (const auto &spec : A.sweeps) run_sweep(ctx, h, A, S, spec);
    }

    // ---- event loop ----
    auto t0 = std::chrono::steady_clock::now(), last = t0;
    uint64_t last_frames = 0;
    int quiet = 0;
    while (!g_stop && S.running) {
        timeval tv{0, 100000};
        libusb_handle_events_timeout(ctx, &tv);
        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - last).count();
        if (since >= 1.0) {
            uint64_t f = S.frames.load();
            std::fprintf(stderr, "[stream] %.1f fps | mean %d std %.2f max %d | dropped %llu bad %llu | inflight %d errors %llu timeouts %llu\n",
                         (f - last_frames) / since, S.last_mean.load(), S.last_std100.load() / 100.0, S.last_max.load(),
                         (unsigned long long)S.dropped.load(), (unsigned long long)S.bad.load(), S.inflight.load(),
                         (unsigned long long)S.errors.load(), (unsigned long long)S.timeouts.load());
            if (f == last_frames) {
                if (++quiet >= 2)
                    std::fprintf(stderr, "[stream] WARNING: no frames for %d s (see inflight/errors/timeouts above)\n", quiet);
            } else {
                quiet = 0;
            }
            last_frames = f;
            last = now;
        }
        if (A.seconds > 0 && std::chrono::duration<double>(now - t0).count() >= A.seconds) break;
    }

    // ---- shutdown ----
    S.running = false;
    for (auto *t : xfers) libusb_cancel_transfer(t);
    for (int i = 0; i < 100 && S.inflight > 0; ++i) {
        timeval tv{0, 50000};
        libusb_handle_events_timeout(ctx, &tv);
    }
    if (writer.joinable()) {
        { std::lock_guard<std::mutex> lk(S.m); S.stopping = true; }
        S.cv.notify_all();
        writer.join();
    }
    for (auto *t : xfers) libusb_free_transfer(t);
    release_all();
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    Args A;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--frame") A.frame = std::atoi(need());
        else if (a == "--format") A.format = std::atoi(need());
        else if (a == "--fps") A.fps = std::atoi(need());
        else if (a == "--seconds") A.seconds = std::atof(need());
        else if (a == "--unit") A.unit = std::atoi(need());
        else if (a == "--vcif") A.vcif = std::atoi(need());
        else if (a == "--transfers") A.ntransfers = std::atoi(need());
        else if (a == "--leds") A.leds = static_cast<int>(std::strtol(need(), nullptr, 0)) & 0xFFFF;
        else if (a == "--xu") A.xu.push_back(need());
        else if (a == "--sweep") A.sweeps.push_back(need());
        else if (a == "--dwell") A.dwell = std::atof(need());
        else if (a == "--start-delay") A.start_delay = std::atof(need());
        else if (a == "--interleave") A.interleave = true;
        else if (a == "--xu-scan") A.scan = true;
        else if (a == "--uvc-scan") A.uvcscan = true;
        else if (a == "--stdout") A.to_stdout = true;
        else if (a == "--vid") A.vid = static_cast<unsigned>(std::strtoul(need(), nullptr, 0));
        else if (a == "--pid") A.pid = static_cast<unsigned>(std::strtoul(need(), nullptr, 0));
        else { std::fprintf(stderr, "Unknown option %s\n", a.c_str()); return 2; }
    }
    if (A.frame < 1 || A.frame > 8) { std::fprintf(stderr, "--frame must be 1..8\n"); return 2; }
    if (A.ntransfers < 2 || A.ntransfers > 64) A.ntransfers = 8;
    if (A.dwell < 0.3) A.dwell = 0.3;
    if (A.seconds < 0) A.seconds = A.to_stdout ? 0 : (A.sweeps.empty() ? 5 : 0.2);
    std::signal(SIGINT, on_sigint);
    std::signal(SIGPIPE, SIG_IGN);
    if (A.to_stdout) std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);

    libusb_context *ctx = nullptr;
    if (libusb_init(&ctx) != 0) return 1;
    libusb_device_handle *h = nullptr;
    for (int attempt = 0; attempt < 20 && !h && !g_stop; ++attempt) {  // device re-enumerates after each run
        h = libusb_open_device_with_vid_pid(ctx, A.vid, A.pid);
        if (!h) {
            if (attempt == 0) std::fprintf(stderr, "Waiting for %04x:%04x (up to 10 s)...\n", A.vid, A.pid);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    if (!h) {
        std::fprintf(stderr, "Could not open %04x:%04x. Use sudo, quit other camera apps, and if it is not listed by "
                             "`eyetribe_uvc list-usb`, unplug and replug the tracker.\n", A.vid, A.pid);
        libusb_exit(ctx);
        return 1;
    }
    int ad = libusb_set_auto_detach_kernel_driver(h, 1);
    if (ad != 0) std::fprintf(stderr, "auto-detach unavailable: %s\n", libusb_error_name(ad));

    int rc = run(A, ctx, h);
    libusb_close(h);
    libusb_exit(ctx);
    return rc;
}
