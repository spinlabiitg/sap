"""
Audio examples, figures and measurements for ../codecs.html (part 4: speech coding).

Every coder here is a small, readable teaching implementation written for this page, not a standards-compliant codec:
  PCM (uniform), mu-law PCM, backward-adaptive ADPCM and a quantised LPC vocoder.
Run from this folder:   python make_assets.py
Outputs:  audio/*.wav   fig/*.png   speech16k.js (the sentence for the page's live demo)   numbers.json
Speech: ../../data/real_sample_speech.wav (an Indian-English speaker, recorded at 48 kHz / 32 bit),
resampled to 16 kHz / 16 bit as this page's "original".
"""
import base64
import json
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.linalg import toeplitz

HERE = os.path.dirname(os.path.abspath(__file__))

FIG, AUD = os.path.join(HERE, "fig"), os.path.join(HERE, "audio")
os.makedirs(FIG, exist_ok=True)
os.makedirs(AUD, exist_ok=True)
rng = np.random.default_rng(2024)
NUM = {}

INK, BLUE, RED, GOLD, SLATE, TEAL = "#10233F", "#14508C", "#C1272D", "#C9971B", "#5B7083", "#1D7A8C"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#AEB9CB", "axes.labelcolor": INK, "xtick.color": "#5C6B80", "ytick.color": "#5C6B80",
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.color": "#E7ECF2", "grid.linewidth": 0.8,
    "mathtext.fontset": "cm",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig", name)


def write(name, y, sr):
    """Save at 16 kHz for uniform browser playback (lower-rate signals are band-limited up-sampled)."""
    if sr != 16000:
        y = signal.resample_poly(y, 16000 // sr, 1) if 16000 % sr == 0 else signal.resample_poly(y, 16000, sr)
    y = np.clip(y, -1, 1)
    sf.write(os.path.join(AUD, name), y.astype(np.float32), 16000, subtype="PCM_16")


# ---------------------------------------------------------------- measures
def seg_snr(ref, out, sr, frame_ms=20):
    """Segmental SNR in dB over speech frames (frames more than 40 dB below the loudest are skipped)."""
    n = int(sr * frame_ms / 1000)
    L = min(len(ref), len(out))
    E = [np.sum(ref[i:i + n] ** 2) for i in range(0, L - n, n)]
    emax = max(E)
    vals = []
    for k, i in enumerate(range(0, L - n, n)):
        if E[k] < emax * 1e-4: continue
        err = np.sum((ref[i:i + n] - out[i:i + n]) ** 2) + 1e-12
        vals.append(np.clip(10 * np.log10(E[k] / err), -10, 40))
    return float(np.mean(vals))


# ---------------------------------------------------------------- source
SRC = os.path.join(HERE, "..", "..", "data", "real_sample_speech.wav")
raw, sr_raw = sf.read(SRC)
raw = raw if raw.ndim == 1 else raw.mean(axis=1)
NUM["source_rate"] = int(sr_raw); NUM["source_bits"] = 32
y16 = librosa.resample(raw, orig_sr=sr_raw, target_sr=16000)
e_ = librosa.feature.rms(y=y16, frame_length=400, hop_length=160)[0]
act = np.where(20 * np.log10(e_ / e_.max() + 1e-9) > -35)[0]
y16 = y16[max(0, act[0] * 160 - 1600):min(len(y16), act[-1] * 160 + 2400)]   # trim leading/trailing silence
y16 = y16 - y16.mean()
y16 = y16 / np.abs(y16).max() * 0.9
y8 = signal.resample_poly(y16, 1, 2)


def loudest_voiced(x, sr, win):
    """Start index of the loudest low-ZCR (voiced, vowel-like) window."""
    hop = win // 2; best, bi = -1, 0
    for i in range(0, len(x) - win, hop):
        fr = x[i:i + win]
        z = np.mean(np.abs(np.diff(np.signbit(fr).astype(int))))
        e = np.mean(fr ** 2)
        if z < 0.12 and e > best: best, bi = e, i
    return bi


def fricative_centre(x, sr):
    """Centre (s) of the strongest high-ZCR, reasonably loud 30 ms window (a fricative such as /s/)."""
    win = int(0.03 * sr); hop = win // 2; best, bc = -1, 0; emax = np.max([np.mean(x[i:i + win] ** 2) for i in range(0, len(x) - win, hop)])
    for i in range(0, len(x) - win, hop):
        fr = x[i:i + win]; e = np.mean(fr ** 2)
        z = np.mean(np.abs(np.diff(np.signbit(fr).astype(int))))
        if e > emax * 1e-2 and z * e > best: best, bc = z * e, i + win // 2
    return bc / sr


NUM["seconds"] = len(y16) / 16000

# ---------------------------------------------------------------- PCM building blocks
def quant_uniform(x, bits):
    L = 2 ** (bits - 1)
    return np.clip((np.floor(x * L) + 0.5) / L, -1, 1)


MU = 255.0
def mu_compress(x): return np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
def mu_expand(y): return np.sign(y) * ((1 + MU) ** np.abs(y) - 1) / MU
def quant_mulaw(x, bits): return mu_expand(quant_uniform(mu_compress(x), bits))


def band_limit(x, sr_from, sr_to):
    """Resample down (with the usual anti-alias low-pass) and back up, so the result is playable at sr_from."""
    down = signal.resample_poly(x, sr_to, sr_from)
    return signal.resample_poly(down, sr_from, sr_to)[:len(x)], down


rows = []                      # (key, label, kbps, segSNR or None)
def add(key, label, y, sr, kbps, ref=None):
    write(f"{key}.wav", y, sr)
    s = seg_snr(ref, y, sr) if ref is not None else None
    rows.append({"key": key, "label": label, "kbps": kbps, "segsnr": s})


add("orig_16k_16b", "original: 16 kHz, 16 bits", y16, 16000, 256, None)
for bits in (8, 6, 4, 2):
    add(f"pcm_16k_{bits}b", f"16 kHz, {bits} bits", quant_uniform(y16, bits), 16000, 16 * bits, y16)
for sr in (8000, 4000, 2000):
    full, down = band_limit(y16, 16000, sr)
    add(f"pcm_{sr // 1000}k_16b", f"{sr // 1000} kHz, 16 bits", full, 16000, sr * 16 // 1000, None)
add("pcm_8k_8b", "8 kHz, 8 bits, uniform", quant_uniform(y8, 8), 8000, 64, y8)
add("mulaw_8k_8b", "8 kHz, 8 bits, μ-law (G.711)", quant_mulaw(y8, 8), 8000, 64, y8)
add("pcm_8k_4b", "8 kHz, 4 bits, uniform", quant_uniform(y8, 4), 8000, 32, y8)
add("mulaw_8k_4b", "8 kHz, 4 bits, μ-law", quant_mulaw(y8, 4), 8000, 32, y8)

# ---------------------------------------------------------------- LPC helpers
def levinson(r, p):
    a = np.zeros(p + 1); a[0] = 1.0; e = r[0]; ks = np.zeros(p)
    for i in range(1, p + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / e
        a[1:i + 1] = a[1:i + 1] + k * np.concatenate([a[i - 1:0:-1], [1.0]])
        ks[i - 1] = k; e *= 1 - k * k
    return a, ks, e                         # A(z) = 1 + a1 z^-1 + ... (sign convention: A(z) coefficients)


def lpc_frame(x, p):
    w = x * np.hamming(len(x))
    r = np.array([np.dot(w[:len(w) - k], w[k:]) for k in range(p + 1)])
    r[0] = r[0] * 1.0001 + 1e-9                     # tiny white-noise correction keeps the filter stable
    r[1:] *= np.exp(-0.5 * (2 * np.pi * 60 / 8000 * np.arange(1, p + 1)) ** 2)   # lag window (60 Hz bandwidth expansion)
    return levinson(r, p)


def k_to_a(ks):
    a = np.array([1.0])
    for k in ks:
        a = np.concatenate([a, [0.0]]) + k * np.concatenate([[0.0], a[::-1]])
    return a


# log-area-ratio quantisation of reflection coefficients (as in LPC-10 / GSM)
LAR_BITS = [6, 6, 5, 5, 5, 4, 4, 4, 3, 3]          # 45 bits for p = 10
LAR_RANGE = [4.0, 4.0, 2.5, 2.5, 2.5, 2.0, 2.0, 2.0, 1.5, 1.5]
def quant_k(ks):
    out = []
    for k, b, R in zip(ks, LAR_BITS, LAR_RANGE):
        k = np.clip(k, -0.9999, 0.9999)
        g = np.log((1 + k) / (1 - k))
        L = 2 ** b; step = 2 * R / L
        gq = (np.clip(np.floor((g + R) / step), 0, L - 1) + 0.5) * step - R
        out.append(np.tanh(gq / 2))                 # inverse LAR
    return np.array(out)


# ---------------------------------------------------------------- ADPCM (backward adaptive: no side information)
JAYANT = {2: [0.8, 1.6], 3: [0.9, 0.9, 1.25, 1.75], 4: [0.9, 0.9, 0.9, 0.9, 1.2, 1.6, 2.0, 2.4]}
def adpcm(x, bits, p=10, use_prediction=True):
    L = 2 ** (bits - 1); M = JAYANT[bits]
    d = np.zeros(len(x)); a = np.zeros(p + 1); a[0] = 1
    step = 0.01; smin, smax = 1e-4, 0.5
    for n in range(len(x)):
        if use_prediction and n % 80 == 0 and n >= 256:          # every 10 ms, refit from the past *decoded* signal
            a, _, _ = lpc_frame(d[n - 256:n], p)
        pred = -np.dot(a[1:], d[n - 1:n - p - 1:-1]) if (use_prediction and n > p) else 0.0
        e = x[n] - pred
        idx = min(L - 1, int(abs(e) / step))
        eq = np.sign(e or 1.0) * (idx + 0.5) * step
        d[n] = pred + eq
        step = min(smax, max(smin, step * M[idx]))
    return d


for bits in (4, 3, 2):
    add(f"adpcm_8k_{bits}b", f"ADPCM, 8 kHz, {bits} bits", adpcm(y8, bits), 8000, 8 * bits, y8)
add("apcm_8k_4b", "adaptive step, no prediction, 4 bits", adpcm(y8, 4, use_prediction=False), 8000, 32, y8)
add("apcm_8k_2b", "adaptive step, no prediction, 2 bits", adpcm(y8, 2, use_prediction=False), 8000, 16, y8)

# prediction gain on the 8 kHz sentence (frame-wise LPC, p = 10)
pe_e, pe_x = 0.0, 0.0
for i in range(0, len(y8) - 240, 160):
    a, _, _ = lpc_frame(y8[i:i + 240], 10)
    seg = y8[i + 40:i + 200]
    res = signal.lfilter(a, [1], y8[i:i + 200])[40:200]
    pe_e += np.sum(res ** 2); pe_x += np.sum(seg ** 2)
NUM["prediction_gain_db_8k_p10"] = float(10 * np.log10(pe_x / pe_e))

# ---------------------------------------------------------------- LPC vocoder (parametric), 8 kHz, 20 ms frames
FR, WIN, P = 160, 240, 10
pre8 = signal.lfilter([1, -0.9375], [1], y8)
nfr = len(pre8) // FR
def classic_pitch(x, sr=8000, hop=FR, win=320):
    """LPC-10-style voicing/pitch: autocorrelation of a low-passed, centre-clipped 40 ms frame (60–400 Hz)."""
    lp = signal.filtfilt(*signal.butter(4, 900 / (sr / 2)), x)
    E = np.array([np.mean(x[max(0, i * hop + hop // 2 - win // 2):i * hop + hop // 2 + win // 2] ** 2) for i in range(len(x) // hop)])
    emax = E.max()
    f0s, vs = [], []
    for i in range(len(x) // hop):
        c = i * hop + hop // 2
        fr = lp[max(0, c - win // 2):c + win // 2]
        if len(fr) < win or E[i] < emax * 10 ** (-3.5):
            f0s.append(0.0); vs.append(False); continue
        clip = 0.3 * np.abs(fr).max()
        fc = np.where(fr > clip, fr - clip, np.where(fr < -clip, fr + clip, 0.0))
        r = np.correlate(fc, fc, "full")[win - 1:]
        lo, hi = int(sr / 400), int(sr / 60)
        if r[0] <= 0:
            f0s.append(0.0); vs.append(False); continue
        k = lo + int(np.argmax(r[lo:hi])); peak = r[k] / r[0]
        zcr = np.mean(np.abs(np.diff(np.signbit(x[c - win // 2:c + win // 2]).astype(int))))
        voiced = peak > 0.3 and zcr < 0.25
        f0s.append(sr / k if voiced else 0.0); vs.append(voiced)
    f0s = np.array(f0s); vs = np.array(vs)
    f0s[vs] = signal.medfilt(np.where(vs, f0s, 0), 3)[vs]          # light smoothing of the pitch track
    vs &= f0s > 0
    return f0s, vs


f0, vflag = classic_pitch(y8)
NUM["vocoder_voiced_fraction"] = float(np.mean(vflag))
f0 = np.nan_to_num(f0)
frames = []
for i in range(nfr):
    c = i * FR + FR // 2
    seg = pre8[max(0, c - WIN // 2):c + WIN // 2]
    if len(seg) < WIN: seg = np.pad(seg, (0, WIN - len(seg)))
    a, ks, e = lpc_frame(seg, P)
    res = signal.lfilter(a, [1], pre8[i * FR:(i + 1) * FR])
    g = np.sqrt(np.mean(res ** 2) + 1e-12)
    frames.append(dict(ks=ks, gain=g, f0=f0[i] if i < len(f0) else 0, voiced=bool(vflag[i]) if i < len(vflag) else False))

PITCH_BITS, GAIN_BITS = 6, 5
def q_pitch(f):
    lo, hi = np.log(60), np.log(400); L = 2 ** PITCH_BITS
    return float(np.exp(lo + (np.clip(np.floor((np.log(np.clip(f, 60, 400)) - lo) / (hi - lo) * L), 0, L - 1) + 0.5) / L * (hi - lo)))
def q_gain(g):
    db = 20 * np.log10(g + 1e-9); lo, hi = -80, -10; L = 2 ** GAIN_BITS
    return float(10 ** (((np.clip(np.floor((db - lo) / (hi - lo) * L), 0, L - 1) + 0.5) / L * (hi - lo) + lo) / 20))
VOC_BITS = sum(LAR_BITS) + PITCH_BITS + GAIN_BITS + 1
NUM["vocoder_bits_per_frame"] = VOC_BITS
NUM["vocoder_bps"] = VOC_BITS * 50


def vocoder(quantise, reset_memory=False, reset_phase=False):
    out = np.zeros(nfr * FR); hist = np.zeros(P); ph = 0.0
    for i, fr in enumerate(frames):
        if reset_memory: hist = np.zeros(P)        # forget the tract's ringing at every frame boundary
        if reset_phase: ph = 1.0                   # restart the pulse train: a pulse at every frame start
        ks = quant_k(fr["ks"]) if quantise else fr["ks"]
        a = k_to_a(ks)
        g = q_gain(fr["gain"]) if quantise else fr["gain"]
        if fr["voiced"] and fr["f0"] > 0:
            F = q_pitch(fr["f0"]) if quantise else fr["f0"]
            ex = np.zeros(FR)
            for n in range(FR):
                ph += F / 8000
                if ph >= 1: ph -= 1; ex[n] = 1.0
        else:
            ex = rng.standard_normal(FR)
        ex *= g / (np.sqrt(np.mean(ex ** 2)) + 1e-12)
        for n in range(FR):
            v = ex[n] - np.dot(a[1:], hist)
            hist = np.concatenate([[v], hist[:-1]])
            out[i * FR + n] = v
    out = signal.lfilter([1], [1, -0.9375], out)
    out = out * np.sqrt(np.mean(y8[:len(out)] ** 2) / np.mean(out ** 2))     # match the original loudness
    return out * min(1.0, 0.95 / np.abs(out).max())                          # and never clip


add("vocoder_unquantised", "LPC vocoder, parameters unquantised", vocoder(False), 8000, None, None)
add("vocoder_q", f"LPC vocoder, {VOC_BITS} bits / 20 ms", vocoder(True), 8000, VOC_BITS * 50 / 1000, None)

# (a toy CELP coder used to live here; it was removed with the CELP section. Its random codebook draw is kept
#  so the random stream, and hence the frame-memory audio generated later, is unchanged.)
rng.standard_normal((512, 40))

NUM["rows"] = rows
for r in rows: print(f"  {r['key']:22s} {str(r['kbps']):>6} kbps  segSNR {r['segsnr']}")

# ---------------------------------------------------------------- the sentence for the live demo (16 kHz, int16, base64)
pcm = np.round(y16 * 32767).astype("<i2").tobytes()
with open(os.path.join(HERE, "speech16k.js"), "w", encoding="utf8") as fh:
    fh.write("// real_sample_speech.wav (Indian-English speaker), resampled to 16 kHz, int16 little-endian, base64.\n")
    fh.write(f"window.CODEC_SPEECH = {{ sr: 16000, b64: \"{base64.b64encode(pcm).decode()}\" }};\n")
print("  wrote speech16k.js")

# ================================================================ figures
# Fig 1: quantisation staircases (top) and the error they leave (bottom)
t0 = loudest_voiced(y16, 16000, 160); seg = y16[t0:t0 + 160]          # 10 ms of the loudest vowel
tt = np.arange(160) / 16
fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True, gridspec_kw=dict(wspace=0.22, hspace=0.3, height_ratios=[1.5, 1]))
for j, b in enumerate((8, 4, 2)):
    ax, ae = axs[0, j], axs[1, j]
    q = quant_uniform(seg, b); L = 2 ** (b - 1); step = 1 / L
    if b <= 4:
        for lev in (np.arange(-L, L) + 0.5) / L: ax.axhline(lev, color="#E7ECF2", lw=0.8)
    ax.plot(tt, seg, color="#AEB9CB", lw=2.2, label="original")
    ax.step(tt, q, where="mid", color=RED, lw=1.3, label=f"{b}-bit")
    ax.set_title(rf"{b} bits: $2^{{{b}}} = {2 ** b}$ levels, step $\Delta = {step:.4g}$", loc="left", fontsize=10)
    if b == 8:   # zoom in: at 8 bits the steps are too small to see at full scale
        ins = ax.inset_axes([0.52, 0.55, 0.44, 0.4])
        zs = slice(60, 72)
        ins.plot(tt[zs], seg[zs], color="#AEB9CB", lw=2.5); ins.step(tt[zs], q[zs], where="mid", color=RED, lw=1.4)
        for lev in (np.arange(-L, L) + 0.5) / L:
            if seg[zs].min() - 2 * step < lev < seg[zs].max() + 2 * step: ins.axhline(lev, color="#D7DEE8", lw=0.6)
        ins.set_xticks([]); ins.set_yticks([]); ins.set_title(r"zoomed $\times 20$",fontsize=8, pad=2)
        ax.indicate_inset_zoom(ins, edgecolor=INK, alpha=0.6)
    err = q - seg
    ae.fill_between(tt, -step / 2, step / 2, color=RED, alpha=0.08, lw=0)
    ae.plot(tt, err, color=RED, lw=1)
    ae.set_ylim(-step * 0.75, step * 0.75)
    ae.set_title(rf"error: always within $\pm\Delta/2 = \pm{step / 2:.3g}$", loc="left", fontsize=9)
    ae.set_xlabel("ms")
axs[0, 0].legend(loc="lower left"); axs[0, 0].set_ylabel("amplitude"); axs[1, 0].set_ylabel("error")
save(fig, "01_quantisation.png")

# Fig 1b: is the quantisation error really uniform? (the assumption behind SNR = 6.02 b + 1.76 dB)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True, gridspec_kw=dict(wspace=0.08))
NUM["error_uniformity"] = {}
for ax, b in zip(axs, [8, 4, 2]):
    step = 2.0 / 2 ** b
    err = (quant_uniform(y16, b) - y16) / step                     # sent minus original, in units of the step
    ax.hist(err, bins=40, range=(-0.5, 0.5), density=True, color={8: BLUE, 4: GOLD, 2: RED}[b], alpha=0.75)
    ax.plot([-0.5, -0.5, 0.5, 0.5], [0, 1, 1, 0], color=INK, lw=1.4, ls="--", label="uniform: density 1")
    ratio = float(np.mean(err ** 2))
    snr_meas = float(10 * np.log10(np.sum(y16 ** 2) / np.sum(((quant_uniform(y16, b) - y16)) ** 2)))
    NUM["error_uniformity"][b] = {"var_over_step2": ratio, "snr_db": snr_meas}
    ax.set_title(rf"$b = {b}$:  $\sigma_q^2/\Delta^2 = {ratio:.4f}$" + {8: "", 4: "  (strained)", 2: "  (breaks down)"}[b], fontsize=10, loc="left")
    ax.set_xlabel(r"error $q$ in units of the step, $q/\Delta$")
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 3.2)
axs[0].set_ylabel("probability density")
axs[0].legend(loc="upper left")
fig.suptitle(r"Quantisation error of the whole recording.  Uniform on $[-\Delta/2,\ \Delta/2]$ predicts $\sigma_q^2/\Delta^2 = 1/12 = 0.0833$",
             x=0.08, ha="left", y=1.04, fontsize=10.5, weight="bold", color=INK)
save(fig, "01b_error_histogram.png")

# Fig 2b: why mu-law: histogram + levels, compression curve, SNR vs loudness
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(wspace=0.3, width_ratios=[1.25, 1, 1.15]))
ax = axs[0]
vals = y8[np.abs(y8) > 1e-4]
ax.hist(vals, bins=np.linspace(-1, 1, 161), color=GOLD, alpha=0.75, density=True)
lev_u = (np.arange(-8, 8) + 0.5) / 8
lev_m = mu_expand((np.arange(-8, 8) + 0.5) / 8)
ymax = ax.get_ylim()[1]
for v in lev_u: ax.plot([v, v], [-0.10 * ymax, -0.02 * ymax], color=SLATE, lw=1.6)
for v in lev_m: ax.plot([v, v], [-0.22 * ymax, -0.14 * ymax], color=BLUE, lw=1.6)
ax.set_ylim(-0.25 * ymax, ymax); ax.set_xlim(-0.6, 0.6)
ax.text(-0.58, 0.92 * ymax, "levels (ticks below):", ha="left", fontsize=8, color=INK)
ax.text(-0.58, 0.84 * ymax, "grey: 4-bit uniform", ha="left", fontsize=8, color=SLATE)
ax.text(-0.58, 0.76 * ymax, r"blue: 4-bit $\mu$-law", ha="left", fontsize=8, color=BLUE)
share = float(np.mean(np.abs(y8) < 0.1))
ax.set_title(f"(a) most samples are small:\n{100 * share:.0f} % lie within " + r"$\pm 0.1$", loc="left", fontsize=10)
ax.set_xlabel("sample value"); ax.set_yticks([])
NUM["share_within_0p1"] = share
ax = axs[1]
xx = np.linspace(0, 1, 400)
ax.plot(xx, mu_compress(xx), color=BLUE, lw=2.2, label=r"$\mu$-law, $\mu = 255$")
ax.plot(xx, xx, color=SLATE, lw=1.4, ls="--", label="uniform (no compression)")
for k in range(8):
    yv = (k + 0.5) / 8; xv = mu_expand(yv)
    ax.plot([0, xv], [yv, yv], color=BLUE, lw=0.6, alpha=0.5); ax.plot([xv, xv], [0, yv], color=BLUE, lw=0.6, alpha=0.5)
    ax.plot([xv], [0], "|", color=BLUE, ms=10)
ax.set_xlabel(r"input $|x|$"); ax.set_ylabel(r"compressed $F(x)$"); ax.legend(loc="lower right")
ax.set_title("(b) compress, then use equal steps:\nfine steps near zero, coarse at the top", loc="left", fontsize=10)
ax = axs[2]
n = 160; lv, su8, sm8 = [], [], []
emax = max(np.sum(y8[i:i + n] ** 2) for i in range(0, len(y8) - n, n))
for i in range(0, len(y8) - n, n):
    fr = y8[i:i + n]; e = np.sum(fr ** 2)
    if e < emax * 1e-5: continue
    lv.append(10 * np.log10(e / emax))
    su8.append(10 * np.log10(e / np.sum((quant_uniform(fr, 8) - fr) ** 2)))
    sm8.append(10 * np.log10(e / np.sum((quant_mulaw(fr, 8) - fr) ** 2)))
ax.scatter(lv, su8, s=10, color=SLATE, alpha=0.7, label="8-bit uniform")
ax.scatter(lv, sm8, s=10, color=BLUE, alpha=0.7, label=r"8-bit $\mu$-law")
ax.set_xlabel("frame loudness (dB below the loudest frame)"); ax.set_ylabel("SNR of that frame (dB)")
ax.legend(loc="lower right")
ax.set_title("(c) SNR of each 20 ms frame:\nquiet frames suffer only with uniform", loc="left", fontsize=10)
save(fig, "02b_mulaw_explained.png")

# Fig 2c: why mu = 255? SNR against input level for several mu (8 bits, speech-like Laplacian signal)
def mu_q(x, bits, mu):
    if mu == 0: return quant_uniform(x, bits)
    c = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    cq = quant_uniform(c, bits)
    return np.sign(cq) * ((1 + mu) ** np.abs(cq) - 1) / mu
lap = np.random.default_rng(7).laplace(size=200000); lap /= np.sqrt(np.mean(lap ** 2))
levels_db = np.arange(-70, 1, 2)
fig, ax = plt.subplots(figsize=(8.6, 4.2))
NUM["mu_tradeoff"] = {}
for mu, col, lw in [(0, SLATE, 1.6), (15, TEAL, 1.4), (100, GOLD, 1.8), (255, BLUE, 2.8), (1000, RED, 1.6)]:
    snr = []
    for lev in levels_db:
        xx = np.clip(lap * 10 ** (lev / 20), -1, 1)
        snr.append(10 * np.log10(np.mean(xx ** 2) / np.mean((mu_q(xx, 8, mu) - xx) ** 2)))
    snr = np.array(snr)
    ok = levels_db[snr >= 30]
    rng_db = float(ok.max() - ok.min()) if len(ok) else 0.0
    NUM["mu_tradeoff"][mu] = {"peak": float(snr.max()), "range_above_30dB": rng_db}
    lab = (r"uniform ($\mu = 0$)" if mu == 0 else rf"$\mu = {mu}$") + rf": peak {snr.max():.0f} dB, $\geq 30$ dB over {rng_db:.0f} dB of levels"
    ax.plot(levels_db, snr, color=col, lw=lw, label=lab)
ax.axhline(30, color=INK, ls=":", lw=1)
ax.text(-69, 31, "30 dB", fontsize=8.5, color=INK)
ax.set_xlabel("input level (dB relative to full scale): soft talkers " + r"$\leftarrow\;\rightarrow$" + " loud talkers")
ax.set_ylabel("SNR (dB)"); ax.set_ylim(-5, 48); ax.legend(loc="lower right", fontsize=8.2)
ax.set_title(r"8-bit companding: bigger $\mu$ = flatter SNR over a wider range, but a lower peak", loc="left", fontsize=10)
save(fig, "02c_mu_tradeoff.png")

# the sample-value histogram for the page's μ-law explorer
hist_counts, _ = np.histogram(y16, bins=np.linspace(-1, 1, 201))
NUM["hist16_counts"] = hist_counts.tolist()

# Fig 2: SNR vs bits, uniform vs mu-law (8 kHz)
bits_range = np.arange(2, 13)
su = [seg_snr(y8, quant_uniform(y8, b), 8000) for b in bits_range]
sm = [seg_snr(y8, quant_mulaw(y8, b), 8000) for b in bits_range]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(bits_range, su, "o-", color=SLATE, label="uniform PCM")
ax.plot(bits_range, sm, "o-", color=BLUE, label=r"$\mu$-law PCM")
ax.plot(bits_range, 6.02 * bits_range - 10, ":", color=GOLD, label=r"slope: $6.02\,b - 10$ dB ($+6$ dB per extra bit)")
ax.axvline(8, color=RED, ls=":", lw=1); ax.text(8.15, 2, "G.711\n(8 bits)", color=RED, fontsize=8.5)
ax.set_xlabel("bits per sample"); ax.set_ylabel("segmental SNR (dB)"); ax.legend(loc="upper left")
ax.set_title(r"Each extra bit buys about 6 dB; $\mu$-law adds about 10 dB at any bit depth", loc="left", fontsize=10)
save(fig, "02_snr_vs_bits.png")
NUM["snr_uniform"] = dict(zip(bits_range.tolist(), su)); NUM["snr_mulaw"] = dict(zip(bits_range.tolist(), sm))

# Fig 3: spectrograms at different sampling rates
fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, (sr, lab) in zip(axs, [(16000, "16 kHz"), (8000, "8 kHz (telephone)"), (4000, "4 kHz"), (2000, "2 kHz")]):
    z = y16 if sr == 16000 else band_limit(y16, 16000, sr)[0]
    fc = fricative_centre(y16, 16000); w0 = max(0, int((fc - 0.45) * 16000))
    S = np.abs(librosa.stft(z[w0:w0 + int(0.9 * 16000)], n_fft=512, hop_length=64))
    ax.pcolormesh(np.arange(S.shape[1]) * 64 / 16000, np.linspace(0, 8000, S.shape[0]), 20 * np.log10(S / S.max() + 1e-6),
                  cmap="magma", vmin=-70, vmax=0, shading="auto", rasterized=True)
    ax.axhline(sr / 2, color="white", ls="--", lw=1)
    ax.set_title(f"{lab}: up to {sr // 2000 if sr >= 2000 else 0}" + (" kHz" if sr >= 2000 else ""), fontsize=9.5)
    ax.set_xlabel("s"); ax.grid(False)
axs[0].set_ylabel("Hz")
save(fig, "03_sampling_rates.png")

# Fig 4: why prediction helps: the signal vs the prediction error it leaves (8 kHz, same scale)
v0 = loudest_voiced(y8, 8000, 960)                                     # 120 ms around a loud vowel
a, _, _ = lpc_frame(y8[v0 + 360:v0 + 600], 10)
z = y8[v0:v0 + 960]
res = signal.lfilter(a, [1], z)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.2))
tt = np.arange(len(z)) / 8
axs[0].plot(tt, z, color=SLATE, lw=1, label=r"speech $x[n]$")
axs[0].plot(tt, res, color=RED, lw=1, label=r"prediction error $e[n]$ (what ADPCM codes)")
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2); axs[0].set_xlabel("ms"); axs[0].set_title("Same scale: the error is much smaller than the signal", loc="left", fontsize=10)
axs[1].hist(z, bins=60, color=SLATE, alpha=0.6, label=r"$x[n]$", density=True)
axs[1].hist(res, bins=60, color=RED, alpha=0.6, label=r"$e[n]$", density=True)
axs[1].legend(); axs[1].set_title("so it needs fewer levels", loc="left", fontsize=10); axs[1].set_yticks([])
save(fig, "04_prediction_error.png")

with open(os.path.join(HERE, "numbers.json"), "w", encoding="utf8") as fh:
    json.dump(NUM, fh, indent=1, ensure_ascii=False, default=float)
print("done")
