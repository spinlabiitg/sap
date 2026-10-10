"""
Figures, audio, embedded demo data and quoted numbers for ../spectrogram.html (module 1: seeing sound).

Run from this folder:   python make_figures.py      (then python build_page.py)
Outputs:  fig/*.png   audio/*.wav   sentence.js   numbers.json

Speech: CMU ARCTIC speaker ksp (Indian-English male, 16 kHz), read through ../common/arctic.py. Phone labels and
pitch marks are the corpus's automatic ones (not hand-corrected); every token used in a figure is checked
acoustically below and the measurements are written to numbers.json. Signals called "synthetic" are generated here.
"""
import base64
import json
import os
import sys

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from PIL import Image

matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import arctic  # noqa: E402

FIG = os.path.join(HERE, "fig")
AUD = os.path.join(HERE, "audio")
os.makedirs(FIG, exist_ok=True)
os.makedirs(AUD, exist_ok=True)
NUM = {}
rng = np.random.default_rng(11)

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
CMAP = "magma"


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # spectrogram images are large as full-colour PNGs; a 256-colour palette is visually identical and ~3x smaller
    Image.open(p).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(p, optimize=True)
    print("  fig", name, os.path.getsize(p) // 1024, "KB")


def wav(name, y, sr=16000):
    y = np.asarray(y, dtype=np.float64)
    pk = np.max(np.abs(y))
    if pk > 0.99:
        y = 0.95 * y / pk
    sf.write(os.path.join(AUD, name), y, sr, subtype="PCM_16")
    print("  wav", name)


SR = arctic.SR


# ================================================================ STFT tools (plain numpy, nothing hidden)
def window(kind, L, sym=False):
    n = np.arange(L)
    D = (L - 1) if sym else L                  # symmetric: ends both at 0; periodic: one period of a cosine
    if kind == "rect":
        return np.ones(L)
    if kind == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / D)
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / D)
    raise ValueError(kind)


def stft(x, L, H, N=None, kind="hann", w=None, pad=None):
    """Frames x bins. Frame m covers x_padded[mH : mH+L]. By default the signal is padded by L/2, so frame m is
    centred at time mH (for plotting); the inversion tests pad by L so every sample is covered by all overlapping frames."""
    N = N or L
    w = window(kind, L) if w is None else w
    pad = L // 2 if pad is None else pad
    xp = np.pad(x, (pad, pad + H))
    nfr = 1 + (len(xp) - L) // H
    idx = np.arange(L)[None, :] + H * np.arange(nfr)[:, None]
    return np.fft.rfft(xp[idx] * w, N, axis=1)


def istft(X, L, H, length, w_an, w_syn, normalise=True, pad=None):
    """Weighted overlap-add. With normalise=True divide by sum_m w_an*w_syn (shifted), which works for any window."""
    pad = L // 2 if pad is None else pad
    fr = np.fft.irfft(X, axis=1)[:, :L] * w_syn
    nfr = fr.shape[0]
    out = np.zeros(L + H * (nfr - 1))
    wsum = np.zeros_like(out)
    for m in range(nfr):
        out[m * H:m * H + L] += fr[m]
        wsum[m * H:m * H + L] += w_an * w_syn
    if normalise:
        out = out / np.maximum(wsum, 1e-12)
    return out[pad:pad + length], wsum[pad:pad + length]


def spec_db(x, L, H, N, kind="hann", floor=80):
    S = np.abs(stft(x, L, H, N, kind)).T
    D = 20 * np.log10(S / S.max() + 1e-12)
    return np.maximum(D, -floor), S


def show_spec(ax, D, dur, fmax=8000, vmin=-80, cmap=CMAP, t0=0.0):
    im = ax.imshow(D, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=0,
                   extent=[t0, t0 + dur, 0, SR / 2 / 1000], interpolation="bilinear")
    ax.set_ylim(0, fmax / 1000)
    ax.grid(False)
    return im


def formants(y, order=14):
    """LPC formant estimates (Hz) of one short frame: roots with bandwidth < 500 Hz, above 150 Hz."""
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]) * np.hamming(len(y))
    a = librosa.lpc(y, order=order)
    r = np.roots(a)
    r = r[np.imag(r) > 0]
    fr = np.angle(r) * SR / (2 * np.pi)
    bw = -np.log(np.abs(r)) * SR / np.pi
    return sorted(f for f, b in zip(fr, bw) if f > 150 and b < 500)


# ================================================================ the demo sentence
UTT = "arctic_a0374"
x = arctic.read_wav(UTT)
x = x - np.mean(x)
x = 0.9 * x / np.max(np.abs(x))
DUR = len(x) / SR
TEXT = arctic.text(UTT)
NUM["dur_s"] = round(DUR, 2)
NUM["n_samples"] = len(x)

CLASS = {}
for p in ["iy", "ih", "eh", "ae", "aa", "ah", "ao", "uh", "uw", "ax", "er"]:
    CLASS[p] = "vowel"
for p in ["ay", "aw", "oy", "ey", "ow"]:
    CLASS[p] = "diphthong"
for p in arctic.FRICATIVES:
    CLASS[p] = "fricative"
for p in arctic.STOPS:
    CLASS[p] = "stop"
for p in ["ch", "jh"]:
    CLASS[p] = "affricate"
for p in arctic.NASALS:
    CLASS[p] = "nasal"
for p in ["w", "y", "r", "l"]:
    CLASS[p] = "approximant"
CLASS["pau"] = CLASS["ssil"] = "pause"

# automatic labels, with the corpus's 10-ms pause inside "cash, she" and the leading double pause merged
raw = arctic.phones(UTT)
ph = []
for s, e, p in raw:
    if ph and p in ("pau", "ssil") and ph[-1][2] in ("pau", "ssil"):
        ph[-1] = (ph[-1][0], e, "pau")
        continue
    if p == "pau" and e - s < 0.02 and ph:
        ph[-1] = (ph[-1][0], e, ph[-1][2])          # absorb into the previous phone
        continue
    ph.append((s, e, "pau" if p == "ssil" else p))
seq = [p for _, _, p in ph]
expect = ("pau b ah t l ay f s pau w er th pau m ao r dh ae n pau k ae sh sh iy aa r g y uw d pau").split()
assert seq == expect, seq
WORDS = [("But", 1, 3), ("life's", 4, 7), ("worth", 9, 11), ("more", 13, 15), ("than", 16, 18),
         ("cash,", 20, 22), ("she", 23, 24), ("argued.", 25, 30)]
word_of = {}
for w, a, b in WORDS:
    for i in range(a, b + 1):
        word_of[i] = w.strip(",.")
ph = [(s, min(e, DUR), p) for s, e, p in ph]
TOK = {"s": 7, "sh": 23, "iy": 24, "ae_cash": 21, "k": 20, "ay": 5, "m": 13, "aa": 25}
for k, i in TOK.items():
    assert ph[i][2] == k.split("_")[0], (k, ph[i])

# ---- F0 from the corpus pitch marks: intervals between marks that fall inside vowels
pm = arctic.pitchmarks(UTT)
T0s = np.array([b - a for a, b in zip(pm[:-1], pm[1:])
                if any(s <= a and b <= e and CLASS[p] in ("vowel", "diphthong") for s, e, p in ph)])
T0 = float(np.median(T0s))
NUM["T0_ms"] = round(T0 * 1000, 1)
NUM["F0_hz"] = round(1 / T0)
NUM["F0_p10"] = round(float(np.percentile(1 / T0s, 10)))
NUM["F0_p90"] = round(float(np.percentile(1 / T0s, 90)))
NUM["n_periods"] = len(T0s)
vmarks = np.array([a for a in pm if any(s <= a <= e and CLASS[p] in ("vowel", "diphthong", "nasal", "approximant") for s, e, p in ph)])

# ---- how fast speech changes: phone durations over the whole ksp corpus
durs = np.array([e - s for u in arctic.utterances() for s, e, p in arctic.phones(u) if p not in ("pau", "ssil")])
NUM["corpus_phones"] = len(durs)
NUM["corpus_utts"] = len(arctic.utterances())
NUM["phone_med_ms"] = round(float(np.median(durs)) * 1000)
NUM["phone_p10_ms"] = round(float(np.percentile(durs, 10)) * 1000)
NUM["phone_p90_ms"] = round(float(np.percentile(durs, 90)) * 1000)
sent_durs = [e - s for s, e, p in ph if p != "pau"]
NUM["sent_phones"] = len(sent_durs)
NUM["sent_phone_med_ms"] = round(float(np.median(sent_durs)) * 1000)

# ================================================================ Fig 1: waveform / whole spectrum / spectrogram
t = np.arange(len(x)) / SR
Xall = np.fft.rfft(x)
fall = np.fft.rfftfreq(len(x), 1 / SR)
NUM["whole_bin_hz"] = round(SR / len(x), 3)
fig = plt.figure(figsize=(10, 7.4))
gs = fig.add_gridspec(3, 1, hspace=0.62, height_ratios=[0.8, 0.9, 1.25])
a0 = fig.add_subplot(gs[0])
a0.plot(t, x, color=BLUE, lw=0.4)
a0.set_xlim(0, DUR); a0.set_ylim(-1, 1.12); a0.set_yticks([])
for w, i, j in WORDS:
    a0.text((ph[i][0] + ph[j][1]) / 2, 1.0, w, ha="center", va="bottom", fontsize=8.5, color=INK)
a0.set_title("1 · The waveform: you see when things happen, but not which frequencies", loc="left", pad=16)
a0.set_xlabel("time (s)")
a1 = fig.add_subplot(gs[1])
Ad = 20 * np.log10(np.abs(Xall) / np.abs(Xall).max() + 1e-12)
a1.plot(fall / 1000, Ad, color=SLATE, lw=0.3)
a1.set_xlim(0, 8); a1.set_ylim(-90, 3)
a1.set_ylabel("dB"); a1.set_xlabel("frequency (kHz)")
a1.set_title(f"2 · One FFT of the whole {DUR:.2f} s: every frequency present, but no idea when", loc="left")
a2 = fig.add_subplot(gs[2])
D, _ = spec_db(x, 400, 32, 1024)
im = show_spec(a2, D, DUR)
a2.set_ylabel("kHz"); a2.set_xlabel("time (s)")
a2.set_title("3 · The spectrogram: 25 ms slices, 2 ms apart, each one's spectrum as a column", loc="left")
for w, i, j in WORDS:
    a2.axvline(ph[i][0], color="white", lw=0.5, alpha=0.5)
cb = fig.colorbar(im, cax=a2.inset_axes([1.01, 0, 0.015, 1])); cb.set_label("dB re max", fontsize=8.5)
save(fig, "01_why_spectrogram.png")

# ================================================================ Fig 2: the STFT step by step
L25, H10 = 400, 160
tc = 1.70                                      # centre of the analysed frame, inside /ae/ of "cash"
assert ph[TOK["ae_cash"]][0] < tc < ph[TOK["ae_cash"]][1]
c0 = int(tc * SR) - L25 // 2
w = window("hann", L25)
fr = x[c0:c0 + L25]
fig = plt.figure(figsize=(10.4, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.25)
a = fig.add_subplot(gs[0, :])
seg_a, seg_b = 1.60, 1.80
tt = np.arange(int(seg_a * SR), int(seg_b * SR))
a.plot(tt / SR, x[tt], color=SLATE, lw=0.6)
for k, col in zip([-1, 0, 1], [GOLD, RED, GOLD]):
    s0 = c0 + k * H10
    a.plot((s0 + np.arange(L25)) / SR, 0.75 * w, color=col, lw=1.6 if k == 0 else 1.0, ls="-" if k == 0 else "--")
a.annotate("", xy=((c0 + H10) / SR, 0.84), xytext=(c0 / SR, 0.84), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
a.text((c0 + H10 / 2) / SR, 0.88, "hop H = 10 ms", ha="center", fontsize=8.5, color=INK)
a.annotate("", xy=((c0 + L25) / SR, -0.86), xytext=(c0 / SR, -0.86), arrowprops=dict(arrowstyle="<->", color=RED, lw=0.9))
a.text((c0 + L25 / 2) / SR, -0.98, "window L = 25 ms (400 samples)", ha="center", fontsize=8.5, color=RED)
a.set_xlim(seg_a, seg_b); a.set_ylim(-1.05, 1.0); a.set_yticks([])
a.set_xlabel("time (s)")
a.set_title("① cut out a frame of length L every H samples  (here: “cash”, Hann windows drawn)", loc="left")
b = fig.add_subplot(gs[1, 0])
tm = np.arange(L25) / SR * 1000
b.plot(tm, fr, color="#AEB9CB", lw=0.9, label="raw frame")
b.plot(tm, fr * w, color=RED, lw=1.0, label="frame × Hann window")
b.set_xlabel("time within the frame (ms)"); b.set_yticks([])
b.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
b.set_title("② multiply by the window", loc="left", pad=22)
c = fig.add_subplot(gs[1, 1])
Xf = np.fft.rfft(fr * w, 1024)
ff = np.fft.rfftfreq(1024, 1 / SR)
c.plot(ff / 1000, 20 * np.log10(np.abs(Xf) / np.abs(Xf).max() + 1e-9), color=BLUE, lw=0.9)
c.set_ylim(-80, 4); c.set_xlim(0, 8)
c.set_xlabel("frequency (kHz)"); c.set_ylabel("dB")
c.set_title(r"③ FFT, keep $|X|$, convert to dB", loc="left", pad=22)
fig.text(0.5, 0.0, "④ repeat for every frame and stand the spectra side by side as columns: that is the spectrogram",
         ha="center", fontsize=9.5, color=INK, weight="bold")
save(fig, "02_stft_steps.png")

# ================================================================ Fig 3: zero-padding interpolates, it does not resolve
f1, f2 = 1000.0, 1040.0
NUM["zp_f1"], NUM["zp_f2"] = int(f1), int(f2)
nn = np.arange(SR)
two = np.sin(2 * np.pi * f1 * nn / SR) + np.sin(2 * np.pi * f2 * nn / SR)
fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True, gridspec_kw={"wspace": 0.12})
cases = [(256, 256, "L = 16 ms, N = 256"), (256, 4096, "L = 16 ms, N = 4096 (zero-padded)"), (1024, 4096, "L = 64 ms, N = 4096")]
for a, (L, N, title) in zip(ax, cases):
    Y = np.abs(np.fft.rfft(two[:L] * window("hann", L), N))
    fy = np.fft.rfftfreq(N, 1 / SR)
    yd = 20 * np.log10(Y / Y.max() + 1e-9)
    if N == L:
        a.plot(fy, yd, "o-", color=BLUE, ms=4, lw=1)
    else:
        a.plot(fy, yd, color=BLUE, lw=1.2)
    for f in (f1, f2):
        a.axvline(f, color=RED, lw=0.8, ls=":")
    a.set_xlim(700, 1360); a.set_ylim(-50, 3)
    a.set_title(title, fontsize=9.5)
    a.set_xlabel("frequency (Hz)")
    a.text(0.03, 0.04, f"bin spacing $f_s/N$ = {SR / N:.1f} Hz\nresolution $\\approx f_s/L$ = {SR / L:.1f} Hz",
           transform=a.transAxes, fontsize=8, color=INK, va="bottom")
ax[0].set_ylabel("dB")
ax[1].text(0.5, 0.5, "smooth, but still\none peak", transform=ax[1].transAxes, ha="center", color=RED, fontsize=8.5)
ax[2].text(0.5, 0.5, "two peaks:\nresolved", transform=ax[2].transAxes, ha="center", color=TEAL, fontsize=8.5)
save(fig, "03_zero_padding.png")

# ================================================================ Fig 4: windows and leakage
Lw, NB = 64, 64 * 512
wins = [("rect", "rectangular", SLATE), ("hann", "Hann", BLUE), ("hamming", "Hamming", RED)]
fb = np.arange(NB // 2 + 1) / (NB / Lw)            # frequency in bins (of an L-point DFT)
info = {}
for k, nm, col in wins:
    W = np.abs(np.fft.rfft(window(k, Lw, sym=True), NB))
    Wd = 20 * np.log10(W / W.max() + 1e-12)
    # first null = first local minimum; side-lobe level = max beyond it
    i_null = int(np.argmax((Wd[1:-1] < Wd[:-2]) & (Wd[1:-1] < Wd[2:]))) + 1
    sl = float(Wd[i_null:].max())
    first_sl = float(Wd[i_null:i_null + 2 * NB // Lw].max())
    bw3 = 2 * fb[np.argmax(Wd < -3.0103)]
    bw6 = 2 * fb[np.argmax(Wd < -6.0206)]
    info[k] = dict(W=Wd, null=fb[i_null], sl=sl, first=first_sl, bw3=bw3, bw6=bw6)
    NUM[f"sl_{k}"] = round(sl, 1)
    NUM[f"null_{k}"] = round(float(fb[i_null]) * 2, 2)
    NUM[f"bw3_{k}"] = round(float(bw3), 2)
    NUM[f"bw6_{k}"] = round(float(bw6), 2)
fig = plt.figure(figsize=(10.6, 6.6))
gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.3, height_ratios=[1, 1.15])
a = fig.add_subplot(gs[0, 0])
for k, nm, col in wins:
    a.plot(np.arange(Lw), window(k, Lw, sym=True), color=col, lw=1.6, label=nm)
a.set_ylim(0, 1.1); a.set_xlabel("sample n"); a.set_title("three windows (L = 64)")
a.legend(loc="lower center", fontsize=8)
b = fig.add_subplot(gs[0, 1:])
for k, nm, col in wins:
    b.plot(fb, info[k]["W"], color=col, lw=1.1, label=f"{nm}: main lobe ±{info[k]['null']:.0f} bins, highest side lobe {info[k]['sl']:.1f} dB")
    b.axhline(info[k]["sl"], color=col, lw=0.6, ls=":")
b.set_xlim(0, 12); b.set_ylim(-90, 3); b.set_xlabel("frequency offset (bins of $f_s/L$)"); b.set_ylabel("dB")
b.set_title("their spectra: main-lobe width vs side-lobe height")
b.legend(loc="upper right", fontsize=7.6)
# two-tone leakage test (synthetic)
L2, N2 = 512, 8192
fA, fB, gB = 1015.0, 1312.0, -50.0
NUM["lk_fA"], NUM["lk_fB"], NUM["lk_gB"] = int(fA), int(fB), int(gB)
NUM["lk_sep_bins"] = round((fB - fA) / (SR / L2), 1)
n2 = np.arange(L2)
sig = np.sin(2 * np.pi * fA * n2 / SR) + 10 ** (gB / 20) * np.sin(2 * np.pi * fB * n2 / SR + 1.0)
f2x = np.fft.rfftfreq(N2, 1 / SR)
for j, (k, nm, col) in enumerate(wins):
    a = fig.add_subplot(gs[1, j])
    w2 = window(k, L2)
    Y = np.abs(np.fft.rfft(sig * w2, N2))
    Yd = 20 * np.log10(Y / Y.max() + 1e-12)
    a.plot(f2x, Yd, color=col, lw=1.0)
    a.axvline(fB, color=GOLD, lw=0.8, ls=":")
    a.set_xlim(600, 1800); a.set_ylim(-100, 3)
    a.set_title(f"{nm} window", fontsize=9.5)
    a.set_xlabel("frequency (Hz)")
    # level at the weak tone vs leakage floor next to it
    iB = np.argmin(np.abs(f2x - fB))
    pkB = Yd[iB - 8:iB + 9].max()
    flo = np.median(Yd[(f2x > fB - 150) & (f2x < fB - 60)])
    NUM[f"lk_floor_{k}"] = round(float(flo), 1)
    seen = pkB - flo > 6
    a.text(fB + 25, -12, "weak tone\n" + ("visible" if seen else "buried"), color=TEAL if seen else RED, fontsize=8.5)
    if j == 0:
        a.set_ylabel("dB")
fig.text(0.5, 0.475, f"synthetic test: a strong tone at {fA:.0f} Hz plus a tone {abs(gB):.0f} dB weaker at {fB:.0f} Hz, L = 32 ms",
         ha="center", fontsize=9.5, color=INK, weight="bold")
save(fig, "04_windows_leakage.png")

# ================================================================ Fig 5/6: wideband vs narrowband
WL = [4, 10, 25, 40]
fig, ax = plt.subplots(4, 1, figsize=(10, 9.4), sharex=True, gridspec_kw={"hspace": 0.42})
for a, ms in zip(ax, WL):
    L = int(ms * SR / 1000)
    D, _ = spec_db(x, L, 16, 1024)
    show_spec(a, D, DUR, fmax=8000)
    a.set_ylabel("kHz")
    kind = "wideband" if L / SR < T0 else ("narrowband" if L / SR > 2 * T0 else "in between")
    a.set_title(f"L = {ms} ms ({L} samples) = {ms / 1000 / T0:.1f} pitch periods · {kind} · $\\Delta f \\approx f_s/L$ = {SR / L:.0f} Hz", loc="left", fontsize=9.5)
ax[-1].set_xlabel("time (s)")
save(fig, "05_wide_vs_narrow.png")
for ms in WL:
    NUM[f"ppw_{ms}"] = round(ms / 1000 / T0, 1)

za, zb = 1.60, 1.84                        # /ae/ of "cash"
seg = x[int((za - 0.05) * SR):int((zb + 0.05) * SR)]
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"wspace": 0.18})
for a, ms, title in [(ax[0], 4, "wideband, L = 4 ms: one striation per glottal pulse"),
                     (ax[1], 40, "narrowband, L = 40 ms: harmonics at multiples of F0")]:
    L = int(ms * SR / 1000)
    D, _ = spec_db(seg, L, 8, 2048)
    show_spec(a, D, len(seg) / SR, fmax=4000, t0=za - 0.05, vmin=-70)
    a.set_xlim(za, zb); a.set_xlabel("time (s)"); a.set_title(title, fontsize=9, loc="left")
ax[0].set_ylabel("kHz")
for m in vmarks[(vmarks > za) & (vmarks < zb)]:
    ax[0].plot([m, m], [3.75, 4.0], color=TEAL, lw=1.6, clip_on=False)
ax[0].text(za + 0.003, 3.5, "pitch marks (corpus)", color="white", fontsize=8)
# local F0 in this vowel from the pitch marks
mk = vmarks[(vmarks > za - 0.01) & (vmarks < zb + 0.01)]
loc = np.diff(mk)
ok = (loc > 0.004) & (loc < 0.0095)                  # drop the 10-ms "unvoiced" marks and gaps
F0loc = 1 / np.median(loc[ok])
NUM["F0_cash"] = round(float(F0loc))
NUM["F0_cash_start"] = round(float(1 / np.median(loc[ok][:5])))
NUM["F0_cash_end"] = round(float(1 / np.median(loc[ok][-5:])))
tm_ = (mk[:-1] + mk[1:])[ok] / 2
f0_ = 1 / loc[ok]
f0s = np.convolve(f0_, np.ones(3) / 3, "same")
for h in (1, 2, 3, 5, 10):
    ax[1].plot(tm_[1:-1], h * f0s[1:-1] / 1000, color="white", lw=0.8, ls=(0, (2, 3)), alpha=0.9)
    ax[1].text(zb + 0.002, h * f0s[-2] / 1000, rf"${h}\times F_0$", fontsize=8, color=INK, va="center")
ax[1].text(za + 0.003, 3.6, "dotted: multiples of F0 measured from the pitch marks", color="white", fontsize=7.5)
save(fig, "06_striations_vs_harmonics.png")

# ================================================================ Fig 7: linear vs dB, dynamic range
D80, S = spec_db(x, 400, 32, 1024, floor=120)
fig, ax = plt.subplots(2, 2, figsize=(10.6, 5.8), sharex=True, sharey=True, gridspec_kw={"hspace": 0.32, "wspace": 0.12})
panels = [(ax[0, 0], "linear magnitude $|X|$ / max", None), (ax[0, 1], "dB, 40 dB range", 40),
          (ax[1, 0], "dB, 60 dB range", 60), (ax[1, 1], "dB, 80 dB range", 80)]
for a, title, rngdb in panels:
    if rngdb is None:
        im = a.imshow(S / S.max(), origin="lower", aspect="auto", cmap=CMAP, vmin=0, vmax=1, extent=[0, DUR, 0, 8], interpolation="bilinear")
    else:
        im = a.imshow(D80, origin="lower", aspect="auto", cmap=CMAP, vmin=-rngdb, vmax=0, extent=[0, DUR, 0, 8], interpolation="bilinear")
    a.grid(False); a.set_title(title, fontsize=9.5, loc="left")
    cb = fig.colorbar(im, ax=a, pad=0.015, fraction=0.05); cb.ax.tick_params(labelsize=7.5)
for a in ax[:, 0]:
    a.set_ylabel("kHz")
for a in ax[1]:
    a.set_xlabel("time (s)")
save(fig, "07_linear_vs_db_range.png")
# how much of the time-frequency plane is visible above each floor
for r in (40, 60, 80):
    NUM[f"vis_{r}"] = round(100 * float(np.mean(D80 > -r)), 1)
NUM["vis_lin10"] = round(100 * float(np.mean(S / S.max() > 0.1)), 2)


# ================================================================ Fig 8: colour maps and their lightness
def lightness(rgb):
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    Y = 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]
    f = np.where(Y > (6 / 29) ** 3, np.cbrt(Y), Y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


CM = [("gray_r", "grey (dark = loud)"), ("viridis", "viridis"), ("magma", "magma"), ("jet", "jet (rainbow)")]
zs = (0.95, 2.35)
fig = plt.figure(figsize=(10.6, 7.0))
gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.12, height_ratios=[1, 0.11, 0.9])
i0, i1 = int(zs[0] * SR / 32), int(zs[1] * SR / 32)
for j, (cm, nm) in enumerate(CM):
    a = fig.add_subplot(gs[0, j])
    a.imshow(D80[:, i0:i1], origin="lower", aspect="auto", cmap=cm, vmin=-70, vmax=0, extent=[zs[0], zs[1], 0, 8], interpolation="bilinear")
    a.grid(False); a.set_title(nm, fontsize=9.5)
    a.set_xlabel("time (s)")
    if j:
        a.set_yticklabels([])
    else:
        a.set_ylabel("kHz")
    s = fig.add_subplot(gs[1, j])
    s.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cm, extent=[-70, 0, 0, 1])
    s.set_yticks([]); s.grid(False); s.set_xticks([-70, -35, 0]); s.tick_params(labelsize=7.5)
    s.set_xlabel("dB", fontsize=8, labelpad=1)
a = fig.add_subplot(gs[2, :])
vv = np.linspace(0, 1, 256)
for (cm, nm), col in zip(CM, [SLATE, TEAL, GOLD, RED]):
    Lst = lightness(plt.get_cmap(cm)(vv)[:, :3])
    a.plot(vv * 70 - 70, Lst, color=col, lw=1.8, label=nm)
    if cm == "jet":
        d = np.diff(Lst)
        NUM["jet_L_turns"] = int(np.sum(np.diff(np.sign(d[np.abs(d) > 1e-3])) != 0))
        NUM["jet_L_max_at_db"] = round(float(vv[np.argmax(Lst)] * 70 - 70), 1)
        NUM["jet_L_max"] = round(float(Lst.max()))
        NUM["jet_L_top"] = round(float(Lst[-1]))
        NUM["jet_L_bottom"] = round(float(Lst[0]))
    if cm == "viridis":
        NUM["viridis_monotonic"] = bool(np.all(np.diff(Lst) > 0))
a.set_xlabel("value (dB)"); a.set_ylabel("lightness $L^*$")
a.set_title("lightness along each colour map: grey, viridis and magma change steadily; jet goes up and down", loc="left", fontsize=9.5)
a.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.22))
save(fig, "08_colour_maps.png")

# ================================================================ Fig 9: overlap-add and COLA
L, Lh = 512, 256
wp, ws = window("hann", L), window("hann", L, sym=True)
M = 12
fig, ax = plt.subplots(3, 1, figsize=(10, 6.8), sharex=True, gridspec_kw={"hspace": 0.55})
setups = [(ax[0], wp, L // 2, 1, "periodic Hann, hop L/2 (50 % overlap): plain overlap-add"),
          (ax[1], ws, L // 2, 1, "symmetric Hann, hop L/2: plain overlap-add"),
          (ax[2], wp, L // 4, 2, "periodic Hann at analysis and synthesis, hop L/4 (75 %): sum of $w^2$")]
for a, wv, Hh, pw, title in setups:
    M = 1 + (11 * 256) // Hh                     # same total length in every row
    tot = np.zeros(L + Hh * (M - 1))
    for m in range(M):
        seg_ = wv ** pw
        tot[m * Hh:m * Hh + L] += seg_
        a.plot(np.arange(m * Hh, m * Hh + L), seg_, color="#AEB9CB", lw=0.8)
    mid = tot[L:-L]
    a.plot(np.arange(len(tot)), tot, color=RED, lw=1.6)
    rip = (mid.max() - mid.min()) / mid.mean()
    rtxt = "flat to machine precision" if rip < 1e-12 else f"ripple {100 * rip:.2g} %"
    a.set_title(title + f": sum = {mid.mean():.3f}, {rtxt}", loc="left", fontsize=9.5)
    a.set_yticks([0, 1, 1.5]); a.set_ylim(0, 1.75)
    a.axvspan(0, L, color="#F5F7FA", zorder=0); a.axvspan(len(tot) - L, len(tot), color="#F5F7FA", zorder=0)
ax[-1].set_xlabel("sample (edges shaded: fewer than the full number of windows overlap there)")
save(fig, "09_overlap_add.png")
# ripple numbers and actual reconstruction errors on the sentence
def ripple(wv, Hh, pw):
    tot = np.zeros(L + Hh * 40)
    for m in range(41):
        tot[m * Hh:m * Hh + L] += wv ** pw
    mid = tot[L:-L]
    return float(mid.mean()), float((mid.max() - mid.min()) / mid.mean())


NUM["cola_periodic50_sum"], r1 = ripple(wp, 256, 1)
NUM["cola_periodic50_ripple"] = r1
_, r2 = ripple(ws, 256, 1)
NUM["cola_sym50_ripple_pct"] = round(100 * r2, 2)
NUM["wola75_sum"], r3 = ripple(wp, 128, 2)
NUM["wola75_ripple"] = r3
_, r4 = ripple(wp, 256, 2)
NUM["wola50_ripple_pct"] = round(100 * r4, 1)
ones = np.ones(L)
X = stft(x, L, 256, kind="hann", pad=L)
y_ola, _ = istft(X, L, 256, len(x), wp, ones, normalise=False, pad=L)            # OLA, divide by nothing (sum is 1)
NUM["err_ola50"] = float(np.max(np.abs(y_ola - x)))
X = stft(x, L, 128, kind="hann", pad=L)
y_wola, _ = istft(X, L, 128, len(x), wp, wp, normalise=False, pad=L)
y_wola /= 1.5
NUM["err_wola75"] = float(np.max(np.abs(y_wola - x)))
X = stft(x, L, 256, w=ws, pad=L)
y_sym, _ = istft(X, L, 256, len(x), ws, ones, normalise=False, pad=L)
NUM["err_sym50"] = float(np.max(np.abs(y_sym - x)))
NUM["snr_sym50"] = round(float(10 * np.log10(np.sum(x ** 2) / np.sum((y_sym - x) ** 2))), 1)
print("recon errors", NUM["err_ola50"], NUM["err_wola75"], NUM["err_sym50"])

# ================================================================ Griffin-Lim and magnitude-only resynthesis
Lg, Hg = 512, 128
wg = window("hann", Lg)
Xg = stft(x, Lg, Hg, kind="hann")
Sg = np.abs(Xg)


def resyn(phase):
    y, _ = istft(Sg * np.exp(1j * phase), Lg, Hg, len(x), wg, wg, normalise=True)
    return y


def sc(y):
    return float(np.linalg.norm(Sg - np.abs(stft(y, Lg, Hg, kind="hann"))) / np.linalg.norm(Sg))


y_true = resyn(np.angle(Xg))
NUM["sc_true_phase"] = sc(y_true)
y_zero = resyn(np.zeros_like(Sg))
NUM["sc_zero_phase"] = round(sc(y_zero), 3)
NUM["zero_phase_buzz_hz"] = round(SR / Hg, 1)
phi = rng.uniform(-np.pi, np.pi, Sg.shape)
y = resyn(phi)
NUM["sc_random_phase"] = round(sc(y), 3)
SC = [sc(y)]
keep = {0: y.copy()}
NIT = 200
for it in range(1, NIT + 1):
    phi = np.angle(stft(y, Lg, Hg, kind="hann"))
    y = resyn(phi)
    SC.append(sc(y))
    if it in (10, 100, NIT):
        keep[it] = y.copy()
SC = np.array(SC)
NUM["sc_monotonic"] = bool(np.all(np.diff(SC) <= 1e-12))
NUM["sc_max_increase"] = float(np.max(np.diff(SC)))
for it in (0, 1, 10, 50, 100, NIT):
    NUM[f"sc_{it}"] = round(float(SC[it]), 3)
    NUM[f"scdb_{it}"] = round(float(20 * np.log10(SC[it])), 1)
NUM["gl_iters"] = NIT
NUM["gl_cells"] = int(Sg.size)
# fast Griffin-Lim (Perraudin, Balazs & Sondergaard 2013): the same two projections plus momentum 0.99
rng_f = np.random.default_rng(1984)
yf = resyn(rng_f.uniform(-np.pi, np.pi, Sg.shape))
Zprev = stft(yf, Lg, Hg, kind="hann")
SCF = [sc(yf)]
for it in range(1, NIT + 1):
    Z = stft(yf, Lg, Hg, kind="hann")
    T = Z + 0.99 * (Z - Zprev)
    Zprev = Z
    yf = resyn(np.angle(T))
    SCF.append(sc(yf))
SCF = np.array(SCF)
for it in (10, 50, 100, NIT):
    NUM[f"fgla_sc_{it}"] = round(float(SCF[it]), 3)
NUM["fgla_iters_to_match_gl100"] = int(np.argmax(SCF <= SC[100])) if np.any(SCF <= SC[100]) else None


def match(y):
    """same rms as the original, so the listening comparison is fair (wav() still guards the peak)"""
    return y * np.sqrt(np.mean(x ** 2) / np.mean(y ** 2))


# the zero-phase output repeats every hop: autocorrelation inside /ae/ of "cash"
yz = y_zero[int(1.62 * SR):int(1.82 * SR)]
rz = np.correlate(yz, yz, "full")[len(yz) - 1:]
rz /= rz[0]
NUM["zero_phase_lag"] = int(np.argmax(rz[40:400]) + 40)
NUM["zero_phase_r"] = round(float(rz[NUM["zero_phase_lag"]]), 2)
NUM["zero_phase_rms_db"] = round(float(10 * np.log10(np.mean(y_zero ** 2) / np.mean(x ** 2))), 1)
wav("sentence.wav", x)
wav("zero_phase.wav", match(y_zero))
wav("gl_000.wav", match(keep[0]))
wav("gl_010.wav", match(keep[10]))
wav("gl_100.wav", match(keep[100]))
wav(f"gl_{NIT}.wav", match(keep[NIT]))

fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.6), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.25]})
a = ax[0]
a.plot(np.arange(0, NIT + 1), SC, color=BLUE, lw=1.6)
a.set_xscale("symlog", linthresh=1)
a.set_xlim(-0.05, NIT * 1.5)
a.plot([0], [SC[0]], "o", color=RED, ms=5)
a.annotate(f"start (random phase): {SC[0]:.2f}", (0, SC[0]), xytext=(0.12, SC[0] + 0.03), fontsize=8, color=RED)
for it, dy in ((10, 0.05), (100, 0.07), (NIT, -0.07)):
    a.plot([it], [SC[it]], "o", color=INK, ms=4)
    a.annotate(f"{it}: {SC[it]:.3f}", (it, SC[it]), xytext=(it * 0.45, SC[it] + dy), fontsize=8, color=INK)
a.axhline(NUM["sc_zero_phase"], color=GOLD, lw=1, ls="--")
a.text(0.05, NUM["sc_zero_phase"] + 0.015, f"zero phase: {NUM['sc_zero_phase']:.2f}", fontsize=8, color=GOLD)
a.set_xlabel("Griffin–Lim iterations"); a.set_ylabel("spectral convergence")
a.set_ylim(0, max(SC[0], NUM["sc_zero_phase"]) + 0.12)
a.set_title(r"$\|S - |\mathrm{STFT}(y)|\|_F \,/\, \|S\|_F$", fontsize=10)
b = ax[1]
q0 = int(1.70 * SR)
tq = np.arange(q0, q0 + 480)
b.plot((tq - q0) / SR * 1000, x[tq], color=SLATE, lw=1.4, label="original")
b.plot((tq - q0) / SR * 1000, keep[100][tq], color=BLUE, lw=1.0, label="Griffin–Lim, 100 iterations")
b.plot((tq - q0) / SR * 1000, y_zero[tq] - 1.2, color=GOLD, lw=1.0, label="zero phase (shifted down)")
b.set_xlabel("time (ms), inside /æ/ of “cash”"); b.set_yticks([])
b.legend(loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.13))
save(fig, "10_griffin_lim.png")

# ================================================================ Fig 11: annotated sentence (IPA)
CCOL = {"vowel": BLUE, "diphthong": TEAL, "fricative": RED, "stop": GOLD, "affricate": GOLD, "nasal": "#7A4BA0",
        "approximant": "#2E8B57", "pause": "#8A94A3"}
Lw6 = int(0.006 * SR)
Dw, _ = spec_db(x, Lw6, 16, 512)
fig = plt.figure(figsize=(11, 5.2))
gs = fig.add_gridspec(3, 1, hspace=0.0, height_ratios=[4, 0.55, 0.45])
a = fig.add_subplot(gs[0])
show_spec(a, Dw, DUR, cmap="gray_r", vmin=-70)
a.set_ylabel("kHz"); a.tick_params(labelbottom=False)
a.set_xlim(0, DUR)
for s, e, p in ph[1:]:
    a.axvline(s, color=CCOL[CLASS[p]], lw=0.5, alpha=0.6)
a.set_title(f"“{TEXT}”  ·  ksp  ·  wideband (6 ms Hann window)", loc="left", fontsize=10)
pl = fig.add_subplot(gs[1], sharex=a)
pl.set_ylim(0, 1); pl.set_yticks([]); pl.grid(False); pl.tick_params(labelbottom=False)
for i, (s, e, p) in enumerate(ph):
    if p == "pau":
        continue
    pl.axvspan(s, e, ymin=0.08, ymax=0.92, color=CCOL[CLASS[p]], alpha=0.16, lw=0)
    pl.text((s + e) / 2, 0.5, arctic.IPA[p], ha="center", va="center", fontsize=10 if e - s > 0.04 else 8.5, color=CCOL[CLASS[p]], weight="bold")
wl = fig.add_subplot(gs[2], sharex=a)
wl.set_ylim(0, 1); wl.set_yticks([]); wl.grid(False)
for w_, i, j in WORDS:
    wl.text((ph[i][0] + ph[j][1]) / 2, 0.45, w_, ha="center", va="center", fontsize=9.5, color=INK, style="italic")
    wl.plot([ph[i][0] + 0.004, ph[j][1] - 0.004], [0.92, 0.92], color=INK, lw=0.8)
wl.set_xlabel("time (s)")
for sp in ("left", "right", "top"):
    pl.spines[sp].set_visible(False); wl.spines[sp].set_visible(False)
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
# same look as the phone boxes: pale fill, solid edge in the class colour
hand = [Patch(facecolor=to_rgba(CCOL[k], 0.22), edgecolor=CCOL[k], lw=1.2, label=k) for k in ["vowel", "diphthong", "fricative", "stop", "nasal", "approximant"]]
fig.legend(handles=hand, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
save(fig, "11_annotated_sentence.png")


# ================================================================ token measurements (sanity checks on the automatic labels)
def band_spec(y, nfft=4096):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), nfft)) ** 2
    return np.fft.rfftfreq(nfft, 1 / SR), Y


def fric_meas(u, s, e):
    xx = arctic.read_wav(u)
    y = xx[int((s + 0.01) * SR):int((e - 0.01) * SR)]
    f, Y = band_spec(y)
    m = f > 500
    Ys = np.convolve(10 * np.log10(Y + 1e-15), np.ones(41) / 41, "same")
    return dict(peak=float(f[m][np.argmax(Ys[m])]), above4=float(Y[f > 4000].sum() / Y[m].sum()),
                level=float(10 * np.log10(np.mean(y ** 2))))


def mid_formants(u, s, e, at=0.5):
    xx = arctic.read_wav(u)
    tm = s + at * (e - s)
    return formants(xx[int((tm - 0.0125) * SR):int((tm + 0.0125) * SR)])


def vot(u, s, e, vowel_end):
    xx = arctic.read_wav(u)
    y = xx[int(s * SR):int(vowel_end * SR)]
    Sx = np.abs(librosa.stft(y, n_fft=256, hop_length=16, win_length=64, window="hann", center=True)) ** 2
    f = np.fft.rfftfreq(256, 1 / SR)
    hf = 10 * np.log10(Sx[f > 2000].sum(0) + 1e-12)
    lf = 10 * np.log10(Sx[(f > 80) & (f < 400)].sum(0) + 1e-12)
    tt_ = np.arange(len(hf)) * 16 / SR + s
    d = np.diff(hf, prepend=hf[0])
    d[~((tt_ > s + 0.01) & (tt_ < e + 0.005))] = -99
    tb = float(tt_[int(np.argmax(d))])
    lfmax = lf[tt_ > e].max()
    tv = float(tt_[np.where((tt_ > tb) & (lf > lfmax - 12))[0][0]])
    return tb, tv


G = {}
# vowels /ae/ and /i/ in the sentence
s, e, _ = ph[TOK["ae_cash"]]
G["ae"] = mid_formants(UTT, s, e)[:3]
s, e, _ = ph[TOK["iy"]]
G["iy"] = mid_formants(UTT, s, e)[:3]
NUM["ae_F1"], NUM["ae_F2"] = round(G["ae"][0], -1), round(G["ae"][1], -1)
NUM["iy_F1"], NUM["iy_F2"] = round(G["iy"][0], -1), round(G["iy"][1], -1)
assert NUM["ae_F1"] > 550 and NUM["iy_F1"] < 400 and NUM["iy_F2"] > 2000
# fricatives: /s/ and /sh/ from the sentence, /f/ from "faced" (the /f/ in "life's" is only 50 ms)
FS = {"s": (UTT, *ph[TOK["s"]][:2]), "sh": (UTT, *ph[TOK["sh"]][:2]), "f": ("arctic_a0009", 1.39, 1.53)}
assert [p for s_, e_, p in arctic.phones("arctic_a0009") if abs(s_ - 1.39) < 1e-6] == ["f"]
for k, (u, s, e) in FS.items():
    m = fric_meas(u, s, e)
    NUM[f"{k}_peak"] = round(m["peak"], -1)
    NUM[f"{k}_above4_pct"] = round(100 * m["above4"])
    NUM[f"{k}_level"] = round(m["level"], 1)
NUM["f_weaker_than_s_db"] = round(NUM["s_level"] - NUM["f_level"], 1)
assert NUM["s_peak"] > NUM["sh_peak"] and NUM["s_above4_pct"] > NUM["sh_above4_pct"] and NUM["f_weaker_than_s_db"] > 6
# stop /k/ of "cash": closure, burst, voice onset
s, e, _ = ph[TOK["k"]]
clo_start = ph[TOK["k"] - 1][0]                     # the short pause before /k/ is the closure's start
tb, tv = vot(UTT, s, e, ph[TOK["k"] + 1][1])
NUM["k_burst_s"], NUM["k_voice_s"] = round(tb, 3), round(tv, 3)
NUM["k_vot_ms"] = round((tv - tb) * 1000)
NUM["k_closure_ms"] = round((tb - clo_start) * 1000)
clo = x[int((clo_start + 0.005) * SR):int((tb - 0.003) * SR)]
vow = x[int(ph[TOK["k"] + 1][0] * SR):int(ph[TOK["k"] + 1][1] * SR)]
NUM["k_closure_db"] = round(float(10 * np.log10(np.mean(clo ** 2) / np.mean(vow ** 2))), 1)
assert NUM["k_vot_ms"] > 25 and NUM["k_closure_db"] < -15
# /b/ of "But" for contrast
s, e, _ = ph[1]
tbb, tvb = vot(UTT, s, e, ph[2][1])
NUM["b_vot_ms"] = round((tvb - tbb) * 1000)
# ksp /k/ VOT over the corpus (word-internal and initial /k/ before a vowel, not after another stop)
vots = []
for u, s, e in arctic.find("k", min_dur=0.08, limit=200):
    P = arctic.phones(u)
    i = [kk for kk, p in enumerate(P) if abs(p[0] - s) < 1e-6][0]
    if i + 1 < len(P) and P[i + 1][2] in arctic.VOWELS and P[i - 1][2] not in arctic.STOPS:
        a_, b_ = vot(u, s, e, P[i + 1][1])
        vots.append(b_ - a_)
NUM["k_vot_corpus_med_ms"] = round(1000 * float(np.median(vots)))
NUM["k_vot_corpus_n"] = len(vots)
# nasal /m/ of "might" (in "all my might")
MU, ms_, me_ = "arctic_b0302", 1.895, 1.99
assert [p for s_, e_, p in arctic.phones(MU) if abs(s_ - ms_) < 1e-6] == ["m"]
xm = arctic.read_wav(MU)
f, Ym = band_spec(xm[int((ms_ + 0.02) * SR):int((me_ - 0.02) * SR)])
nxt = [(s_, e_) for s_, e_, p in arctic.phones(MU) if abs(s_ - me_) < 1e-6][0]
f, Yv = band_spec(xm[int((nxt[0] + 0.02) * SR):int((nxt[1] - 0.02) * SR)])
NUM["m_below500_pct"] = round(100 * float(Ym[f < 500].sum() / Ym.sum()))
NUM["vow_below500_pct"] = round(100 * float(Yv[f < 500].sum() / Yv.sum()))
band = (f > 1000) & (f < 3000)
NUM["m_vs_vow_1to3k_db"] = round(float(10 * np.log10(Ym[band].mean() / Yv[band].mean())), 1)
assert NUM["m_below500_pct"] > 90
# approximant /r/ of "destroy": F3 dip, and the speaker's typical /r/
RU, rs, re_ = "arctic_a0532", 1.09, 1.20
assert [p for s_, e_, p in arctic.phones(RU) if abs(s_ - rs) < 1e-6] == ["r"]
xr = arctic.read_wav(RU)


def f3_track(xx, a_, b_, step=0.005):
    out = []
    for tm in np.arange(a_, b_, step):
        F = formants(xx[int((tm - 0.0125) * SR):int((tm + 0.0125) * SR)])
        out.append((tm, F[2] if len(F) >= 3 else np.nan))
    return np.array(out)


tr = f3_track(xr, rs + 0.01, re_ - 0.005)
NUM["r_F3_min"] = round(float(np.nanmin(tr[:, 1])), -1)
oy = [(s_, e_) for s_, e_, p in arctic.phones(RU) if abs(s_ - re_) < 1e-6][0]
tro = f3_track(xr, oy[0] + 0.03, oy[1] - 0.03)
NUM["oy_F3_med"] = round(float(np.nanmedian(tro[:, 1])), -1)
rmins = []
for u, s, e in arctic.find("r", min_dur=0.06, limit=300):
    tr_ = f3_track(arctic.read_wav(u), s + 0.01, e - 0.005)
    if np.any(np.isfinite(tr_[:, 1])):
        rmins.append(np.nanmin(tr_[:, 1]))
NUM["r_corpus_n"] = len(rmins)
NUM["r_corpus_med_F3"] = round(float(np.median(rmins)), -1)
NUM["r_corpus_below2200_pct"] = round(100 * float(np.mean(np.array(rmins) < 2200)))
# diphthong /ai/ of "life's"
s, e, _ = ph[TOK["ay"]]
a_ = mid_formants(UTT, s, e, 0.18)
b_ = mid_formants(UTT, s, e, 0.85)
NUM["ay_F1_start"], NUM["ay_F2_start"] = round(a_[0], -1), round(a_[1], -1)
NUM["ay_F1_end"], NUM["ay_F2_end"] = round(b_[0], -1), round(b_[1], -1)
assert NUM["ay_F1_start"] > NUM["ay_F1_end"] and NUM["ay_F2_end"] > NUM["ay_F2_start"] + 300
print({k: v for k, v in NUM.items() if k.startswith(("ae", "iy", "s_", "sh", "f_", "k_", "b_", "m_", "uw", "r_", "oy", "ay"))})

# ================================================================ Fig 12: gallery of phone classes
def ctx(u, i_target, before=1, after=1):
    P = arctic.phones(u)
    a_ = P[max(0, i_target - before)][0]
    b_ = P[min(len(P) - 1, i_target + after)][1]
    return a_, b_, P


def idx_at(u, s):
    return [k for k, p in enumerate(arctic.phones(u)) if abs(p[0] - s) < 1e-6][0]


GAL = [
    ("vowel", "/æ/ in “cash”", UTT, 1.62, 1, 1, f"F1 ≈ {NUM['ae_F1']:.0f}, F2 ≈ {NUM['ae_F2']:.0f} Hz"),
    ("vowel", "/i/ in “she”", UTT, ph[TOK["iy"]][0], 1, 0, f"F1 ≈ {NUM['iy_F1']:.0f}, F2 ≈ {NUM['iy_F2']:.0f} Hz"),
    ("diphthong", "/aɪ/ in “life's”", UTT, ph[TOK["ay"]][0], 1, 1, f"F2 {NUM['ay_F2_start']:.0f} → {NUM['ay_F2_end']:.0f} Hz"),
    ("fricative", "/s/ in “life's”", UTT, ph[TOK["s"]][0], 1, 1, f"peak ≈ {NUM['s_peak'] / 1000:.1f} kHz, {NUM['s_above4_pct']:.0f}% above 4 kHz"),
    ("fricative", "/ʃ/ in “she”", UTT, ph[TOK["sh"]][0], 0, 1, f"peak ≈ {NUM['sh_peak'] / 1000:.1f} kHz, {NUM['sh_above4_pct']:.0f}% above 4 kHz"),
    ("fricative", "/f/ in “faced”", "arctic_a0009", 1.39, 1, 1, f"{NUM['f_weaker_than_s_db']:.0f} dB weaker than /s/, flat"),
    ("stop", "/k/ in “cash”", UTT, ph[TOK["k"]][0], 2, 1, f"closure {NUM['k_closure_ms']:.0f} ms, VOT ≈ {NUM['k_vot_ms']:.0f} ms"),
    ("nasal", "/m/ in “my might”", MU, ms_, 1, 1, f"{NUM['m_below500_pct']:.0f}% of energy below 500 Hz"),
    ("approximant", "/ɹ/ in “destroy”", RU, rs, 2, 1, f"F3 dips to ≈ {NUM['r_F3_min']:.0f} Hz"),
]
fig, ax = plt.subplots(3, 3, figsize=(11, 9.6), gridspec_kw={"hspace": 0.55, "wspace": 0.12})
# one fixed dB reference for every panel (the loudest bin of the demo sentence, as recorded), so a weak sound looks weak
REF = np.abs(stft(arctic.read_wav(UTT), Lw6, 8, 512)).max()
for n_, (cls, title, u, s0, bef, aft, meas) in enumerate(GAL):
    a = ax.flat[n_]
    it = idx_at(u, s0)
    a0_, b0_, P = ctx(u, it, bef, aft)
    pad = 0.03
    xx = arctic.read_wav(u)
    seg_ = xx[int((a0_ - pad) * SR):int((b0_ + pad) * SR)]
    D_ = 20 * np.log10(np.abs(stft(seg_, Lw6, 8, 512)).T / REF + 1e-12)
    show_spec(a, D_, len(seg_) / SR, cmap="gray_r", vmin=-70, t0=a0_ - pad)
    a.set_xlim(a0_ - pad, b0_ + pad)
    ts_, te_ = P[it][0], P[it][1]
    if u == UTT and P[it][2] == "sh":
        te_ = ph[TOK["sh"]][1]
    for tb_ in (ts_, te_):
        a.axvline(tb_, color=CCOL[cls], lw=1.6, ls="--")
    for k in range(max(0, it - bef), min(len(P), it + aft + 1)):
        s_, e_, p_ = P[k]
        if u == UTT and p_ == "pau" and e_ - s_ < 0.02:
            continue
        mid = (max(s_, a0_ - pad) + min(e_, b0_ + pad)) / 2
        a.text(mid, 7.35, arctic.IPA.get(p_, p_) if p_ not in ("pau", "ssil") else "", ha="center", fontsize=10,
               color=CCOL[cls] if k == it else INK, weight="bold" if k == it else "normal",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    a.set_title(f"{cls}: {title}", fontsize=9.5, loc="left", color=CCOL[cls], pad=15)
    a.text(0.0, 1.02, meas, transform=a.transAxes, fontsize=8, color=INK, va="bottom")
    a.tick_params(labelsize=7.5)
    if n_ % 3 == 0:
        a.set_ylabel("kHz")
    else:
        a.set_yticklabels([])
    # extra marks
    if cls == "stop":
        a.annotate("closure", (clo_start + 0.02, 0.3), fontsize=8, color=CCOL[cls], ha="left",
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        a.annotate("burst", (tb, 5.6), xytext=(tb - 0.075, 6.2), fontsize=8, color=CCOL[cls], arrowprops=dict(arrowstyle="->", color=CCOL[cls]),
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        a.annotate("", xy=(tv, 4.6), xytext=(tb, 4.6), arrowprops=dict(arrowstyle="<->", color=CCOL[cls], lw=1.2))
        a.text((tb + tv) / 2 + 0.03, 4.75, "VOT", fontsize=8, color=CCOL[cls], bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    if cls == "approximant":
        okr = tr[:, 1] < 2500                      # LPC picks a spurious "F3" in some frames; keep the dip
        a.plot(tr[okr, 0], tr[okr, 1] / 1000, "o", color=CCOL[cls], ms=3.5, mec="white", mew=0.5)
        a.plot([oy[0] + 0.03, oy[1] - 0.03], [NUM["oy_F3_med"] / 1000] * 2, color=CCOL[cls], lw=1.4, ls="--")
        a.text(tr[0, 0] - 0.005, np.nanmin(tr[:, 1]) / 1000 - 0.6, "F3", fontsize=8.5, color=CCOL[cls], weight="bold",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        a.text(oy[0] + 0.03, NUM["oy_F3_med"] / 1000 + 0.25, "F3 of /ɔɪ/", fontsize=7.5, color=CCOL[cls],
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    if cls in ("vowel", "diphthong"):
        F_ = G["ae"] if "cash" in title else G["iy"] if "she" in title else None
        if F_ is not None:
            for k, fv in enumerate(F_[:2]):
                a.plot([ts_ + 0.01, te_ - 0.01], [fv / 1000] * 2, color=CCOL[cls], lw=1.2, alpha=0.9)
                a.text(te_ + 0.004, fv / 1000, f"F{k + 1}", fontsize=8, color=CCOL[cls], va="center", weight="bold",
                       bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        else:
            for frac, F_ in ((0.18, a_), (0.85, b_)):
                tm = ts_ + frac * (te_ - ts_)
                a.plot([tm] * 2, [F_[0] / 1000, F_[1] / 1000], "o", color=CCOL[cls], ms=4.5, mec="white")
            a.annotate("", xy=(ts_ + 0.85 * (te_ - ts_), b_[1] / 1000), xytext=(ts_ + 0.18 * (te_ - ts_), a_[1] / 1000),
                       arrowprops=dict(arrowstyle="->", color=CCOL[cls], lw=1.3))
            a.text(ts_, a_[1] / 1000 + 0.45, "F2 glides up", fontsize=8, color=CCOL[cls],
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    wav(f"gal_{n_ + 1}.wav", seg_ * (0.9 / np.max(np.abs(seg_))))
for a in ax[-1]:
    a.set_xlabel("time (s)")
save(fig, "12_phone_gallery.png")

# ================================================================ Fig 13: average fricative spectra over the corpus
fig, a = plt.subplots(figsize=(9.6, 3.4))
for ph_, col, nm in [("s", RED, "/s/"), ("sh", BLUE, "/ʃ/"), ("f", GOLD, "/f/")]:
    acc, n_tok = 0, 0
    for u, s, e in arctic.find(ph_, min_dur=0.08, limit=150):
        xx = arctic.read_wav(u)
        y_ = xx[int((s + 0.02) * SR):int((e - 0.02) * SR)]
        Fr = librosa.util.frame(y_, frame_length=320, hop_length=160).T
        acc = acc + np.mean(np.abs(np.fft.rfft(Fr * np.hanning(320), 512, axis=1)) ** 2, axis=0)
        n_tok += 1
    fsp = np.fft.rfftfreq(512, 1 / SR)
    P_ = 10 * np.log10(acc / n_tok)
    a.plot(fsp / 1000, P_, color=col, lw=1.8, label=f"{nm} (mean of {n_tok} tokens)")
    pk_ = fsp[1:][np.argmax(P_[1:] * (fsp[1:] > 1000) - 1e9 * (fsp[1:] <= 1000))]
    NUM[f"avg_{ph_}_peak"] = round(float(pk_), -1)
    NUM[f"avg_{ph_}_ntok"] = n_tok
    NUM[f"avg_{ph_}_meandb"] = float(10 * np.log10(np.mean(acc / n_tok)))
    # "lower edge": the lowest frequency above 1 kHz where the average comes within 6 dB of its 1-7 kHz maximum
    band_ = (fsp > 1000) & (fsp < 7000)
    NUM[f"avg_{ph_}_edge"] = round(float(fsp[band_][np.argmax(P_[band_] > P_[band_].max() - 6)]), -2)
NUM["avg_s_minus_f_db"] = round(NUM["avg_s_meandb"] - NUM["avg_f_meandb"], 1)
NUM["avg_s_minus_sh_db"] = round(NUM["avg_s_meandb"] - NUM["avg_sh_meandb"], 1)
a.set_xlim(0, 8); a.set_ylim(-35, 9); a.set_xlabel("frequency (kHz)"); a.set_ylabel("power (dB, common ref.)")
a.set_title(f"ksp's fricatives on average: /ʃ/ rises from about {NUM['avg_sh_edge'] / 1000:.1f} kHz, /s/ from about "
            f"{NUM['avg_s_edge'] / 1000:.1f} kHz; /f/ is weak and flat", loc="left", fontsize=10)
a.legend(loc="lower center", ncol=3)
save(fig, "13_fricative_spectra.png")


# ================================================================ demo data: the sentence, its labels, colour maps
def lut(name, n=64):
    cm = plt.get_cmap(name)
    return "".join("%02x%02x%02x" % tuple(int(round(255 * v)) for v in cm(i / (n - 1))[:3]) for i in range(n))


pcm = np.round(x * 32767).astype("<i2").tobytes()
data = {
    "sr": SR, "utt": UTT, "text": TEXT, "b64": base64.b64encode(pcm).decode("ascii"),
    "phones": [[round(s, 3), round(e, 3), p, arctic.IPA[p] if p != "pau" else "", CLASS[p], word_of.get(i, "")] for i, (s, e, p) in enumerate(ph)],
    "words": [[round(ph[i][0], 3), round(ph[j][1], 3), w_] for w_, i, j in WORDS],
    "T0": round(T0, 5), "F0": NUM["F0_hz"],
    "cmaps": {"grey": lut("gray_r"), "viridis": lut("viridis"), "magma": lut("magma"), "jet": lut("jet")},
}
with open(os.path.join(HERE, "sentence.js"), "w", encoding="utf8") as fh:
    fh.write("// CMU ARCTIC ksp, " + UTT + " (Indian-English male, 16 kHz), peak-normalised to 0.9, int16 little-endian, base64;\n")
    fh.write("// automatic phone labels (start s, end s, ARPAbet, IPA, class, word); colour maps as 64 hex colours. Made by make_figures.py.\n")
    fh.write("window.SPEC_SENT = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("  js sentence.js", os.path.getsize(os.path.join(HERE, "sentence.js")) // 1024, "KB")

NUM = {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in NUM.items()}
json.dump(NUM, open(os.path.join(HERE, "numbers.json"), "w"), indent=1)
print(json.dumps(NUM, indent=1))
