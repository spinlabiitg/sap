"""
The one corpus-AGGREGATE figure of ../../source_filter.html (Figure 7: the uniform tube in the vowel space).

TIMIT (LDC) is not redistributable, so on the public site it is used ONLY for aggregate statistics:
this script measures F1/F2 on every long vowel token of the 168 TIMIT TEST speakers and publishes nothing
but a 2-D density of those measurements and the per-vowel medians. No audio, waveform, spectrogram or
per-token value of any TIMIT utterance is written into docs/. The per-token measurements are cached
OUTSIDE the repository (set SF_CACHE to change where).

Run from this folder:   python make_vowel_space.py      (also called by ../make_figures.py)
"""
import glob
import json
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
TIMIT_ROOT = os.environ.get("TIMIT_ROOT", r"C:\Users\IITG\Documents\tommy_assistant\research\dBase\TIMIT\data\lisa\data\timit\raw\TIMIT")
CACHE = os.environ.get("SF_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "sap"))
SR = 16000
C_SOUND = 350.0
VOWELS = ["iy", "ih", "eh", "ae", "aa", "ah", "ao", "er"]
IPA = {"iy": "i", "ih": "ɪ", "eh": "ɛ", "ae": "æ", "aa": "ɑ", "ah": "ʌ", "ao": "ɔ", "er": "ɝ"}

INK, BLUE, RED, GOLD = "#10233F", "#14508C", "#C1272D", "#C9971B"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#AEB9CB", "axes.labelcolor": INK, "xtick.color": "#5C6B80", "ytick.color": "#5C6B80",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.color": "#E7ECF2", "grid.linewidth": 0.8,
    "mathtext.fontset": "cm",
})


def read_sphere(path):
    """NIST SPHERE: 1024-byte ASCII header, then little-endian int16 at 16 kHz."""
    return np.fromfile(path, dtype="<i2", offset=1024).astype(np.float32) / 32768.0


def read_phn(path):
    with open(path) as fh:
        return [(int(a), int(b), p) for a, b, p in (l.split() for l in fh)]


def lpc_formants(fr, order):
    fr = signal.lfilter([1, -0.97], [1], fr) * np.hamming(len(fr))
    a = librosa.lpc(fr, order=order)
    roots = [r for r in np.roots(a) if np.imag(r) > 0]
    F = np.angle(roots) * SR / (2 * np.pi)
    B = -SR / np.pi * np.log(np.abs(roots))
    return sorted(f for f, b in zip(F, B) if 150 < f < 5000 and b < 400)


def measure():
    """(sex, vowel, F1, F2) for every vowel token >= 60 ms of the TEST set, LPC on the middle 30 ms."""
    path = os.path.join(CACHE, "timit_test_vowel_formants.json")
    if os.path.exists(path):
        return json.load(open(path))
    print("  measuring F1/F2 of TIMIT TEST vowels (LPC, middle 30 ms of each token) ...")
    res, spk = [], sorted(glob.glob(os.path.join(TIMIT_ROOT, "TEST", "DR*", "*")))
    assert len(spk) == 168, f"expected 168 TEST speakers under {TIMIT_ROOT}, found {len(spk)}"
    for d in spk:
        sex = os.path.basename(d)[0]
        order = 12 if sex == "F" else 14
        for wav in sorted(glob.glob(os.path.join(d, "*.WAV"))):
            y, phn = read_sphere(wav), read_phn(wav[:-4] + ".PHN")
            for a, b, p in phn:
                if p in VOWELS and b - a >= int(0.06 * SR):
                    c = (a + b) // 2
                    Fs = lpc_formants(y[c - 240:c + 240], order)
                    if len(Fs) >= 2 and Fs[0] < 1200:
                        res.append((sex, p, Fs[0], Fs[1]))
    os.makedirs(CACHE, exist_ok=True)
    json.dump(res, open(path, "w"))
    return res


def quarter_wave(L_cm, n=4):
    return [(2 * k - 1) * C_SOUND / (4 * L_cm / 100) for k in range(1, n + 1)]


vf = measure()
_spk = [os.path.basename(d)[0] for d in glob.glob(os.path.join(TIMIT_ROOT, "TEST", "DR*", "*"))]
stats = {"n_speakers": {s: _spk.count(s) for s in "MF"}, "n_tokens": {}, "median": {}}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(wspace=0.22))
for ax, sex, L, col, cmap, lab in [(axs[0], "M", 17.5, BLUE, "Blues", "male speakers"), (axs[1], "F", 14.5, RED, "Reds", "female speakers")]:
    pts = np.array([(f1, f2) for g, p_, f1, f2 in vf if g == sex])
    stats["n_tokens"][sex] = int(len(pts))
    # corpus distribution as a smoothed 2-D density (aggregate), not individual tokens
    H, xe, ye = np.histogram2d(pts[:, 1], pts[:, 0], bins=[np.arange(800, 2925, 25), np.arange(250, 1012.5, 12.5)])
    H = gaussian_filter(H, 2.5)
    H = H / H.max()
    xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    ax.contourf(xc, yc, H.T, levels=[0.08, 0.25, 0.45, 0.7, 1.01], cmap=cmap, alpha=0.55, vmin=-0.3, vmax=1.2)
    stats["median"][sex] = {}
    for v in VOWELS:
        q = np.array([(f1, f2) for g, p_, f1, f2 in vf if g == sex and p_ == v])
        F1, F2 = np.median(q, 0)
        stats["median"][sex][v] = [round(float(F1)), round(float(F2)), int(len(q))]
        ax.plot(F2, F1, "o", color=col, ms=8, mec="white")
        dx, dy = (-40, 30) if v == "er" else (30, -15)              # keep /ɝ/ clear of the star
        ax.text(F2 + dx, F1 + dy, "/" + IPA[v] + "/", fontsize=10, color=INK, ha="right" if v == "er" else "left")
    F = quarter_wave(L)
    ax.plot(F[1], F[0], "*", color=GOLD, ms=22, mec=INK, mew=0.8, zorder=5)
    ax.annotate(rf"uniform tube, $L = {L}$ cm" "\n" rf"$F_1 = {F[0]:.0f}$ Hz, $F_2 = {F[1]:.0f}$ Hz", xy=(F[1], F[0]), xytext=(2850, 930),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GOLD))
    ax.set_xlim(2900, 800); ax.set_ylim(1000, 250)
    ax.set_xlabel(r"$F_2$ (Hz)   ← front … back →")
    ax.set_title(f"TIMIT {lab} ({stats['n_speakers'][sex]}): vowel medians", fontsize=10)
axs[0].set_ylabel(r"$F_1$ (Hz)   ← close … open →")
fig.savefig(os.path.join(HERE, "05_neutral_tube_in_vowel_space.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
with open(os.path.join(HERE, "vowel_space_stats.json"), "w", encoding="utf8") as fh:
    json.dump(stats, fh, indent=1)
print("  fig aggregate/05_neutral_tube_in_vowel_space.png", stats["n_tokens"])
