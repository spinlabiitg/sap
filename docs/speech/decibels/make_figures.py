"""
Figures and quoted numbers for ../decibels.html (extras: why decibels?).

Run from this folder:   python make_figures.py
Outputs:  fig/*.png   numbers.json
Speech: ../../data/real_sample_speech.wav (an Indian-English speaker, 48 kHz / 32 bit), resampled to 16 kHz and
peak-normalised to 0.9, the same "original" as the codecs page. The live demos reuse ../codecs/speech16k.js.
"""
import json
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)
NUM = {}
rng = np.random.default_rng(7)

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


SR = 16000
x, _ = librosa.load(os.path.join(HERE, "..", "..", "data", "real_sample_speech.wav"), sr=SR, mono=True)
x = x - np.mean(x)
x = 0.9 * x / np.max(np.abs(x))
t = np.arange(len(x)) / SR

# ---- pick a loud vowel frame and a fricative frame acoustically (no transcript available)
FL, HOP = 512, 160
frames = librosa.util.frame(np.pad(x, (0, FL)), frame_length=FL, hop_length=HOP).T
E = np.sum(frames ** 2, axis=1)
Z = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
v_idx = int(np.argmax(np.where(Z < 0.08, E, 0)))
f_idx = int(np.argmax(np.where(Z > 0.35, E, 0)))
NUM["vowel_t"] = round(v_idx * HOP / SR, 3)
NUM["fric_t"] = round(f_idx * HOP / SR, 3)
win = np.hanning(FL)


def spec(fr):
    X = np.fft.rfft(fr * win, 2048)
    return np.fft.rfftfreq(2048, 1 / SR), X


# ================================================================ Fig: everyday levels, linear vs dB
levels = [(0, "threshold of hearing"), (10, "breathing"), (30, "whisper"), (40, "quiet library"),
          (60, "conversation at 1 m"), (80, "busy road traffic"), (90, "lawnmower"), (110, "rock concert"),
          (120, "threshold of pain")]
P0 = 20e-6
fig, ax = plt.subplots(2, 1, figsize=(9.2, 4.6), gridspec_kw={"hspace": 0.95})
cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(levels)))
for (L, name), c in zip(levels, cols):
    p = P0 * 10 ** (L / 20)
    ax[0].plot(p, 0, "o", color=c, ms=8, mec="white")
    ax[1].plot(L, 0, "o", color=c, ms=8, mec="white")
    ax[1].annotate(name, (L, 0), xytext=(0, 12 if levels.index((L, name)) % 2 == 0 else -22),
                   textcoords="offset points", ha="center", fontsize=8, color=INK)
for (L, name) in levels[-2:]:
    ax[0].annotate(name, (P0 * 10 ** (L / 20), 0), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8, color=INK)
ax[0].annotate("all other seven sounds\nare squashed in here", (0.05, 0), xytext=(8, 0.55), textcoords="data",
               fontsize=8, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax[0].set_xlim(-0.6, 22); ax[0].set_ylim(-1, 1.2); ax[0].set_yticks([])
ax[0].set_xlabel("sound pressure (pascal), linear scale")
ax[0].set_title("Linear scale: sound pressure in Pa")
ax[1].set_xlim(-6, 132); ax[1].set_ylim(-1.2, 1.2); ax[1].set_yticks([])
ax[1].set_xlabel(r"sound pressure level (dB SPL, re $20\ \mu$Pa)")
ax[1].set_title("Decibel scale: every factor of 10 in pressure is the same step (20 dB)")
for a in ax:
    a.spines["left"].set_visible(False); a.grid(axis="y", visible=False)
save(fig, "01_levels_linear_vs_db.png")
NUM["pain_over_threshold_pressure"] = round(10 ** (120 / 20))

# ================================================================ Fig: one vowel spectrum, three views
f, Xv = spec(frames[v_idx])
f, Xf = spec(frames[f_idx])
fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.0), sharex=True, gridspec_kw={"hspace": 0.45, "wspace": 0.3})
for r, (X, nm) in enumerate([(Xv, "vowel"), (Xf, "fricative")]):
    mag = np.abs(X)
    ax[r, 0].plot(f / 1000, mag / mag.max(), color=BLUE, lw=0.9)
    ax[r, 1].plot(f / 1000, (mag / mag.max()) ** 2, color=BLUE, lw=0.9)
    db = 20 * np.log10(mag / mag.max() + 1e-6)
    ax[r, 2].plot(f / 1000, db, color=BLUE, lw=0.9)
    ax[r, 2].set_ylim(-80, 3)
    ax[r, 0].set_ylabel(f"{nm}\n(frame at {[NUM['vowel_t'], NUM['fric_t']][r]:.2f} s)")
ax[0, 0].set_title(r"magnitude $|X|$, linear")
ax[0, 1].set_title(r"power $|X|^2$, linear")
ax[0, 2].set_title(r"level $20\log_{10}|X|$ (dB)")
for a in ax[1]:
    a.set_xlabel("frequency (kHz)")
ax[0, 2].annotate("upper formants\nvisible in dB", (2.6, -38), xytext=(4.6, -12), fontsize=8, color=RED,
                  arrowprops=dict(arrowstyle="->", color=RED))
save(fig, "02_spectrum_three_views.png")
# how much of the vowel's power lies above 1 kHz
pv = np.abs(Xv) ** 2
NUM["vowel_power_above_1k_pct"] = round(100 * pv[f > 1000].sum() / pv.sum(), 2)
NUM["vowel_power_above_2k_pct"] = round(100 * pv[f > 2000].sum() / pv.sum(), 3)

# ================================================================ Fig: spectrogram linear vs dB
S = np.abs(librosa.stft(x, n_fft=512, hop_length=80, win_length=400, window="hann"))
fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"hspace": 0.3})
ext = [0, len(x) / SR, 0, SR / 2000]
ax[0].imshow(S / S.max(), origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=0.5)
ax[0].set_title("Linear magnitude: only the strongest low harmonics of the loudest vowels show")
Sdb = 20 * np.log10(S / S.max() + 1e-9)
ax[1].imshow(Sdb, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=-80, vmax=0)
ax[1].set_title("Decibels (80 dB range): formants, fricatives, harmonics and quiet parts all visible")
for a in ax:
    a.set_ylabel("kHz"); a.grid(False)
ax[1].set_xlabel("time (s)")
save(fig, "03_spectrogram_linear_vs_db.png")

# ================================================================ Fig: source + filter = speech, in dB
p = 18
fr = frames[v_idx] * win
a = librosa.lpc(fr.astype(np.float64), order=p)
w, A = signal.freqz(a, 1, worN=1025, fs=SR)
X = np.fft.rfft(fr, 2048)
Res = X * A                                        # spectrum of the prediction error (the "source")
g = np.sqrt(np.mean(np.abs(Res) ** 2))
src_db = 20 * np.log10(np.abs(Res) / g + 1e-9)
flt_db = 20 * np.log10(g / np.abs(A))
tot_db = 20 * np.log10(np.abs(X) + 1e-9)
fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.3), sharey=False, gridspec_kw={"wspace": 0.35})
ax[0].plot(w / 1000, src_db, color=GOLD, lw=0.9); ax[0].set_title("source (error spectrum)")
ax[1].plot(w / 1000, flt_db, color=TEAL, lw=1.6); ax[1].set_title(rf"filter (LPC envelope, $p = {p}$)")
ax[2].plot(w / 1000, tot_db, color=BLUE, lw=0.9); ax[2].plot(w / 1000, flt_db, color=TEAL, lw=1.2, alpha=0.7)
ax[2].set_title("speech = source + filter")
for a_ in ax:
    a_.set_xlabel("frequency (kHz)")
ax[0].set_ylabel("dB")
ax[0].set_ylim(-40, 25)
fig.text(0.355, 0.5, r"$+$", fontsize=22, color=INK, ha="center", va="center", weight="bold")
fig.text(0.648, 0.5, r"$=$", fontsize=22, color=INK, ha="center", va="center", weight="bold")
save(fig, "04_source_plus_filter_db.png")
NUM["sf_max_abs_err_db"] = float(np.max(np.abs(src_db + flt_db - tot_db)))

# ================================================================ Fig: same speech at four SNRs
loud = int(np.argmax(np.convolve(x ** 2, np.ones(4800), "valid")))
seg = x[loud:loud + 4800]
P = np.mean(x ** 2)
fig, ax = plt.subplots(4, 1, figsize=(9.5, 5.6), sharex=True, gridspec_kw={"hspace": 0.5})
for a_, snr in zip(ax, [30, 20, 10, 0]):
    nz = rng.standard_normal(len(seg)) * np.sqrt(np.mean(seg ** 2) / 10 ** (snr / 10))
    a_.plot(np.arange(len(seg)) / SR * 1000, seg + nz, color=RED, lw=0.5, label="speech + noise")
    a_.plot(np.arange(len(seg)) / SR * 1000, seg, color=BLUE, lw=0.5, alpha=0.8, label="speech alone")
    a_.set_title(f"SNR {snr} dB: noise amplitude (rms) = {100 * 10 ** (-snr / 20):.1f}% of the speech's", loc="left", fontsize=9.5)
    a_.set_ylim(-1.05, 1.05); a_.set_yticks([])
ax[-1].set_xlabel("time (ms)")
ax[0].legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1.45))
save(fig, "05_snr_ladder.png")

# ================================================================ Fig: one global SNR, many local SNRs
nz = rng.standard_normal(len(x)) * np.sqrt(P / 10 ** (20 / 10))
n = 320
Ef = np.array([np.sum(x[i:i + n] ** 2) for i in range(0, len(x) - n, n)])
Nf = np.array([np.sum(nz[i:i + n] ** 2) for i in range(0, len(x) - n, n)])
tf = (np.arange(len(Ef)) * n + n / 2) / SR
snr_f = 10 * np.log10(Ef / Nf)
keep = Ef > Ef.max() * 1e-4
seg_snr = float(np.mean(np.clip(snr_f[keep], -10, 40)))
glob = 10 * np.log10(np.sum(x ** 2) / np.sum(nz ** 2))
fig, ax = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True, gridspec_kw={"hspace": 0.35, "height_ratios": [1, 1.4]})
ax[0].plot(t, x + nz, color=RED, lw=0.3); ax[0].plot(t, x, color=BLUE, lw=0.4); ax[0].set_yticks([]); ax[0].set_title("the recording, with white noise added at a global SNR of 20 dB", loc="left")
ax[1].plot(tf, snr_f, color=SLATE, lw=1.1, label="SNR of each 20 ms frame")
ax[1].axhline(glob, color=RED, lw=1.4, ls="--", label=f"global SNR (total energies) = {glob:.1f} dB")
ax[1].axhline(seg_snr, color=GOLD, lw=1.4, label=f"segmental SNR (mean of frame dB) = {seg_snr:.1f} dB")
ax[1].set_ylabel("dB"); ax[1].set_xlabel("time (s)"); ax[1].legend(loc="lower left", ncol=3, fontsize=8)
ax[1].set_ylim(-25, 45)
save(fig, "06_global_vs_segmental_snr.png")
NUM["glob_snr"] = round(float(glob), 1)
NUM["seg_snr"] = round(seg_snr, 1)
NUM["frame_snr_max"] = round(float(snr_f[keep].max()), 1)
NUM["frame_snr_min"] = round(float(snr_f[keep].min()), 1)

json.dump(NUM, open(os.path.join(HERE, "numbers.json"), "w"), indent=1)
print(json.dumps(NUM, indent=1))
