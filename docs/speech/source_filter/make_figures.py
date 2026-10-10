"""
Figures and audio for ../source_filter.html.

Run from this folder:   python make_figures.py

All per-utterance speech (every clip, waveform, spectrum and spectrogram) comes from CMU ARCTIC speaker ksp
(Indian-English male, 16 kHz) through the shared reader ../common/arctic.py. Its phone labels are automatic,
so every token below was chosen by hand and checked acoustically (steady, voiced, plausible LPC formants).
Figure 7, the only corpus-aggregate figure, is made by aggregate/make_vowel_space.py (called at the end).
"""
import json
import os
import subprocess
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import arctic as A  # noqa: E402

FIG, AUD = os.path.join(HERE, "fig"), os.path.join(HERE, "audio")
os.makedirs(FIG, exist_ok=True)
os.makedirs(AUD, exist_ok=True)
SR = A.SR
NUM = {"speaker": A.SPEAKER, "credit": A.CREDIT}
SPK = "ksp, CMU ARCTIC"

INK, BLUE, RED, GOLD, SLATE = "#10233F", "#14508C", "#C1272D", "#C9971B", "#5B7083"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#AEB9CB", "axes.labelcolor": INK, "xtick.color": "#5C6B80", "ytick.color": "#5C6B80",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.color": "#E7ECF2", "grid.linewidth": 0.8,
    "mathtext.fontset": "cm",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig", name)


def write(name, y):
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.85
    fade = int(0.01 * SR)                                              # 10 ms fades: no clicks at the cut points
    ramp = np.ones(len(y)); ramp[:fade] = np.linspace(0, 1, fade); ramp[-fade:] = np.linspace(1, 0, fade)
    sf.write(os.path.join(AUD, name), (y * ramp).astype(np.float32), SR, subtype="PCM_16")
    print("  wav", name)


class Word:
    """One word of a ksp utterance, located by its (Festival) phone sequence. `vowel` is the phone of interest;
    `nth` picks which occurrence of that phone inside the word, `occ` which match of the phone
    sequence in the utterance (e.g. "dark" before "darker")."""

    def __init__(self, utt, word, pron, vowel=None, nth=0, occ=0):
        assert word.lower() in A.text(utt).lower(), (utt, word)
        self.utt, self.word, self.y = utt, word, A.read_wav(utt)
        ph, pr = A.phones(utt), pron.split()
        seq = [p for _, _, p in ph]
        i = [k for k in range(len(seq) - len(pr) + 1) if seq[k:k + len(pr)] == pr]
        self.phones = [(int(round(s * SR)), int(round(e * SR)), p) for s, e, p in ph[i[occ]:i[occ] + len(pr)]]
        self.a, self.b = self.phones[0][0], self.phones[-1][1]
        if vowel:
            self.va, self.vb, _ = [t for t in self.phones if t[2] == vowel][nth]

    def vowel(self):
        return self.y[self.va:self.vb]

    def clip(self, pad=0.05):
        return self.y[max(0, self.a - int(pad * SR)):self.b + int(pad * SR)]


def cepstral_envelope(frame, n_fft=2048, lifter_ms=2.0):
    """Smooth spectral envelope by keeping only low-quefrency cepstrum (method explained later on the page)."""
    X = np.fft.rfft(frame * np.hanning(len(frame)), n_fft)
    logm = np.log(np.abs(X) + 1e-9)
    c = np.fft.irfft(logm, n_fft)
    L = int(lifter_ms * SR / 1000)
    c[L:n_fft - L] = 0
    return np.fft.rfft(c, n_fft).real


def lpc_formants(fr, order=14, fmax=4000):
    pre = signal.lfilter([1, -0.97], [1], fr) * np.hamming(len(fr))
    a = librosa.lpc(pre, order=order)
    rts = [r for r in np.roots(a) if np.imag(r) > 0]
    return sorted(np.angle(r) * SR / (2 * np.pi) for r in rts
                  if -SR / np.pi * np.log(np.abs(r)) < 400 and 150 < np.angle(r) * SR / (2 * np.pi) < fmax), a


def f0_of(x, mid_ms=None):
    """Median pYIN F0 of a token; with mid_ms, only over the frames within ±mid_ms/2 of its middle
    (i.e. over the analysis window that a figure actually shows)."""
    f0, _, _ = librosa.pyin(x, fmin=60, fmax=300, sr=SR, frame_length=1024, hop_length=160)
    if mid_ms:
        c, h = len(f0) // 2, max(1, int(mid_ms / 20))
        f0 = f0[c - h:c + h + 1]
    return float(np.nanmedian(f0))


# ==================================================================== FIG 2: source vs filter in real speech
# One speaker, two pitches: the same vowels said near the start of a sentence (pitch high) and at the end (pitch low).
ROWS = [("low", "lower pitch", [Word("arctic_a0446", "me", "m iy", "iy", occ=0), Word("arctic_a0447", "dark", "d aa r k", "aa", occ=0)]),
        ("high", "higher pitch", [Word("arctic_a0586", "see", "s iy", "iy"), Word("arctic_b0524", "far", "f aa r", "aa")])]
LAB = {"iy": "/i/", "aa": "/ɑ/"}
fig, axs = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.32, wspace=0.08))
n_fft = 2048
ff = np.fft.rfftfreq(n_fft, 1 / SR)
NUM["fig2"] = {}
for r, (tag, desc, words) in enumerate(ROWS):
    for c, w in enumerate(words):
        seg = w.vowel()
        mid = len(seg) // 2
        fr = seg[mid - 320:mid + 320]                               # 40 ms: resolves harmonics
        X = np.abs(np.fft.rfft(fr * np.hanning(len(fr)), n_fft))
        ref = X.max()
        env = cepstral_envelope(fr, n_fft) * 20 / np.log(10) - 20 * np.log10(ref)
        F0 = f0_of(seg, 40)
        ax = axs[r, c]
        ax.plot(ff, 20 * np.log10(X / ref + 1e-9), color=SLATE, lw=0.8, label="spectrum (harmonics)")
        ax.plot(ff, env + 6, color=RED, lw=2.4, label="smooth envelope")
        for k in range(1, 6):                                         # mark a few harmonics: spacing = F0
            ax.plot([k * F0] * 2, [4, 9], color=BLUE, lw=1.2)
        ax.annotate("", xy=(F0, 6.5), xytext=(5 * F0, 6.5), arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
        ax.text(5 * F0 + 60, 5, rf"harmonics every $F_0 \approx {F0:.0f}$ Hz", color=BLUE, fontsize=8.5, ha="left", va="center")
        ax.set_xlim(0, 4000); ax.set_ylim(-70, 18)
        ax.set_title(f"{LAB['iy' if c == 0 else 'aa']} in “{w.word}” · {desc}", fontsize=10)
        if c == 0: ax.set_ylabel("dB")
        if r == 1: ax.set_xlabel("Frequency (Hz)")
        NUM["fig2"][w.word] = {"utt": w.utt, "F0": round(F0, 1)}
        write(f"ksp_{w.word.lower()}.wav", w.clip())
axs[0, 0].legend(loc="upper right")
save(fig, "01_source_vs_filter.png")


# ==================================================================== STEP 2: the source
def lpc_residual(x, order=18):
    pre = signal.lfilter([1, -0.97], [1], x)
    a = librosa.lpc(pre * np.hamming(len(pre)), order=order)
    return signal.lfilter(a, [1], pre)


DARK = ROWS[0][2][1]                                                   # “dark” in “growing dark and darker”
aa = DARK.vowel()
f0_aa = f0_of(aa, 40)
NUM["dark"] = {"utt": DARK.utt, "aa_ms": round(1000 * len(aa) / SR), "F0": round(f0_aa, 1)}
mid = len(aa) // 2
s0 = mid - 400
seg = aa[s0:s0 + 800]                                                  # 50 ms from the middle of /ɑ/
res = lpc_residual(aa)                                                 # residual over the whole vowel: no start-up transient
env = np.abs(res[s0:s0 + 800])
gci, _ = signal.find_peaks(env, distance=int(0.7 * SR / f0_aa), height=0.3 * np.percentile(env, 99.5))
# cross-check against the pitch marks distributed with the corpus
pm = np.round(A.pitchmarks(DARK.utt) * SR).astype(int) - DARK.va - s0
pm = pm[(pm >= 0) & (pm < 800)]
NUM["dark"]["gci_vs_pitchmarks_ms"] = [round(float(np.min(np.abs(pm - g))) / SR * 1000, 2) for g in gci]
t = np.arange(len(seg)) / SR * 1000
fig = plt.figure(figsize=(11, 3.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.18)
ax = fig.add_subplot(gs[0])
ax.plot(t, seg / np.abs(seg).max(), color=SLATE, lw=1.1)
for g in gci:
    ax.axvline(t[g], color=RED, lw=1, ls="--", alpha=0.8)
ax.plot(t[gci], np.full(len(gci), 1.12), "v", color=RED, ms=7)
ax.text(0.99, 0.03, "▼ = glottal closure (a “tap”)", transform=ax.transAxes, ha="right", va="bottom", color=RED, fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
T0s = np.diff(t[gci])
T0 = float(np.median(T0s))
ax.annotate("", xy=(t[gci[1]], -1.12), xytext=(t[gci[2]], -1.12), arrowprops=dict(arrowstyle="<->", color=BLUE))
ax.text((t[gci[1]] + t[gci[2]]) / 2, -1.3, rf"$T_0 \approx {t[gci[2]] - t[gci[1]]:.1f}$ ms  $\rightarrow$  $F_0 \approx {1000 / (t[gci[2]] - t[gci[1]]):.0f}$ Hz",
        ha="center", va="top", color=BLUE, fontsize=9)
ax.set_ylim(-1.5, 1.3); ax.set_xlabel("Time (ms)"); ax.set_yticks([])
ax.set_title(f"50 ms of /ɑ/ in “dark” ({SPK})", fontsize=10, loc="left")
ax = fig.add_subplot(gs[1])
L = int(0.9 * SR / f0_aa)
cols = plt.cm.Blues(np.linspace(0.45, 0.95, max(1, len(gci) - 1)))
for k, g in enumerate(gci[:-1]):
    p = seg[max(0, g - 8):g + L]
    ax.plot(np.arange(len(p)) / SR * 1000 - 0.5, p / np.abs(seg).max(), color=cols[k], lw=1.1)
ax.axvline(0, color=RED, lw=1, ls="--")
ax.set_xlabel("Time after closure (ms)"); ax.set_yticks([])
ax.set_title(f"{len(gci) - 1} periods, overlaid at their closure", fontsize=10, loc="left")
ax.text(0.97, 0.05, "kick → ring → decay → next kick", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK)
save(fig, "02_glottal_pulses.png")
NUM["dark"]["gci_periods_ms"] = [round(float(v), 2) for v in T0s]
NUM["dark"]["T0_median_ms"] = round(T0, 2)
NUM["dark"]["n_periods_overlaid"] = int(len(gci) - 1)

# wideband spectrogram: one vertical striation per glottal closure
pad = int(0.03 * SR)
a0, b0 = DARK.a - pad, DARK.b + pad
seg2 = DARK.y[a0:b0]
S = np.abs(librosa.stft(seg2, n_fft=512, win_length=64, hop_length=8, window="hann"))
tt = np.arange(S.shape[1]) * 8 / SR * 1000
fs_ = np.linspace(0, SR / 2, S.shape[0])
fig, axs = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.08))
axs[0].plot(np.arange(len(seg2)) / SR * 1000, seg2, color=SLATE, lw=0.6); axs[0].set_yticks([])
for a, b, p in DARK.phones:
    for k in (a, b):
        axs[0].axvline((k - a0) / SR * 1000, color="#AEB9CB", lw=0.6)
    axs[0].text(((a + b) / 2 - a0) / SR * 1000, 1.02, p, transform=axs[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=INK)
axs[1].pcolormesh(tt, fs_, 20 * np.log10(S / S.max() + 1e-6), cmap="magma", vmin=-60, vmax=0, shading="auto", rasterized=True)
axs[1].set_ylim(0, 5000); axs[1].set_ylabel("Frequency (Hz)"); axs[1].set_xlabel("Time (ms)"); axs[1].grid(False)
axs[0].set_title("“dark”, wideband (4 ms window) spectrogram: every vertical stripe is one glottal closure", fontsize=10, loc="left", pad=16)
save(fig, "03_striations.png")
pmv = A.pitchmarks(DARK.utt) * SR
NUM["dark"]["pitchmarks_in_aa"] = int(np.sum((pmv >= DARK.va) & (pmv < DARK.vb)))
write("ksp_dark.wav", DARK.clip())

# three source types: voiced, unvoiced, mixed
PLACE = Word("arctic_a0377", "place", "p l ey s", "s")
EASILY = Word("arctic_a0324", "easily", "iy z ax l iy", "z")
toks = [("voiced: /ɑ/ in “dark”", aa, BLUE), ("unvoiced: /s/ in “place”", PLACE.vowel(), RED),
        ("both: /z/ in “easily”", EASILY.vowel(), GOLD)]
fig, axs = plt.subplots(2, 3, figsize=(11.5, 5.2), gridspec_kw=dict(hspace=0.45, wspace=0.12, height_ratios=[1, 1.3]))
for j, (name, x, c) in enumerate(toks):
    mid = len(x) // 2
    w = x[mid - 240:mid + 240]                                          # 30 ms
    axs[0, j].plot(np.arange(len(w)) / SR * 1000, w / np.abs(w).max(), color=c, lw=0.9)
    if j == 2:  # reveal the voicing under the frication: low-pass below 400 Hz
        lp = signal.filtfilt(*signal.butter(4, 400 / (SR / 2)), x)[mid - 240:mid + 240]
        z_f0 = f0_of(x)
        axs[0, j].plot(np.arange(len(w)) / SR * 1000, lp / np.abs(lp).max() * 0.6 - 1.9, color=INK, lw=1.4)
        axs[0, j].text(30, -1.15, rf"low-passed $< 400$ Hz: a $\sim\!{1000 / z_f0:.1f}$ ms voicing ripple", ha="right", fontsize=8, color=INK)
        axs[0, j].set_ylim(-2.6, 1.1)
    axs[0, j].set_title(name, fontsize=10); axs[0, j].set_yticks([]); axs[0, j].set_xlabel("ms")
    fr = x[mid - 320:mid + 320]
    X = np.abs(np.fft.rfft(fr * np.hanning(len(fr)), 4096)); f = np.fft.rfftfreq(4096, 1 / SR)
    axs[1, j].plot(f, 20 * np.log10(X / X.max() + 1e-9), color=c, lw=0.8)
    axs[1, j].set_xlim(0, 8000); axs[1, j].set_ylim(-70, 3); axs[1, j].set_xlabel("Frequency (Hz)")
    if j: axs[1, j].set_yticklabels([])
    else: axs[1, j].set_ylabel("dB")
    if j == 1:
        NUM["s_energy_above_3k5"] = round(float((X[f > 3500] ** 2).sum() / (X ** 2).sum()), 3)
axs[1, 0].text(2900, -8, "harmonic comb\n" r"(lines every $F_0$)", fontsize=8.5, color=INK, bbox=dict(fc="white", ec="none", pad=1))
axs[1, 1].text(300, -66, "no lines: a continuous, noisy\nspectrum, energy mostly above 3.5 kHz", fontsize=8.5, color=INK, bbox=dict(fc="white", ec="none", pad=1))
axs[1, 2].text(1500, -66, "weak harmonics at the bottom,\nfrication noise on top", fontsize=8.5, color=INK, bbox=dict(fc="white", ec="none", pad=1))
save(fig, "04_source_types.png")
NUM["z"] = {"utt": EASILY.utt, "F0": round(z_f0, 1), "ripple_ms": round(1000 / z_f0, 1)}
NUM["s"] = {"utt": PLACE.utt}
for name, x in [("aa", aa), ("s", PLACE.vowel()), ("z", EASILY.vowel())]:
    write(f"ksp_{name}_token.wav", np.concatenate([x, np.zeros(1600)]))


# ==================================================================== STEP 3: resonance and formants
C_SOUND = 350.0  # m/s, warm humid air in the vocal tract


def quarter_wave(L_cm, n=4):
    return [(2 * k - 1) * C_SOUND / (4 * L_cm / 100) for k in range(1, n + 1)]


NUM["tube"] = {f"{L}cm": [round(v) for v in quarter_wave(L)] for L in (17.5, 14.5, 11.0)}

# --- formants are not harmonics: real /ɑ/ (ksp) and the same envelope under a synthetic 220 Hz source
ma = len(aa) // 2                                                      # 40 ms from the middle of the /ɑ/ in “dark”
fr = aa[ma - 320:ma + 320]
X = np.abs(np.fft.rfft(fr * np.hanning(len(fr)), n_fft)); ref = X.max()
fmts, a_lpc = lpc_formants(fr, order=14, fmax=3200)
_, Hr = signal.freqz([1], np.convolve(a_lpc, [1, -0.97]), worN=ff, fs=SR)   # AR envelope, pre-emphasis undone
F0_syn = 220.0
P = SR / F0_syn                                                        # synthetic: impulses every 1/220 s through the same all-pole filter
n = np.arange(int(0.3 * SR))
e = np.zeros(len(n)); e[np.round(np.arange(0, len(n), P)).astype(int)[:-1]] = 1.0
syn = signal.lfilter([1], a_lpc, e)
syn = signal.lfilter([1], [1, -0.97], syn)                             # undo the pre-emphasis the LPC fit was made on
fr_syn = syn[len(syn) // 2 - 320:len(syn) // 2 + 320]
NUM["fig8"] = {"F0_real": round(f0_aa, 1), "formants": [round(v) for v in fmts[:3]], "F0_synthetic": F0_syn}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, frame, F0, col, title in [(axs[0], fr, f0_aa, BLUE, rf"real: /ɑ/ in “dark” ({SPK}), $F_0 \approx {f0_aa:.0f}$ Hz"),
                                  (axs[1], fr_syn, F0_syn, RED, rf"synthetic: same envelope, $F_0 = {F0_syn:.0f}$ Hz")]:
    Xf = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n_fft)); spec = 20 * np.log10(Xf / Xf.max() + 1e-9)
    env = 20 * np.log10(np.abs(Hr))                                    # the same envelope in both panels
    env = env - env.max() + spec.max() + 3
    ax.plot(ff, spec, color=SLATE, lw=0.8)
    ax.plot(ff, env, color=col, lw=2.4)
    pk = [(int(np.argmin(np.abs(ff - F))), F) for F in fmts if F < 3200][:3]
    for k, (p_, Fk) in enumerate(pk):
        ax.annotate(rf"$F_{{{k + 1}}} \approx {Fk:.0f}$ Hz", xy=(ff[p_], env[p_]), xytext=(ff[p_] + 120, env[p_] + 9), fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col))
    for h in range(1, 40):
        if h * F0 < 3400 and (F0 > 150 or h % 2 == 1):
            ax.text(h * F0, -66, rf"$h_{{{h}}}$", ha="center", fontsize=7.5, color=SLATE)
    ax.set_xlim(0, 3500); ax.set_ylim(-70, 22); ax.set_xlabel("Frequency (Hz)")
    ax.set_title(title, fontsize=10)
axs[0].set_ylabel("dB")
save(fig, "06_formants_vs_harmonics.png")
NUM["fig8"]["F1_over_F0_real"] = round(fmts[0] / f0_aa, 2)
NUM["fig8"]["F1_over_F0_synthetic"] = round(fmts[0] / F0_syn, 2)

# ==================================================================== vowel gallery: waveform + spectrogram per vowel
GALLERY = [("i", Word("arctic_b0171", "seat", "s iy t", "iy")), ("ae", Word("arctic_a0248", "sad", "s ae d", "ae")),
           ("a", DARK), ("u", Word("arctic_b0236", "two", "t uw", "uw"))]
GAL_COL = {"i": BLUE, "ae": "#1D7A8C", "a": RED, "u": GOLD}
NUM["gallery"] = {}
for key, w in GALLERY:
    yy, va, vb = w.y, w.va, w.vb
    Fs = []                                                            # formants: LPC at 3 frames around the vowel middle, median
    for off in (-400, 0, 400):
        c0 = (va + vb) // 2 + off
        Fs.append(lpc_formants(yy[c0 - 240:c0 + 240], order=14)[0][:2])
    F1, F2 = np.median([f[0] for f in Fs]), np.median([f[1] for f in Fs])
    NUM["gallery"][key] = {"word": w.word, "utt": w.utt, "F1": round(float(F1)), "F2": round(float(F2)), "F0": round(f0_of(yy[va:vb]), 1),
                           "frames": [[round(v) for v in f] for f in Fs]}
    col = GAL_COL[key]
    mid_ = (va + vb) // 2                                              # waveform: 40 ms from the vowel middle
    seg = yy[mid_ - 320:mid_ + 320]
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    ax.plot(np.arange(len(seg)) / SR * 1000, seg / np.abs(seg).max(), color=col, lw=1.1)
    ax.set_xlim(0, 40); ax.set_ylim(-1.1, 1.1); ax.set_yticks([]); ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("ms", fontsize=8); ax.tick_params(labelsize=7.5)
    ax.spines["left"].set_visible(False)
    save(fig, f"07_gallery_wave_{key}.png")
    pad = 480                                                          # spectrogram of the word, vowel and F1/F2 marked
    s_ = max(0, w.a - pad)
    ws = yy[s_:w.b + pad]
    S = np.abs(librosa.stft(ws, n_fft=512, win_length=96, hop_length=32, window="hann"))
    tt = np.arange(S.shape[1]) * 32 / SR * 1000
    fs_ = np.linspace(0, SR / 2, S.shape[0])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.pcolormesh(tt, fs_, 20 * np.log10(S / S.max() + 1e-6), cmap="magma", vmin=-65, vmax=0, shading="auto", rasterized=True)
    t0 = (va - s_) / SR * 1000; t1 = (vb - s_) / SR * 1000
    for F, lab in [(F1, "F1"), (F2, "F2")]:
        ax.plot([t0, t1], [F, F], color="white", lw=2.2, ls=(0, (4, 2)))
        ax.text(t1 - 3, F + 60, rf"$\mathbf{{F_{lab[1]}}}$ {F:.0f}", color="white", fontsize=8, ha="right", va="bottom", fontweight="bold",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=1))      # label sits just above its line, inside the vowel
    ax.axvline(t0, color="white", lw=0.8, alpha=0.6); ax.axvline(t1, color="white", lw=0.8, alpha=0.6)
    ax.set_ylim(0, 4000); ax.set_xlim(tt[0], tt[-1]); ax.grid(False)
    ax.set_ylabel("Hz", fontsize=8); ax.set_xlabel("ms", fontsize=8); ax.tick_params(labelsize=7.5)
    save(fig, f"07_gallery_spec_{key}.png")
    write(f"gallery_{w.word}.wav", w.clip())

with open(os.path.join(HERE, "numbers.json"), "w", encoding="utf8") as fh:
    json.dump(NUM, fh, indent=1, ensure_ascii=False)
print(json.dumps(NUM, indent=1, ensure_ascii=False))

# ==================================================================== FIG 7 (corpus aggregate) lives in its own folder
agg = os.path.join(HERE, "aggregate")
if subprocess.run([sys.executable, "make_vowel_space.py"], cwd=agg).returncode:
    print("  WARNING: aggregate/make_vowel_space.py failed (needs the licensed corpus locally); Figure 7 not rebuilt")
