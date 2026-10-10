"""
Figures and audio for ../ar_model.html (part 3: the AR / all-pole model).

Run from this folder:   python make_figures.py
Outputs:  fig/*.png   audio/*.wav   numbers.json
Real speech: CMU ARCTIC speaker ksp (Indian-English male, 16 kHz), read through ../common/arctic.py.
CMU ARCTIC is free for any use with credit (J. Kominek & A. W. Black, Carnegie Mellon University);
the clips published here are analysed / resynthesised versions of one sentence.
"""
import json
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import arctic  # noqa: E402

FIG, AUD = os.path.join(HERE, "fig"), os.path.join(HERE, "audio")
os.makedirs(FIG, exist_ok=True)
os.makedirs(AUD, exist_ok=True)
SR = arctic.SR
rng = np.random.default_rng(11)
NUM = {}

UTT = "arctic_a0505"                       # "We fished sharks on Niihau together."  (2.97 s)
VOWEL_WORD = "sharks"                      # the /ɑ/ token used for figures 2 and 3

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


def write(name, y, peak=0.85):
    y = y / (np.max(np.abs(y)) + 1e-12) * peak
    sf.write(os.path.join(AUD, name), y.astype(np.float32), SR, subtype="PCM_16")


def lpc(frame, p):
    """Autocorrelation method (Hamming window + Levinson–Durbin). Returns predictor weights a[1..p] and the final error."""
    x = frame * np.hamming(len(frame))
    r = np.array([np.dot(x[:len(x) - k], x[k:]) for k in range(p + 1)])
    r[0] *= 1 + 1e-9
    A = np.zeros(p + 1); A[0] = 1; e = r[0]
    for i in range(1, p + 1):
        k = -(r[i] + np.dot(A[1:i], r[i - 1:0:-1])) / e
        A[1:i + 1] = A[1:i + 1] + k * np.concatenate([A[i - 1:0:-1], [1]])
        e *= 1 - k * k
    return -A[1:], e                       # x̂[n] = Σ a_k x[n−k]


def formants_from(a):
    roots = np.roots(np.concatenate([[1], -a]))
    out = []
    for z in roots:
        if np.imag(z) <= 0: continue
        F = np.angle(z) * SR / (2 * np.pi); B = -SR / np.pi * np.log(np.abs(z))
        if F > 90 and B < 400: out.append((F, B))
    return sorted(out)


y = arctic.read_wav(UTT)
NUM["utterance"] = UTT
NUM["text"] = arctic.text(UTT)
NUM["duration_s"] = round(len(y) / SR, 2)
phn = [(int(round(s * SR)), int(round(e * SR)), p) for s, e, p in arctic.phones(UTT)]
pre = signal.lfilter([1, -0.97], [1], y)

# ==================================================================== FIG 2 (page numbering): LPC envelope vs order
a0, b0 = [(a, b) for a, b, p in phn if p == "aa"][0]          # /ɑ/ in "sharks" (1.01–1.27 s, 260 ms)
c = (a0 + b0) // 2
NUM["vowel_token_s"] = [round(a0 / SR, 3), round(b0 / SR, 3)]
frame = pre[c - 256:c + 256]
X = np.abs(np.fft.rfft(frame * np.hamming(512), 2048)); f = np.fft.rfftfreq(2048, 1 / SR)
XdB = 20 * np.log10(X / X.max() + 1e-9)
orders = [2, 6, 12, 18, 40]
fig, axs = plt.subplots(len(orders), 1, figsize=(10, 9.6), sharex=True, gridspec_kw=dict(hspace=0.35))
NUM["order_formants"] = {}
NUM["order_bandwidths"] = {}
for ax, p in zip(axs, orders):
    a, err = lpc(frame, p)
    w, H = signal.freqz([1], np.concatenate([[1], -a]), worN=f, fs=SR)
    HdB = 20 * np.log10(np.abs(H)); HdB += XdB.max() - HdB.max() + 2
    ax.plot(f, XdB, color="#AEB9CB", lw=0.8)
    ax.plot(f, HdB, color=RED, lw=2.2)
    fm = formants_from(a)
    NUM["order_formants"][p] = [round(F) for F, B in fm][:8]
    NUM["order_bandwidths"][p] = [round(B) for F, B in fm][:8]
    for F, B in fm:
        if F < 4800: ax.plot([F], [np.interp(F, f, HdB) + 3], "v", color=INK, ms=5)
    label = {2: "too low: one broad hump", 6: r"still too low: $F_1$ and $F_2$ merged into one hump",
             12: r"better: $F_1$ and $F_2$ resolved, but $F_3$ only roughly placed (about 200 Hz high)",
             18: r"the rule of thumb for 16 kHz ($2 + 16$)", 40: "too high: starts tracing individual harmonics"}[p]
    ax.set_title(rf"$p = {p}$: {label}", loc="left", fontsize=10)
    ax.set_ylim(-75, 8); ax.set_ylabel("dB")
axs[-1].set_xlabel("Frequency (Hz)"); axs[-1].set_xlim(0, 5000)
save(fig, "01_envelope_vs_order.png")

# ==================================================================== frame-by-frame analysis / synthesis of the sentence
P, HOP, WIN = 18, 160, 400
N = len(pre)
nfr = N // HOP + (N % HOP > 0)
coefs = []
for i in range(nfr):
    cc = i * HOP + HOP // 2
    seg = pre[max(0, cc - WIN // 2):cc + WIN // 2]
    if len(seg) < WIN: seg = np.pad(seg, (0, WIN - len(seg)))
    a, err = lpc(seg, P)
    coefs.append(a)


def inverse_filter(x):
    e = np.zeros_like(x)
    for i in range(nfr):
        a = coefs[i]
        for n in range(i * HOP, min(N, (i + 1) * HOP)):
            past = x[max(0, n - P):n][::-1]
            e[n] = x[n] - np.dot(a[:len(past)], past)
    return e


def synth(e):
    out = np.zeros_like(e)
    for i in range(nfr):
        a = coefs[i]
        for n in range(i * HOP, min(N, (i + 1) * HOP)):
            past = out[max(0, n - P):n][::-1]
            out[n] = e[n] + np.dot(a[:len(past)], past)
    return out


deemph = lambda x: signal.lfilter([1], [1, -0.97], x)
res = inverse_filter(pre)
rebuilt = synth(res)
NUM["rebuild_error_db"] = float(10 * np.log10(np.sum((rebuilt - pre) ** 2) / np.sum(pre ** 2) + 1e-30))
NUM["residual_vs_speech_db"] = float(10 * np.log10(np.sum(res ** 2) / np.sum(pre ** 2)))

# excitation built from scratch: pulses at the tracked pitch where voiced, noise elsewhere, energy matched per frame
f0, vflag, _ = librosa.pyin(y, fmin=60, fmax=300, sr=SR, frame_length=1024, hop_length=HOP)
f0 = np.nan_to_num(f0, nan=0.0)
NUM["median_f0_hz"] = round(float(np.median(f0[vflag])), 1)
NUM["voiced_fraction"] = round(float(np.mean(vflag)), 2)
pm = arctic.pitchmarks(UTT)                                  # cross-check against the corpus pitch marks
NUM["pitchmark_median_f0_hz"] = round(float(1 / np.median(np.diff(pm))), 1)
res_rms = np.array([np.sqrt(np.mean(res[i * HOP:(i + 1) * HOP] ** 2) + 1e-12) for i in range(nfr)])


def excitation(pitch_fn, mode="voiced"):
    ex = np.zeros(N); ph = 0.0
    for i in range(nfr):
        voiced = mode != "whisper" and i < len(vflag) and vflag[i]
        blk = slice(i * HOP, min(N, (i + 1) * HOP))
        L = blk.stop - blk.start
        if voiced:
            F = pitch_fn(f0[i])
            seg = np.zeros(L)
            for n in range(L):
                ph += F / SR
                if ph >= 1: ph -= 1; seg[n] = 1.0
        else:
            seg = rng.standard_normal(L)
        seg = seg * res_rms[i] / (np.sqrt(np.mean(seg ** 2)) + 1e-12)
        ex[blk] = seg
    return ex


versions = {
    "original": y,
    "residual": res,
    "rebuilt": deemph(rebuilt),
    "vocoder": deemph(synth(excitation(lambda F: F))),
    "monotone": deemph(synth(excitation(lambda F: 100.0))),
    "higher": deemph(synth(excitation(lambda F: F * 1.8))),
    "whisper": deemph(synth(excitation(lambda F: F, "whisper"))),
}
for k, v in versions.items():
    write(f"ksp_{k}.wav", v)
print("  wrote vocoder audio")

# ==================================================================== FIG 3 (page numbering): speech, residual, rebuilt
t0 = c - 640; t1 = c + 640                        # 80 ms from the middle of /ɑ/ in "sharks"
tt = np.arange(t1 - t0) / SR * 1000
fig, axs = plt.subplots(3, 1, figsize=(10, 5.2), sharex=True, gridspec_kw=dict(hspace=0.3))
m = np.abs(pre[t0:t1]).max()
axs[0].plot(tt, pre[t0:t1] / m, color=SLATE, lw=1)
axs[0].set_title(f"speech (pre-emphasised): 80 ms of /ɑ/ in “{VOWEL_WORD}”", loc="left", fontsize=10)
axs[1].plot(tt, res[t0:t1] / np.abs(res[t0:t1]).max(), color=RED, lw=1)
axs[1].set_title(r"residual $e[n]$ = speech passed through $A(z)$: the vocal-tract ringing removed (rescaled to fill the panel)", loc="left", fontsize=10)
axs[2].plot(tt, pre[t0:t1] / m, color="#C9D2DE", lw=3, label="original")
axs[2].plot(tt, rebuilt[t0:t1] / m, color=BLUE, lw=1, label=r"residual passed back through $1/A(z)$")
axs[2].legend(loc="upper right", ncol=2); axs[2].set_title("rebuilt: identical to the original", loc="left", fontsize=10)
for ax in axs: ax.set_yticks([]); ax.set_ylim(-1.15, 1.15)
axs[2].set_xlabel("ms")
save(fig, "02_residual_and_rebuild.png")

# glottal closures inside the 80 ms window (corpus pitch marks), to check the "one pulse per period" claim
NUM["pitchmarks_in_fig3"] = int(np.sum((pm * SR >= t0) & (pm * SR < t1)))

with open(os.path.join(HERE, "numbers.json"), "w", encoding="utf8") as fh:
    json.dump(NUM, fh, indent=1, ensure_ascii=False)
print(json.dumps(NUM, indent=1, ensure_ascii=False))
