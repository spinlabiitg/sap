"""
Figures, audio and embedded sample data for ../linear_prediction.html.

Run from this folder:   python make_figures.py
Outputs:  fig/*.png   audio/*.wav   signals.js (short signal excerpts for the page's interactive demos)   numbers.json
Real speech: CMU ARCTIC speaker ksp (Indian-English male, 16 kHz), read through ../common/arctic.py.
signals.js is also loaded by ../ar_model.html: keep the global names (LP_SIGNALS with keys noise/tone/aa/iy/s,
LP_SR, LP_T0, LP_SUNSPOTS) and the 1280-sample excerpt length.
"""
import json
import os
import sys

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
SR = 16000
rng = np.random.default_rng(4)
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


def write(name, y):
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.85
    sf.write(os.path.join(AUD, name), y.astype(np.float32), SR, subtype="PCM_16")


def corr(x, k):
    """Normalised correlation between x[n] and x[n+k] (same estimator the page uses)."""
    x = x - x.mean()
    return float(np.dot(x[:-k], x[k:]) / np.dot(x, x)) if k else 1.0


def lpc(x, p):
    """Autocorrelation-method linear prediction (Levinson–Durbin). Returns predictor weights a[1..p]."""
    x = x * np.hamming(len(x))
    r = np.array([np.dot(x[:len(x) - k], x[k:]) for k in range(p + 1)])
    a = np.zeros(p + 1); a[0] = 1; e = r[0]
    for i in range(1, p + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        a[1:i + 1] = a[1:i + 1] + k * np.concatenate([a[i - 1:0:-1], [1]])
        e *= 1 - k * k
    return -a[1:]                     # x̂[n] = Σ w_k x[n-k],  w = -a


# ==================================================================== signals
# Speech excerpts: CMU ARCTIC speaker ksp. The phone labels are automatic (EHMM), so every token was checked
# acoustically before being chosen (long, steady pitch marks, clear formants, a strong pitch-period
# autocorrelation peak for the vowels; an unvoiced, high-frequency spectrum for /s/):
#   /ɑ/  arctic_a0143 "Ah, I had forgotten, he exclaimed."   the exclamation "Ah": 260 ms, F1 ≈ 660 Hz, F2 ≈ 1210 Hz,
#        pitch marks 116 ± 2 samples apart. Chosen over "arm"/"dark" tokens because its F1 ringing is strong enough
#        for r(k) to swing clearly negative at half an F1 cycle (the point Figure 1 makes).
#   /i/  arctic_a0236 "Don't you see I hate you."   "see": 230 ms, F1 ≈ 290 Hz, F2 ≈ 2250 Hz, pitch marks 107 ± 1.5.
#   /s/  the same "see": 160 ms, spectral centroid ≈ 5 kHz, no voicing.
UTT_AA, UTT_SEE = "arctic_a0143", "arctic_a0236"
UTTS = {u: {"y": arctic.read_wav(u), "pm": arctic.pitchmarks(u),
            "phn": [(int(round(a * SR)), int(round(b * SR)), p) for a, b, p in arctic.phones(u)]} for u in (UTT_AA, UTT_SEE)}
NUM["source"] = {"speaker": arctic.SPEAKER, "aa": [UTT_AA, arctic.text(UTT_AA)], "iy_s": [UTT_SEE, arctic.text(UTT_SEE)]}


def token(utt, phone, which):
    return [(a, b) for a, b, p in UTTS[utt]["phn"] if p == phone][which]


def middle(utt, phone, which, n):
    a, b = token(utt, phone, which)
    c = (a + b) // 2
    return UTTS[utt]["y"][c - n // 2:c + n // 2]


def word_clip(utt, first, last):
    """From the start of phone `first` to the end of phone `last` (each (phone, occurrence)), padded by 37.5 ms."""
    a, b = token(utt, *first)[0], token(utt, *last)[1]
    return UTTS[utt]["y"][max(0, a - 600):b + 600]


N = 1280                                              # 80 ms excerpts
AA, IY, S_ = (UTT_AA, "aa", 0), (UTT_SEE, "iy", 0), (UTT_SEE, "s", 0)
SIG = {
    "noise": rng.standard_normal(N),
    "tone": np.sin(2 * np.pi * 200 * np.arange(N) / SR),
    "aa": middle(*AA, N),
    "iy": middle(*IY, N),
    "s": middle(*S_, N),
}
LABEL = {"noise": "white noise", "tone": "pure tone (200 Hz)", "aa": "vowel /ɑ/ (“Ah”)", "iy": "vowel /i/ (“see”)", "s": "fricative /s/ (“see”)"}
for k in SIG:
    SIG[k] = SIG[k] - SIG[k].mean()
    SIG[k] = SIG[k] / np.abs(SIG[k]).max()

# pitch period of the /ɑ/ excerpt from its autocorrelation peak, cross-checked against the corpus pitch marks
r_aa = np.array([corr(SIG["aa"], k) for k in range(0, 300)])
T0 = int(70 + np.argmax(r_aa[70:267]))   # search 60–230 Hz (adult male)
a_, b_ = token(*AA)
c_ = (a_ + b_) // 2
pm = UTTS[UTT_AA]["pm"]
pm_in = pm[(pm >= (c_ - N // 2) / SR) & (pm <= (c_ + N // 2) / SR)]
NUM["T0_samples"], NUM["F0_aa"] = T0, SR / T0
NUM["T0_pitchmarks"] = float(np.median(np.diff(pm_in)) * SR)
assert abs(NUM["T0_pitchmarks"] - T0) <= 3, "autocorrelation and pitch marks disagree on the /ɑ/ pitch period"

# embedded data for the page (short, rounded)
with open(os.path.join(HERE, "signals.js"), "w", encoding="utf8") as fh:
    fh.write("// Generated by make_figures.py: 80 ms excerpts (16 kHz), peak-normalised.\n")
    fh.write("// Speech: CMU ARCTIC speaker ksp (Indian English), Carnegie Mellon University (J. Kominek & A. W. Black); "
             f"/ɑ/ from {UTT_AA} (\"Ah\"), /i/ and /s/ from {UTT_SEE} (\"see\"). Excerpts trimmed and peak-normalised for this site.\n")
    fh.write("window.LP_SIGNALS = " + json.dumps({k: {"label": LABEL[k], "x": [round(float(v), 4) for v in SIG[k]]} for k in SIG}, ensure_ascii=False) + ";\n")
    fh.write(f"window.LP_SR = {SR}; window.LP_T0 = {T0};\n")
print("  wrote signals.js")

write("arctic_ah.wav", word_clip(UTT_AA, ("aa", 0), ("aa", 0)))      # "Ah"
write("arctic_see.wav", word_clip(UTT_SEE, ("s", 0), ("iy", 0)))     # "see"
write("white_noise.wav", rng.standard_normal(SR) * 0.3)
write("tone_200.wav", 0.5 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR))


# numbers the page quotes: the same least-squares (covariance-method) gains its demos compute from the rounded excerpts
def ls_gains(x, pmax=24):
    x = np.round(x, 4)
    ex = np.sum(x[pmax:] ** 2)
    out = [0.0]
    for p in range(1, pmax + 1):
        X = np.array([x[pmax - k:len(x) - k] for k in range(1, p + 1)]).T
        w = np.linalg.solve(X.T @ X + 1e-9 * np.eye(p), X.T @ x[pmax:])
        out.append(float(10 * np.log10(ex / np.sum((x[pmax:] - X @ w) ** 2))))
    return out


NUM["demo_gain_db"] = {k: [round(g, 2) for g in ls_gains(SIG[k])] for k in ("noise", "aa", "iy", "s")}
NUM["r1"] = {k: corr(SIG[k], 1) for k in ("noise", "aa", "iy", "s")}
NUM["one_weight_gain_db"] = {k: float(-10 * np.log10(1 - corr(SIG[k], 1) ** 2)) for k in ("noise", "aa", "iy", "s")}
sp_s = np.abs(np.fft.rfft(SIG["s"] * np.hanning(N))) ** 2
f_s = np.fft.rfftfreq(N, 1 / SR)
NUM["s_centroid_hz"] = float(np.sum(f_s * sp_s) / np.sum(sp_s))
cum = np.cumsum(sp_s) / np.sum(sp_s)
NUM["s_energy_band_hz"] = [float(f_s[np.searchsorted(cum, 0.1)]), float(f_s[np.searchsorted(cum, 0.9)])]

# ==================================================================== FIG 1: scatter grid
rows = [("noise", SLATE), ("aa", RED), ("s", GOLD)]
K_HALF = int(6 + np.argmin(r_aa[6:21]))                # most negative r of /ɑ/: half a cycle of its F1 ringing
NUM["K_half"] = K_HALF
lags = [1, 4, K_HALF, T0]
fig, axs = plt.subplots(len(rows), len(lags), figsize=(11, 8.2), gridspec_kw=dict(hspace=0.42, wspace=0.18))
NUM["r"] = {}
for i, (key, col) in enumerate(rows):
    x = SIG[key]
    for j, k in enumerate(lags):
        ax = axs[i, j]
        r = corr(x, k)
        NUM["r"][f"{key}_{k}"] = r
        ax.scatter(x[:-k], x[k:], s=4, color=col, alpha=0.35, rasterized=True)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.text(0.04, 0.93, f"$r = {r:+.2f}$", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D7DEE8"))
        if i == 0:
            ax.set_title(f"$k = {k}$" + (f"  (one pitch period of /ɑ/)" if k == T0 else f" sample{'s' if k > 1 else ''}") + "\n" + f"$= {k / SR * 1000:.2f}$ ms", fontsize=9.5)
        if j == 0:
            ax.set_ylabel(f"{LABEL[key]}\n\n" + r"$x[n+k]$", fontsize=9.5)
        if i == len(rows) - 1:
            ax.set_xlabel(r"$x[n]$")
save(fig, "01_scatter_grid.png")

# ==================================================================== FIG 2: correlation vs lag
fig, ax = plt.subplots(figsize=(11, 3.6))
K = np.arange(0, 241)
for key, col in [("noise", SLATE), ("aa", RED), ("s", GOLD)]:
    ax.plot(K / SR * 1000, [corr(SIG[key], k) for k in K], color=col, lw=1.8, label=LABEL[key])
ax.axhline(0, color="#AEB9CB", lw=1)
ax.axvline(T0 / SR * 1000, color=RED, ls=":", lw=1.2)
ax.text(T0 / SR * 1000 + 0.1, 0.85, f"one pitch period\n({T0} samples $=$ {T0 / SR * 1000:.1f} ms)", color=RED, fontsize=8.5)
ax.set_xlabel(r"lag $k$ (ms)"); ax.set_ylabel(r"correlation $r(k)$"); ax.set_ylim(-1, 1.05); ax.set_xlim(0, 15)
ax.legend(loc="lower right", ncol=3)
save(fig, "02_correlation_vs_lag.png")

# ==================================================================== FIG 3: what prediction leaves behind (teaser)
a, b = token(*AA)
seg = UTTS[UTT_AA]["y"][a:b]
pre = signal.lfilter([1, -0.97], [1], seg)
c = len(pre) // 2
w = lpc(pre[c - 320:c + 320], 14)
pred = signal.lfilter(np.concatenate([[0], w]), [1], pre)          # x̂[n] = Σ w_k x[n-k]
res = pre - pred
show = slice(c - 480, c + 480)
t = np.arange(960) / SR * 1000
gain = 10 * np.log10(np.var(pre[show]) / np.var(res[show]))
NUM["teaser_gain_db"] = float(gain)
fig, axs = plt.subplots(3, 1, figsize=(11, 5.4), sharex=True, gridspec_kw=dict(hspace=0.25))
m = np.abs(pre[show]).max()
axs[0].plot(t, pre[show] / m, color=SLATE, lw=1.1); axs[0].set_title("speech: /ɑ/ in “Ah” (CMU ARCTIC speaker ksp)", loc="left", fontsize=10)
axs[1].plot(t, pre[show] / m, color="#C9D2DE", lw=2.6, label="actual")
axs[1].plot(t, pred[show] / m, color=BLUE, lw=1.1, label="predicted from the previous 14 samples")
axs[1].legend(loc="upper right", ncol=2); axs[1].set_title("prediction", loc="left", fontsize=10)
axs[2].plot(t, res[show] / m, color=RED, lw=1.1)
axs[2].set_title(f"prediction error (what could not be predicted): {gain:.0f} dB weaker, and spiky once per pitch period", loc="left", fontsize=10)
for ax in axs: ax.set_yticks([]); ax.set_ylim(-1.15, 1.15)
axs[2].set_xlabel("ms")
save(fig, "03_prediction_error_teaser.png")

# ==================================================================== Yule (1927) and Walker (1931): sunspots
import statsmodels.api as sm  # noqa: E402

sun = sm.datasets.sunspots.load_pandas().data           # yearly Wolf sunspot numbers, 1700–2008 (NOAA)
yr_all, ss_all = sun.YEAR.values.astype(int), sun.SUNACTIVITY.values
sel = (yr_all >= 1749) & (yr_all <= 1924)                # the span Yule analysed (Wolfer's numbers)
years, ss = yr_all[sel], ss_all[sel]
xs = ss - ss.mean()
Xl = np.c_[xs[1:-1], xs[:-2]]; yl = xs[2:]
a_yule = np.linalg.lstsq(Xl, yl, rcond=None)[0]           # least squares, as Yule did
pred = Xl @ a_yule
shock = yl - pred
rho_y = np.sqrt(-a_yule[1]); per_y = 2 * np.pi / np.arccos(a_yule[0] / (2 * rho_y))
rs = lambda k: float(np.dot(xs[:-k], xs[k:]) / np.dot(xs, xs)) if k else 1.0
a1_only = rs(1)
NUM["yule"] = {"a1": float(a_yule[0]), "a2": float(a_yule[1]), "rho": float(rho_y), "period_years": float(per_y),
               "resid_var_frac": float(np.var(shock) / np.var(yl)), "ar1_resid_frac": float(1 - a1_only ** 2),
               "r1": rs(1), "r2": rs(2), "mean": float(ss.mean())}

# Fig 4: the series, Yule's one-year-ahead predictions, and the shocks
fig, axs = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, gridspec_kw=dict(hspace=0.18, height_ratios=[1.6, 1]))
axs[0].plot(years, ss, color="#AEB9CB", lw=3, label="observed sunspot number")
axs[0].plot(years[2:], pred + ss.mean(), color=BLUE, lw=1.3, label=rf"predicted from the two previous years: ${a_yule[0]:.2f}\,x_{{t-1}} {a_yule[1]:+.2f}\,x_{{t-2}}$")
axs[0].legend(loc="upper left"); axs[0].set_ylabel("yearly mean\nsunspot number")
axs[0].set_title("Yule (1927): each year's sunspot number predicted from the two years before", loc="left")
axs[1].bar(years[2:], shock, color=RED, width=0.8)
axs[1].set_ylabel("“shock”"); axs[1].set_xlabel("year")
axs[1].set_title(f"what the two-year rule could not predict: random disturbances ({100 * np.var(shock) / np.var(yl):.0f} % of the variance)", loc="left", fontsize=10)
save(fig, "04_yule_sunspots.png")

# Fig 5: Yule's lagged scatter plots
fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), gridspec_kw=dict(wspace=0.3))
for ax, k in zip(axs, [1, 2, 5]):
    ax.scatter(ss[:-k], ss[k:], s=10, color=GOLD, alpha=0.7)
    ax.set_xlabel(r"sunspots in year $t$"); ax.set_ylabel(f"sunspots in year $t + {k}$")
    ax.set_title(f"lag {k} year{'s' if k > 1 else ''}:  $r = {rs(k):+.2f}$", fontsize=10)
    ax.set_xlim(0, 160); ax.set_ylim(0, 160); ax.set_aspect("equal")
save(fig, "05_yule_scatter.png")

# Fig 6: Walker's correlogram check
K = np.arange(0, 41)
r_meas = np.array([rs(k) for k in K])
R2 = np.array([[1, rs(1)], [rs(1), 1]]); a_yw = np.linalg.solve(R2, [rs(1), rs(2)])
r_ar = np.zeros(len(K)); r_ar[0] = 1; r_ar[1] = a_yw[0] / (1 - a_yw[1])
for k in range(2, len(K)): r_ar[k] = a_yw[0] * r_ar[k - 1] + a_yw[1] * r_ar[k - 2]
NUM["walker"] = {"a1": float(a_yw[0]), "a2": float(a_yw[1])}
P9 = 9                                                        # Walker's generalisation: more past values
R9 = np.array([[rs(abs(i - j)) for j in range(P9)] for i in range(P9)])
a9 = np.linalg.solve(R9, [rs(k) for k in range(1, P9 + 1)])
r_ar9 = np.zeros(len(K)); r_ar9[:P9 + 1] = r_meas[:P9 + 1]
for k in range(P9 + 1, len(K)): r_ar9[k] = np.dot(a9, r_ar9[k - 1:k - P9 - 1:-1])
Xl9 = np.array([xs[t - P9:t][::-1] for t in range(P9, len(xs))]); y9 = xs[P9:]
NUM["walker"]["ar9_resid_frac"] = float(np.var(y9 - Xl9 @ a9) / np.var(y9))
rho2 = np.sqrt(-a_yw[1])                                       # pole radius of the 2-year model: its damping per year
pk = [k for k in range(2, len(K) - 1) if r_meas[k] > r_meas[k - 1] and r_meas[k] >= r_meas[k + 1] and r_meas[k] > 0]
NUM["walker"].update({"rho": float(rho2), "peaks": pk})
fig, ax = plt.subplots(figsize=(11, 4.6))
for (k1, yb, col, txt) in [(2, 1.12, BLUE, "lags the 2-year model was fitted to"), (P9, 1.22, RED, "lags the 9-year model was fitted to")]:
    ax.plot([0, k1], [yb, yb], color=col, lw=2.2, solid_capstyle="butt")
    ax.plot([0, 0], [yb - 0.03, yb + 0.03], color=col, lw=1.5); ax.plot([k1, k1], [yb - 0.03, yb + 0.03], color=col, lw=1.5)
    ax.text(k1 + 0.4, yb, txt, va="center", fontsize=8, color=col)
ax.text(40, 1.22, "beyond the brackets, the curves are pure predictions from the recursion", ha="right", va="center", fontsize=8, color=INK)
ax.axhline(0, color="#AEB9CB", lw=1)
kk = np.linspace(0, K[-1], 400)
ax.fill_between(kk, -rho2 ** kk, rho2 ** kk, color=BLUE, alpha=0.06, lw=0)
ax.plot(kk, rho2 ** kk, color=BLUE, lw=0.9, ls=":")
ax.plot(kk, -rho2 ** kk, color=BLUE, lw=0.9, ls=":")
ax.annotate(rf"2-year model's envelope $\pm\rho^k$, $\rho=\sqrt{{-a_2}}={rho2:.2f}$" "\n" rf"(by $k=11$ it is down to ${rho2 ** 11:.2f}$)",
            xy=(14, rho2 ** 14), xytext=(15.5, 0.62), fontsize=8.5, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
ax.plot(K, r_meas, "o-", color=GOLD, lw=1.4, ms=4, label=r"measured correlation of the sunspot numbers, $r(k)$", zorder=3)
ax.plot(K, r_ar, color=BLUE, lw=2.2,
        label=rf"2 past years: $r(k) = {a_yw[0]:.2f}\,r(k-1) {a_yw[1]:+.2f}\,r(k-2)$", zorder=4)
ax.plot(K, r_ar9, color=RED, lw=1.8, ls="--", label=r"9 past years (Walker): $r(k) = \sum_{i=1}^{9} a_i\, r(k-i)$", zorder=4)
for k in pk:
    ax.annotate(f"{k} yr", (k, r_meas[k]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=GOLD, weight="bold")
ax.set_xlabel(r"lag $k$ (years)"); ax.set_ylabel(r"$r(k)$"); ax.set_ylim(-0.5, 1.28); ax.set_yticks(np.arange(-0.4, 1.01, 0.2)); ax.set_xlim(-0.5, K[-1] + 0.5)
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17))
ax.set_title(r"Walker (1931): the correlogram obeys the model's own recursion, $r(k)=\sum_i a_i\,r(k-i)$", loc="left", pad=8)
save(fig, "06_walker_correlogram.png")

# Fig 7: two hypotheses (Schuster vs Yule) next to the real data
n = len(ss); t_ = np.arange(n)
rs_rng = np.random.default_rng(1927)
schuster = ss.mean() + 40 * np.sin(2 * np.pi * t_ / 11.1 + 1.0) + rs_rng.normal(0, 15, n)
yule_sim = np.zeros(n); sh = rs_rng.normal(0, np.std(shock), n)
for k in range(2, n): yule_sim[k] = a_yule[0] * yule_sim[k - 1] + a_yule[1] * yule_sim[k - 2] + sh[k]
yule_sim += ss.mean()
fig, axs = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True, gridspec_kw=dict(hspace=0.32))
axs[0].plot(years, schuster, color=SLATE, lw=1.3); axs[0].set_title("Hypothesis 1 (Schuster): a perfect 11-year wave + independent measurement errors", loc="left", fontsize=10)
axs[1].plot(years, yule_sim, color=BLUE, lw=1.3); axs[1].set_title("Hypothesis 2 (Yule): a damped pendulum pelted with random “peas” (simulated)", loc="left", fontsize=10)
axs[2].plot(years, ss, color=GOLD, lw=1.6); axs[2].set_title("The real sunspot record, 1749–1924", loc="left", fontsize=10)
for ax in axs: ax.set_ylabel("number")
axs[2].set_xlabel("year")
save(fig, "07_two_hypotheses.png")

# Fig 8: the whole sunspot record, its cycles, and how long each cycle lasted
from scipy.signal import argrelmin  # noqa: E402
xs3 = np.convolve(ss_all, np.ones(3) / 3, "same")
mins = []
for k in argrelmin(xs3, order=3)[0]:
    if mins and yr_all[k] - yr_all[mins[-1]] < 7:
        if xs3[k] < xs3[mins[-1]]: mins[-1] = k
    else:
        mins.append(k)
min_years = [int(yr_all[k]) for k in mins]
cyc_len = np.diff(min_years)
NUM["solar_cycles"] = {"minima": min_years, "lengths": cyc_len.tolist(), "mean": float(cyc_len.mean()), "min": int(cyc_len.min()), "max": int(cyc_len.max())}
fig = plt.figure(figsize=(11.5, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1], wspace=0.18)
ax = fig.add_subplot(gs[0])
ax.axvspan(1749, 1924, color=GOLD, alpha=0.12, lw=0)
ax.text(1752, 176, "the years Yule analysed\n(1749–1924)", ha="left", color="#8a6a10", fontsize=8.5)
ax.fill_between(yr_all, ss_all, color=GOLD, alpha=0.35, lw=0)
ax.plot(yr_all, ss_all, color="#a07a12", lw=1.2)
start1 = min_years.index(1755)                            # solar cycle 1 begins at the 1755 minimum
for n, (m0, m1) in enumerate(zip(min_years[start1:-1], min_years[start1 + 1:]), start=1):
    seg = (yr_all >= m0) & (yr_all < m1)
    pk = yr_all[seg][np.argmax(ss_all[seg])]
    ax.text(pk, ss_all[yr_all == pk][0] + 4, str(n), ha="center", fontsize=7.5, color=INK)
for yv, lab, yy in [(1843, "1843 Schwabe", 209), (1859, "1859 Carrington", 200), (1904, "1904 Walker → India", 209)]:
    ax.axvline(yv, color=SLATE, ls=":", lw=1)
    ax.text(yv + 1.5, yy, lab, fontsize=8, color=INK, va="center")
ax.set_xlim(1700, 2010); ax.set_ylim(0, 216); ax.set_xlabel("year"); ax.set_ylabel("sunspot number\n(yearly mean of daily values)")
ax.set_title("Three centuries of sunspot counts (small numbers = solar cycle number)", loc="left", fontsize=10.5)
ax2 = fig.add_subplot(gs[1])
bins = np.arange(8.5, 15.5, 1)
ax2.hist(cyc_len, bins=bins, color=GOLD, edgecolor="white")
ax2.set_xlabel("cycle length (years)"); ax2.set_ylabel("number of cycles")
ax2.set_title(f"not a clock: {cyc_len.min()}–{cyc_len.max()} years,\nmean {cyc_len.mean():.1f}", loc="left", fontsize=10)
save(fig, "08_sunspot_record.png")

# data for the page's "pelt the pendulum" demo
with open(os.path.join(HERE, "signals.js"), "a", encoding="utf8") as fh:
    fh.write("window.LP_SUNSPOTS = " + json.dumps({"years": years.tolist(), "x": [round(float(v), 1) for v in ss],
             "a": [round(float(a_yule[0]), 4), round(float(a_yule[1]), 4)], "shock_sd": round(float(np.std(shock)), 2)}) + ";\n")

with open(os.path.join(HERE, "numbers.json"), "w", encoding="utf8") as fh:
    json.dump(NUM, fh, indent=1)
print(json.dumps(NUM, indent=1))
