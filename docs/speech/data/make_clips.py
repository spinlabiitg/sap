"""
Builds the ten example clips used by vocal_tract.html, and
recomputes every number those pages quote about them (vowel formants, zero-crossing rate, spectral
centroid, plus the per-token sanity checks).

Provenance
----------
All ten clips are single words cut from the CMU ARCTIC database, speaker "ksp" (Indian-English male,
16 kHz), Carnegie Mellon University (J. Kominek and A. W. Black), http://www.festvox.org/cmu_arctic/ .
Licence: free for any use; the CMU copyright notice must be kept, authors credited, and modifications
marked.  MODIFICATIONS: each clip is a word excerpt (word span from the corpus' automatic Festival
labels, +/- 40 ms padding), with 10 ms raised-cosine fades, DC removal and peak normalisation to 0.9;
written as 16 kHz, 16-bit mono WAV.  Nothing else (no filtering, no resampling, no time-stretching).

These replace earlier text-to-speech clips.  ksp says none of the classic h-V-d words (heed, hod,
who'd) nor pea/chip, so the closest real words carrying the same target phone were used:
    heed -> heat, hod -> hot, who'd -> who, pea -> pay, chip -> chips  (had, heard, sea, moon, run kept).

Word boundaries are not hand-labelled in ARCTIC.  The search mode below reads them from the Festival
.utt files (SylStructure relation: word -> syllables -> segments), the same parser as
the (since removed) audio-features page generator (arctic_words); the chosen tokens are then hard-coded in CLIPS.

    python make_clips.py              # write the ten WAVs, print checks + EDA numbers
    python make_clips.py --search who # list every token of a word (utterance, times, phones, F1-F3)

Requires numpy, scipy, soundfile and the corpus (path in ../common/arctic.py, or set ARCTIC_KSP).
"""
import os
import re
import sys

import numpy as np
import soundfile as sf
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import arctic  # noqa: E402

SR = arctic.SR
PAD, FADE, PEAK = 0.040, 0.010, 0.9

# file stem, word, ARCTIC utterance, word start (s), word end (s), target phone (ARPAbet), class
CLIPS = [
    ("vowel_heat_i",         "heat",  "arctic_a0217", 1.395, 1.755, "iy", "vowel"),
    ("vowel_had_ae",         "had",   "arctic_b0330", 0.705, 1.100, "ae", "vowel"),
    ("vowel_hot_a",          "hot",   "arctic_a0091", 0.915, 1.365, "aa", "vowel"),
    ("vowel_who_u",          "who",   "arctic_a0528", 0.480, 0.750, "uw", "vowel"),
    ("vowel_heard_er",       "heard", "arctic_b0095", 0.340, 0.620, "er", "vowel"),
    ("cons_sea_fricative",   "sea",   "arctic_a0438", 0.315, 0.740, "s",  "fricative"),
    ("cons_pay_stop",        "pay",   "arctic_a0272", 1.560, 1.950, "p",  "stop"),
    ("cons_moon_nasal",      "moon",  "arctic_b0163", 0.245, 0.565, "m",  "nasal"),
    ("cons_chips_affricate", "chips", "arctic_b0329", 1.335, 1.705, "ch", "affricate"),
    ("cons_run_approximant", "run",   "arctic_b0467", 0.480, 0.765, "r",  "approx."),
]


# ----------------------------------------------------------------------------------------------
# word boundaries from the Festival .utt file (originally from the since-removed audio-features generator)
def arctic_words(utt):
    """[(start_s, end_s, word)] read from the Festival .utt file."""
    lines = open(os.path.join(arctic.ROOT, "festival", "utts", utt + ".utt"), encoding="latin-1").read().splitlines()
    i, items = lines.index("Stream_Items") + 1, {}
    while lines[i] != "End_of_Stream_Items":
        idx, rest = lines[i].split(" ", 1)
        items[int(idx)] = dict(t.strip().split(" ", 1) for t in rest.split(" ; ") if " " in t.strip())
        i += 1
    j, nodes = lines.index('Relation SylStructure ; "(" ")" ; ') + 1, {}
    while lines[j] != "End_of_Relation":
        nid, item, up, down, nxt, prv = map(int, lines[j].split())
        nodes[nid] = (item, down, nxt, prv)
        j += 1

    def children(n):
        out, c = [], nodes[n][1]
        while c:
            out.append(c)
            c = nodes[c][2]
        return out

    def leaves(n):
        ch = children(n)
        return [n] if not ch else [lf for c in ch for lf in leaves(c)]

    kids = {c for n in nodes for c in children(n)}
    tops = [n for n in nodes if n not in kids]
    r, seq = [n for n in tops if nodes[n][3] not in tops][0], []
    while r:
        seq.append(r)
        r = nodes[r][2]
    ph = arctic.phones(utt)
    ends = np.array([e for _, e, _ in ph])
    out = []
    for w in seq:
        segs = [items[nodes[s][0]] for s in leaves(w)]
        if any("end" not in s for s in segs):          # punctuation tokens have no segments
            continue
        k = int(np.argmin(np.abs(ends - float(segs[0]["end"]))))
        out.append((ph[k][0], float(segs[-1]["end"]), items[nodes[w][0]]["name"]))
    return out


# ----------------------------------------------------------------------------------------------
# analysis helpers
def lpc_formants(frame, order=18):
    """Formant candidates (Hz) of one frame: pre-emphasis, Hamming, autocorrelation LPC, root picking."""
    x = lfilter([1, -0.97], 1, frame) * np.hamming(len(frame))
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + order + 1]
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    roots = np.roots(np.r_[1, a])
    roots = roots[np.imag(roots) > 0]
    f = np.angle(roots) * SR / (2 * np.pi)
    bw = -np.log(np.abs(roots)) * SR / np.pi
    return sorted(fi for fi, b in zip(f, bw) if fi > 90 and b < 400)


def vowel_formants(x, t0, t1):
    """Median F1-F3 over 25 ms frames (5 ms hop) in the middle 50 % of the labelled vowel."""
    a, b = t0 + 0.25 * (t1 - t0), t0 + 0.75 * (t1 - t0)
    rows = []
    for t in np.arange(a, max(a + 1e-3, b - 0.025), 0.005):
        fr = lpc_formants(x[int(t * SR):int(t * SR) + int(0.025 * SR)])
        if len(fr) >= 3:
            rows.append(fr[:3])
    return np.median(np.array(rows), 0)


def frames(y, n=int(0.025 * SR), hop=int(0.010 * SR)):
    return [y[i:i + n] for i in range(0, len(y) - n + 1, hop)]


def eda_measures(y):
    """Clip-level ZCR (crossings/s) and spectral centroid (Hz): means over 25 ms / 10 ms frames whose
    energy is within 40 dB of the loudest frame (drops the padding and the silent stop closure)."""
    fr = frames(y)
    en = np.array([10 * np.log10(np.mean(f ** 2) + 1e-12) for f in fr])
    keep = [f for f, e in zip(fr, en) if e > en.max() - 40]
    freqs = np.fft.rfftfreq(512, 1 / SR)
    zcr, cen = [], []
    for f in keep:
        zcr.append(np.count_nonzero(np.diff(np.signbit(f))) / (len(f) / SR))
        X = np.abs(np.fft.rfft(f * np.hanning(len(f)), 512))
        cen.append((freqs * X).sum() / (X.sum() + 1e-12))
    return float(np.mean(zcr)), float(np.mean(cen))


def band_centroid(x, t0, t1):
    seg = x[int(t0 * SR):int(t1 * SR)]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 4096)) ** 2
    f = np.fft.rfftfreq(4096, 1 / SR)
    return float((f * X).sum() / X.sum()), float(X[f < 400].sum() / X.sum())


def stop_check(x, t0, t1):
    """Closure (silent stretch), burst instant and voicing onset for a word-initial voiceless stop.
    1 ms hop, 5 ms window energy contour; closure = longest run of frames > 35 dB below the vowel
    peak; burst = end of that run; voicing onset = first later frame within 20 dB of the peak with
    at most 3 zero crossings in 5 ms (i.e. dominated by low-frequency voicing, not aspiration noise).
    VOT = voicing onset - burst."""
    hop, win = int(0.001 * SR), int(0.005 * SR)
    seg = x[int(t0 * SR):int(t1 * SR)]
    e = np.array([10 * np.log10(np.mean(seg[i:i + win] ** 2) + 1e-12) for i in range(0, len(seg) - win, hop)])
    peak = e.max()
    quiet = np.r_[e[:int(np.argmax(e))] < peak - 35, False]   # near-silent frames before the vowel peak
    runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start, i))
            start = None
    c0, c1 = max(runs, key=lambda r: r[1] - r[0])            # longest silent run = the closure
    closure, burst = (c1 - c0) / 1000, t0 + c1 / 1000
    vo = None
    for i in range(c1, len(e)):
        zc = np.count_nonzero(np.diff(np.signbit(seg[i * hop:i * hop + win])))
        if e[i] > peak - 20 and zc <= 3:                       # loud and low-frequency = voiced vowel
            vo = t0 + i / 1000
            break
    return closure, burst, vo


def search(word):
    for u in arctic.utterances():
        if word not in re.findall(r"[a-z']+", arctic.text(u).lower()):
            continue
        x, ph = arctic.read_wav(u), arctic.phones(u)
        for s, e, w in arctic_words(u):
            if w.lower() != word:
                continue
            segs = [(a, b, p) for a, b, p in ph if a >= s - 1e-6 and b <= e + 1e-6]
            fm = ["%s:%s" % (p, vowel_formants(x, a, b).astype(int).tolist()) for a, b, p in segs if p in arctic.VOWELS]
            print(u, "%.3f-%.3f" % (s, e), " ".join(p for _, _, p in segs), *fm, "|", arctic.text(u))


# ----------------------------------------------------------------------------------------------
def main():
    eda = []
    print("%-22s %-6s %-13s %-13s %6s %6s  check" % ("file", "word", "utterance", "times (s)", "dur", "peak"))
    for stem, word, utt, s, e, target, cls in CLIPS:
        x = arctic.read_wav(utt)
        ph = [(a, b, p) for a, b, p in arctic.phones(utt) if a >= s - 1e-6 and b <= e + 1e-6]
        assert any(p == target for _, _, p in ph), (stem, ph)
        a, b = int((s - PAD) * SR), int((e + PAD) * SR)
        y = x[a:b] - np.mean(x[a:b])
        n = int(FADE * SR)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
        y = PEAK * y / np.max(np.abs(y))
        sf.write(os.path.join(HERE, stem + ".wav"), y, SR, subtype="PCM_16")
        dur = len(y) / SR
        assert 0.25 <= dur <= 0.8 and np.max(np.abs(y)) < 0.999

        t0, t1 = [(p0, p1) for p0, p1, p in ph if p == target][0]
        if cls == "vowel":
            f1, f2, f3 = vowel_formants(x, t0, t1)
            chk = "F1=%d F2=%d F3=%d Hz (vowel %d ms)" % (f1, f2, f3, (t1 - t0) * 1000)
        elif cls in ("fricative", "affricate"):
            fr0 = t0 + (0.4 * (t1 - t0) if cls == "affricate" else 0.0)   # frication part of /tS/
            c, lo = band_centroid(x, fr0, t1)
            chk = "frication centroid=%d Hz, energy<400 Hz=%.0f%% (%d ms)" % (c, 100 * lo, (t1 - t0) * 1000)
        elif cls == "stop":
            clo, burst, vo = stop_check(x, s - 0.01, e)
            chk = "closure=%d ms, burst at %.3f s, VOT=%s" % (clo * 1000, burst, "%d ms" % ((vo - burst) * 1000) if vo else "n/a")
        elif cls == "nasal":
            c, lo = band_centroid(x, t0, t0 + 0.08)
            chk = "murmur centroid=%d Hz, energy<400 Hz=%.0f%%" % (c, 100 * lo)
        else:
            f1, f2, f3 = vowel_formants(x, t0, t1)
            chk = "/r/ F1=%d F2=%d F3=%d Hz" % (f1, f2, f3)
        print("%-22s %-6s %-13s %.3f-%.3f %5.0fms %6.2f  %s" % (stem, word, utt, s, e, dur * 1000, np.max(np.abs(y)), chk))
        zcr, cen = eda_measures(y)
        eda.append((word, cls, zcr, cen, stem))

    print("\nEDA (word-level ZCR and centroid of each clip)")
    for name, k in (("ZCR", 2), ("CENT", 3)):
        print("var %s = [" % name)
        for r in sorted(eda, key=lambda r: -r[k]):
            print('  { label: "%s", manner: "%s", value: %d },' % (r[0], "vowel" if r[1] == "vowel" else r[1], round(r[k])))
        print("];")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--search":
        search(sys.argv[2].lower())
    else:
        main()
