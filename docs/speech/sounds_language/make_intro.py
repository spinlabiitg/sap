"""
Data, audio and the one figure for the OPENING HALF of ../sounds_language.html (sections 01-04: speech as a programming
language, the phoneme as a token, same token or not?, phonemes as bit fields).

Run on its own (merges its numbers into numbers.json):   python make_intro.py
or through make_figures.py, which calls main(NUM, ph, picks) before writing numbers.json.

Outputs
  sentence.js       one ksp sentence (int16 base64) + automatic phone labels + words, for the "sentence, sound by sound" demo
  intro_data.js     minimal pairs, one-phoneme-or-two quiz, IPA chart membership, PHOIBLE feature vectors
  audio/*.wav       minimal-pair words (16 kHz, 16-bit, 10 ms fades, peak 0.9)
  numbers.json      keys starting with "i_"

Speech: CMU ARCTIC speaker ksp (Indian-English male, 16 kHz; Kominek & Black 2004) via ../common/arctic.py. Its phone
labels are automatic (EHMM, General-American CMUdict phone set) and were not hand-corrected: every token used is
measured acoustically here, and implausible measurements are discarded (rules below).
PHOIBLE 2.0 (CC BY 4.0): feature vectors and the English / Hindi inventories (same median-inventory rule as make_figures).
"""
import base64
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.linalg import solve_toeplitz  # noqa: E402
from scipy.signal import butter, lfilter, sosfiltfilt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import arctic  # noqa: E402

FIG = os.path.join(HERE, "fig")
AUD = os.path.join(HERE, "audio")
os.makedirs(FIG, exist_ok=True)
os.makedirs(AUD, exist_ok=True)
DB = os.path.join(HERE, "..", "..", "..", "..", "..", "research", "dBase")
if not os.path.isdir(DB):
    DB = r"C:/Users/IITG/Documents/tommy_assistant/research/dBase"
SR = arctic.SR
MS = SR // 1000

INK, BLUE, RED, GOLD, SLATE, TEAL = "#10233F", "#14508C", "#C1272D", "#C9971B", "#5B7083", "#1D7A8C"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#AEB9CB", "axes.labelcolor": INK, "xtick.color": "#5C6B80", "ytick.color": "#5C6B80",
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.color": "#E7ECF2", "grid.linewidth": 0.8,
    "mathtext.fontset": "cm",
    "axes.axisbelow": True,
})
CLASS = {}
for p in arctic.VOWELS:
    CLASS[p] = "vowel"
for p in arctic.STOPS:
    CLASS[p] = "stop"
for p in arctic.FRICATIVES:
    CLASS[p] = "fricative"
for p in arctic.NASALS:
    CLASS[p] = "nasal"
for p in ("ch", "jh"):
    CLASS[p] = "affricate"
for p in ("l", "r", "w", "y"):
    CLASS[p] = "approximant"


# ====================================================================== ksp words (Festival .utt: word -> syllable -> segment)
def utt_words(utt):
    """[(word, [(phone, start_s, end_s, syllable_stress), ...]), ...] for one ksp utterance.
    Parser idea from the since-removed audio-features generator (arctic_words), extended with syllable stress."""
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
        segs = []
        for syl in children(w):
            st = int(items[nodes[syl][0]].get("stress", "0"))
            for s in children(syl):
                it = items[nodes[s][0]]
                if "end" not in it:
                    continue
                k = int(np.argmin(np.abs(ends - float(it["end"]))))
                segs.append((ph[k][2], ph[k][0], ph[k][1], st))
        if segs:
            out.append((items[nodes[w][0]]["name"], segs))
    return out


# ====================================================================== acoustic helpers
TUNE = {"burst_db": float(os.environ.get("VOT_BURST_DB", 12)), "lf_db": float(os.environ.get("VOT_LF_DB", 6)),
        "per": float(os.environ.get("VOT_PER", 0.85)), "lead": float(os.environ.get("VOT_LEAD", 0.02)),
        "abs_db": float(os.environ.get("VOT_ABS_DB", 25))}
HP = butter(4, 1000, "highpass", fs=SR, output="sos")
LP = butter(4, 350, "lowpass", fs=SR, output="sos")


def db(y):
    return 10 * np.log10(np.mean(y ** 2) + 1e-12) if len(y) else -120.0


def periodicity(y, i, n=20 * MS):
    """max normalised autocorrelation of y[i:i+n] at lags 3-13 ms (F0 75-330 Hz)."""
    seg = y[i:i + n]
    if len(seg) < n:
        return 0.0
    seg = seg - seg.mean()
    best = 0.0
    for lag in range(int(SR / 330), int(SR / 75)):
        a, b = seg[:-lag], seg[lag:]
        best = max(best, float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-20)))
    return best


def vot(xh, xl, s, e, ve):
    """Burst and voicing onset for a voiceless stop labelled [s, e] and followed by a vowel that ends at ve.
    burst  = first frame, searching from 20 ms after the label start (inside the closure), where the high-passed
             (>1 kHz) energy in 2 ms frames rises 12 dB above the quietest frame of the preceding 2-12 ms
             and is within 25 dB of the loudest frame of the search window (ignores faint noise in long closures).
    voicing onset = first time after burst + 3 ms where the low-passed (<350 Hz) signal is within 6 dB of the
             vowel's maximum AND periodic (autocorrelation > 0.85 over 20 ms) and stays periodic (> 0.6) for 10 ms.
    The thresholds (TUNE) were set by checking the detector against a hand-labelled corpus of American English.
    Returns (burst_s, onset_s, rise_dB) or None."""
    i0, i1 = int((s + TUNE["lead"]) * SR), int(min(e + 0.03, ve - 0.02) * SR)
    if i1 - i0 < 10 * MS:
        return None
    ts = np.arange(i0, i1, MS // 2)
    E = np.array([db(xh[max(0, i - MS):i + MS]) for i in ts])
    Eb = np.array([min(db(xh[max(0, j - MS):j + MS]) for j in range(i - 12 * MS, i - 2 * MS, 2 * MS)) for i in ts])
    rise = E - Eb
    above = np.nonzero((rise >= TUNE["burst_db"]) & (E >= E.max() - TUNE["abs_db"]))[0]
    if not len(above):
        return None
    k = int(above[0])
    burst = ts[k] - MS
    vi0, vi1 = int(e * SR), int(ve * SR)
    ref = max(db(xl[j:j + 5 * MS]) for j in range(vi0, max(vi0 + 1, vi1 - 5 * MS), MS))
    j, lim = burst + 3 * MS, min(vi1 - 15 * MS, burst + 160 * MS)
    while j < lim:
        if db(xl[j:j + 5 * MS]) > ref - TUNE["lf_db"] and periodicity(xl, j) > TUNE["per"]:
            if all(periodicity(xl, jj) > 0.6 for jj in range(j + 2 * MS, j + 11 * MS, 3 * MS)):
                return burst / SR, j / SR, float(rise[k])
        j += MS
    return None


def plausible(m, e):
    """drop measurements that disagree with the label or are physically implausible."""
    if m is None:
        return False
    b, v, _ = m
    return 0 <= (v - b) * 1000 <= 120 and abs(v - e) <= 0.04


def lpc_formants(frame, order=18):
    x = lfilter([1, -0.97], 1, frame) * np.hamming(len(frame))
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + order + 1]
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    roots = np.roots(np.r_[1, a])
    roots = roots[np.imag(roots) > 0]
    f = np.angle(roots) * SR / (2 * np.pi)
    bw = -np.log(np.abs(roots)) * SR / np.pi
    return sorted(fi for fi, b in zip(f, bw) if fi > 90 and b < 400)


def vowel_formants(x, t0, t1):
    """median F1, F2 (Hz) over 25 ms frames in the middle half of a labelled vowel (as ../data/make_clips.py)."""
    a, b = t0 + 0.25 * (t1 - t0), t0 + 0.75 * (t1 - t0)
    rows = []
    for t in np.arange(a, max(a + 1e-3, b - 0.025), 0.005):
        fr = lpc_formants(x[int(t * SR):int(t * SR) + int(0.025 * SR)])
        if len(fr) >= 2:
            rows.append(fr[:2])
    return np.median(np.array(rows), 0)


def centroid(x, t0, t1):
    seg = x[int(t0 * SR):int(t1 * SR)]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 4096)) ** 2
    f = np.fft.rfftfreq(4096, 1 / SR)
    return float((f * X).sum() / X.sum())


def voiced_fraction(xl, t0, t1):
    """share of 5 ms steps inside [t0, t1] whose 20 ms window (centred) is periodic (> 0.6)."""
    js = range(int(t0 * SR) - 10 * MS, int(t1 * SR) - 10 * MS, 5 * MS)
    v = [periodicity(xl, j) > 0.6 for j in js]
    return float(np.mean(v)) if v else 0.0


def write_clip(name, x, t0, t1, pad=0.04, fade=0.010, peak=0.9):
    a, b = max(0, int((t0 - pad) * SR)), min(len(x), int((t1 + pad) * SR))
    y = x[a:b] - np.mean(x[a:b])
    n = int(fade * SR)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    y = peak * y / np.max(np.abs(y))
    sf.write(os.path.join(AUD, name), y, SR, subtype="PCM_16")
    return len(y) / SR


# ====================================================================== PHOIBLE helpers
PREF = ["ph", "ea", "ra", "upsid", "spa", "saphon", "aa", "er"]


def median_pick(ph, gc):
    """the 'median inventory' of make_figures.py for one (language-level) Glottocode."""
    d = ph[ph.Glottocode == gc]
    t = d.groupby("InventoryID").agg(src=("Source", "first"), name=("LanguageName", "first"), dial=("SpecificDialect", "first"),
                                     C=("SegmentClass", lambda s: int((s == "consonant").sum())),
                                     V=("SegmentClass", lambda s: int((s == "vowel").sum())))
    t["CV"] = t.C + t.V
    t["pref"] = t.src.map(lambda s: PREF.index(s) if s in PREF else 99)
    t["dist"] = (t.CV - t.CV.median()).abs()
    t = t.reset_index().sort_values(["dist", "pref", "InventoryID"])
    return int(t.InventoryID.iloc[0]), t


def inventory(ph, iid):
    d = ph[ph.InventoryID == iid]
    return {p: bool(m) for p, m in zip(d.Phoneme, d.Marginal.astype(str).eq("True"))}


# ====================================================================== main
def main(NUM=None, ph=None, picks=None):
    standalone = NUM is None
    if standalone:
        NUM = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf8"))
    NUM = {k: v for k, v in NUM.items() if not k.startswith("i_")}
    if ph is None:
        ph = pd.read_csv(os.path.join(DB, "PHOIBLE", "phoible.csv"), low_memory=False, keep_default_na=False, na_values=["NA"])
    I = {}  # numbers for this half, prefixed "i_" at the end

    print("parsing ksp .utt files")
    allw = {u: utt_words(u) for u in arctic.utterances()}
    NUM_UTT = len(allw)

    # ------------------------------------------------------------------ minimal pairs
    print("minimal pairs")
    # (word, utterance, index of the word in the utterance); chosen with the search in ../../../ (see README of this
    # function: every ksp word type whose most common pronunciation differs from another word's in one phone)
    PAIRS = [
        ("p · b", "voicing", [("pair", "arctic_a0369", 5), ("bear", "arctic_a0179", 5)], "stop"),
        ("k · ɡ", "voicing", [("came", "arctic_a0420", 2), ("game", "arctic_a0311", 4)], "stop"),
        ("s · ʃ", "place (alveolar vs post-alveolar)", [("see", "arctic_a0007", 5), ("she", "arctic_b0421", 0)], "fric"),
        ("f · v", "voicing", [("ferry", "arctic_b0269", 7), ("very", "arctic_b0527", 1)], "fv"),
        ("i · ɪ", "vowel quality (and length)", [("steel", "arctic_b0151", 2), ("still", "arctic_b0393", 2)], "vowel"),
        ("ɛ · æ", "vowel height", [("men", "arctic_a0528", 0), ("man", "arctic_b0207", 5)], "vowel"),
        ("l · ɹ", "lateral vs rhotic", [("light", "arctic_b0118", 3), ("right", "arctic_b0295", 1)], "liquid"),
        ("n · ŋ", "place of the nasal", [("ran", "arctic_b0438", 1), ("rang", "arctic_b0426", 6)], "nasal"),
    ]
    # word-type inventory and one-phone neighbours, for the numbers quoted on the page
    prons = defaultdict(Counter)
    for u, ws in allw.items():
        for w, segs in ws:
            prons[w.lower()][tuple(s[0] for s in segs)] += 1
    wp = {w: c.most_common(1)[0][0] for w, c in prons.items()}
    idx = defaultdict(set)
    for w, p in wp.items():
        for i in range(len(p)):
            idx[(len(p), i, p[:i] + p[i + 1:])].add((w, p[i]))
    npairs = sum(1 for v in idx.values() for a in v for b in v if a < b and a[1] != b[1])
    I["mp_n_types"] = len(wp)
    I["mp_n_pairs"] = int(npairs)
    print(f"  {len(wp)} word types, {npairs} one-phone minimal pairs")
    MP = []
    for contrast, what, words, kind in PAIRS:
        card = dict(contrast=contrast, what=what, words=[])
        for word, utt, wi in words:
            w, segs = allw[utt][wi]
            assert w.lower() == word, (utt, wi, w)
            x = arctic.read_wav(utt)
            xl = sosfiltfilt(LP, x)
            fn = f"mp_{word}.wav"
            dur = write_clip(fn, x, segs[0][1], segs[-1][2])
            seq = [s[0] for s in segs]
            vows = [s for s in segs if s[0] in arctic.VOWELS]
            meas = ""
            if kind == "stop":
                s0, e0 = segs[0][1], segs[0][2]
                xh = sosfiltfilt(HP, x)
                m = vot(xh, xl, s0, e0, segs[1][2])
                vf = voiced_fraction(xl, s0, e0 - 0.01)
                if word in ("pair", "came"):
                    meas = f"VOT {1000 * (m[1] - m[0]):.0f} ms" if plausible(m, e0) else "VOT n/a"
                else:
                    meas = f"voicing through {100 * vf:.0f}% of the closure"
                print(f"  {word}: VOT {1000 * (m[1] - m[0]) if m else -1:.0f} ms, closure voiced {100 * vf:.0f}%")
            elif kind == "fric":
                f = segs[0]
                c = centroid(x, f[1] + 0.25 * (f[2] - f[1]), f[2] - 0.25 * (f[2] - f[1]))
                meas = f"noise centroid {c / 1000:.1f} kHz"
            elif kind == "fv":
                f = segs[0]
                q = 0.25 * (f[2] - f[1])
                vf = voiced_fraction(xl, f[1] + q, f[2] - q)
                c = centroid(x, f[1] + 0.25 * (f[2] - f[1]), f[2] - 0.25 * (f[2] - f[1]))
                meas = f"/{arctic.IPA[f[0]]}/ {100 * vf:.0f}% voiced, centroid {c / 1000:.1f} kHz"
            elif kind == "vowel":
                v = vows[0]
                F1, F2 = vowel_formants(x, v[1], v[2])
                meas = f"F1 {F1:.0f}, F2 {F2:.0f} Hz, {1000 * (v[2] - v[1]):.0f} ms"
            elif kind == "liquid":
                f = segs[0]
                meas = f"/{arctic.IPA[f[0]]}/ label {1000 * (f[2] - f[1]):.0f} ms"
            elif kind == "nasal":
                f = segs[-1]
                meas = f"nasal {1000 * (f[2] - f[1]):.0f} ms"
            print(f"  {word:6s} {utt} {' '.join(seq):14s} {dur * 1000:4.0f} ms clip  {meas}")
            card["words"].append(dict(w=word, utt=utt.replace("arctic_", ""), ipa="/" + "".join(arctic.IPA[p] for p in seq) + "/",
                                      file=fn, m=meas, text=arctic.text(utt)))
        MP.append(card)
    I["mp_n_cards"] = len(MP)

    # ------------------------------------------------------------------ the demo sentence
    print("demo sentence")
    LEX = "arctic_a0480"   # "Tom Spink has a harpoon."
    x = arctic.read_wav(LEX)
    lph = arctic.phones(LEX)
    speech = [p for p in lph if p[2] not in ("pau", "ssil")]
    t0, t1 = max(0.0, speech[0][0] - 0.12), min(len(x) / SR, speech[-1][1] + 0.12)
    y = x[int(t0 * SR):int(t1 * SR)]
    y = 0.9 * (y - y.mean()) / np.max(np.abs(y - y.mean()))
    b64 = base64.b64encode((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()).decode("ascii")
    words = utt_words(LEX)
    word_of = {}
    W = []
    for wi, (w, segs) in enumerate(words):
        W.append([round(segs[0][1] - t0, 3), round(segs[-1][2] - t0, 3), w])
        for s in segs:
            word_of[round(s[1], 3)] = wi
    P = []
    for s, e, p in lph:
        if e <= t0 or s >= t1:
            continue
        s2, e2 = max(s, t0) - t0, min(e, t1) - t0
        if p in ("pau", "ssil"):
            P.append([round(s2, 3), round(e2, 3), p, "", "pause", -1])
        else:
            P.append([round(s2, 3), round(e2, 3), p, arctic.IPA[p], CLASS[p], word_of.get(round(s, 3), -1)])
    data = {"sr": SR, "utt": LEX, "text": arctic.text(LEX), "b64": b64, "phones": P, "words": W}
    with open(os.path.join(HERE, "sentence.js"), "w", encoding="utf8") as fh:
        fh.write("// CMU ARCTIC ksp, " + LEX + " (Indian-English male, 16 kHz; Kominek & Black 2004), trimmed, peak-normalised to 0.9,\n")
        fh.write("// int16 little-endian, base64. Automatic phone labels (start s, end s, ARPAbet, IPA, class, word index) and words\n")
        fh.write("// from the Festival .utt file; not hand-corrected. Made by make_intro.py.\n")
        fh.write("window.LEX_SENT = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    kb = os.path.getsize(os.path.join(HERE, "sentence.js")) / 1024
    I["lex_text"] = arctic.text(LEX)
    I["lex_utt"] = LEX.replace("arctic_", "")
    I["lex_dur"] = round(len(y) / SR, 2)
    I["lex_nphones"] = sum(1 for p in P if p[4] != "pause")
    I["lex_nwords"] = len(W)
    I["lex_kb"] = round(kb)
    print(f"  {arctic.text(LEX)} {len(y) / SR:.2f} s, {I['lex_nphones']} phones, {len(W)} words, sentence.js {kb:.0f} KB")
    for p in P:
        print("   ", p[:5])

    # ------------------------------------------------------------------ PHOIBLE: English and Hindi
    print("PHOIBLE: English / Hindi inventories")
    LANG = {}
    for key, gc, name in [("eng", "stan1293", "English"), ("hind", "hind1269", "Hindi")]:
        iid, table = median_pick(ph, gc)
        if picks is not None:
            assert picks[gc] == iid, (gc, picks[gc], iid)
        if key == "eng":
            # the median rule lands on a Liverpool inventory; for teaching, compare with General American
            # (PHOIBLE 2175, "Western and Mid-Western US"). All nine English inventories give the same quiz answers.
            iid = 2175
        r = table[table.InventoryID == iid].iloc[0]
        LANG[key] = dict(id=iid, src=r.src, dial=r.dial if isinstance(r.dial, str) else "", name=name,
                         inv=inventory(ph, iid), all={int(i): inventory(ph, int(i)) for i in table.InventoryID})
        I[f"{key}_inv_id"] = iid
        I[f"{key}_inv_src"] = r.src
        I[f"{key}_inv_dial"] = LANG[key]["dial"] or "no dialect given"
        I[f"{key}_inv_n"] = len(table)
        print(f"  {name}: median inventory {iid} ({r.src}, {LANG[key]['dial']}), {len(table)} inventories")

    def present(inv, syms):
        """'yes' / 'marginal' / '' for the best match of any symbol in syms."""
        hit = [inv[s] for s in syms if s in inv]
        if not hit:
            return ""
        return "yes" if not all(hit) else "marginal"

    QPAIRS = [
        dict(a="p", b="pʰ", what="plain vs aspirated", A=["p"], B=["pʰ"],
             en=("<em>pin</em> [pʰɪn] and <em>spin</em> [spɪn]: the puff of air depends on position, never on the word.",),
             hi=("<span lang=\"hi\">पल</span> /pəl/ &lsquo;moment&rsquo; vs <span lang=\"hi\">फल</span> /pʰəl/ &lsquo;fruit&rsquo;.",)),
        dict(a="t̪", b="ʈ", what="dental vs retroflex", A=["t̪", "t̪ʰ", "t̻", "t̻ʰ"], B=["ʈ", "ʈʰ"],
             en=("English has one /t/ at the gum ridge (alveolar); Indian-English speakers often say it retroflex, and <em>ten</em> is still <em>ten</em>.",),
             hi=("<span lang=\"hi\">ताल</span> /t̪aːl/ &lsquo;beat, rhythm&rsquo; vs <span lang=\"hi\">टाल</span> /ʈaːl/ &lsquo;put off!&rsquo; (from <span lang=\"hi\">टालना</span>).",)),
        dict(a="b", b="bʱ", what="voiced vs breathy-voiced", A=["b"], B=["b̤", "bʱ"],
             en=("English has no breathy-voiced stops; an English listener hears <span lang=\"hi\">भार</span> as a slightly odd &ldquo;bar&rdquo;.",),
             hi=("<span lang=\"hi\">बार</span> /baːr/ &lsquo;time, turn&rsquo; vs <span lang=\"hi\">भार</span> /bʱaːr/ &lsquo;weight&rsquo;.",)),
        dict(a="k", b="kʰ", what="plain vs aspirated", A=["k"], B=["kʰ"],
             en=("<em>kit</em> [kʰɪt] and <em>skit</em> [skɪt]: again position, not meaning.",),
             hi=("<span lang=\"hi\">काल</span> /kaːl/ &lsquo;time&rsquo; vs <span lang=\"hi\">खाल</span> /kʰaːl/ &lsquo;skin, hide&rsquo;.",)),
        dict(a="v", b="w", what="labiodental vs labial-velar", A=["v", "ʋ"], B=["w"],
             en=("<em>vine</em> vs <em>wine</em>, <em>vest</em> vs <em>west</em>.",),
             hi=("One labial sound, usually [ʋ] (between English v and w): <span lang=\"hi\">वन</span> /ʋən/ &lsquo;forest&rsquo;. Hindi speakers learning English often merge <em>vine</em> and <em>wine</em>.",)),
        dict(a="s", b="ʃ", what="alveolar vs post-alveolar", A=["s", "s̪", "s̪|s"], B=["ʃ"],
             en=("<em>see</em> vs <em>she</em>, <em>sore</em> vs <em>shore</em>.",),
             hi=("<span lang=\"hi\">सेर</span> /seːr/ (a weight) vs <span lang=\"hi\">शेर</span> /ʃeːr/ &lsquo;lion&rsquo;; but /ʃ/ comes mainly from Sanskrit and Perso-Arabic words, and many speakers say [s] for both.",)),
        dict(a="n", b="ŋ", what="alveolar vs velar nasal", A=["n", "n̪", "n̻", "n̪|n"], B=["ŋ"],
             en=("<em>sin</em> vs <em>sing</em>, <em>ran</em> vs <em>rang</em>.",),
             hi=("[ŋ] occurs before /k ɡ/ (<span lang=\"hi\">रंग</span> [rəŋɡ] &lsquo;colour&rsquo;), where [n] cannot: an allophone in most analyses.",)),
        dict(a="z", b="dʒ", what="fricative vs affricate", A=["z", "z̪|z", "z̪"], B=["d̠ʒ", "dʒ"],
             en=("<em>zest</em> vs <em>jest</em>, <em>rays</em> vs <em>rage</em>.",),
             hi=("<span lang=\"hi\">ज़रा</span> /zəraː/ &lsquo;a little&rsquo; vs <span lang=\"hi\">जरा</span> /dʒəraː/ &lsquo;old age&rsquo;; /z/ is a loan sound (the dot, <em>nuqtā</em>, marks it) and many speakers say [dʒ] for both.",)),
    ]
    Q = []
    for qp in QPAIRS:
        row = dict(a=qp["a"], b=qp["b"], what=qp["what"])
        for key in ("eng", "hind"):
            L = LANG[key]
            pa, pb = present(L["inv"], qp["A"]), present(L["inv"], qp["B"])
            verdict = "two" if pa and pb else "one"
            # agreement over all inventories of the language
            agree = sum(1 for inv in L["all"].values() if (bool(present(inv, qp["A"])) and bool(present(inv, qp["B"]))) == (verdict == "two"))
            listed = [s for s in qp["A"] + qp["B"] if s in L["inv"]]
            row[key] = dict(v=verdict, a=pa, b=pb, marg=("marginal" in (pa, pb)), listed=listed, agree=agree, n=len(L["all"]),
                            ex=(qp["en"] if key == "eng" else qp["hi"])[0])
        Q.append(row)
        print(f"  {qp['a']}/{qp['b']}: English {row['eng']['v']} ({row['eng']['agree']}/{row['eng']['n']}, {row['eng']['listed']}), "
              f"Hindi {row['hind']['v']} ({row['hind']['agree']}/{row['hind']['n']}, {row['hind']['listed']}{' marginal' if row['hind']['marg'] else ''})")
    I["q_n_pairs"] = len(Q)

    # IPA chart: place x manner, with which cells each language's median inventory fills
    PLACES = ["bilabial", "labio-dental", "dental", "alveolar", "post-alveolar", "retroflex", "palatal", "velar", "uvular", "glottal"]
    CH = [  # manner, [cells: list of symbols (voiceless first) or None]
        ("plosive", [["p", "b"], None, ["t̪", "d̪"], ["t", "d"], None, ["ʈ", "ɖ"], ["c", "ɟ"], ["k", "ɡ"], ["q", "ɢ"], ["ʔ"]]),
        ("affricate", [None, None, None, ["ts", "dz"], ["t̠ʃ", "d̠ʒ"], None, None, None, None, None]),
        ("nasal", [["m"], ["ɱ"], ["n̪"], ["n"], None, ["ɳ"], ["ɲ"], ["ŋ"], ["ɴ"], None]),
        ("trill", [["ʙ"], None, None, ["r"], None, None, None, None, ["ʀ"], None]),
        ("tap / flap", [None, ["ⱱ"], None, ["ɾ"], None, ["ɽ"], None, None, None, None]),
        ("fricative", [["ɸ", "β"], ["f", "v"], ["θ", "ð"], ["s", "z"], ["ʃ", "ʒ"], ["ʂ", "ʐ"], ["ç", "ʝ"], ["x", "ɣ"], ["χ", "ʁ"], ["h", "ɦ"]]),
        ("approximant", [["w"], ["ʋ"], None, ["ɹ"], None, ["ɻ"], ["j"], ["ɰ"], None, None]),
        ("lateral approx.", [None, None, ["l̪"], ["l"], None, ["ɭ"], ["ʎ"], ["ʟ"], None, None]),
    ]
    STRIP = "ʰʱ\u0324ːˑ\u0303"

    def base_cons(p):
        """chart cell symbol of an inventory phoneme: drop aspiration, breathy voice, length, nasalisation;
        'X̪|X' (dental or alveolar, undecided) goes to the alveolar cell; laminal t̻ counts as dental."""
        if "|" in p:
            p = p.split("|")[1]
        p = p.replace("\u033b", "\u032a").replace("\u0361", "")
        q = "".join(ch for ch in p if ch not in STRIP)
        if q in ("tʃ", "dʒ"):
            q = q[0] + "\u0320" + q[1]
        if q == "β\u031e":
            q = "ʋ"
        return q

    VCELLS = {"i": (0, 0), "y": (0, 0), "ɨ": (0, 2), "ʉ": (0, 2), "ɯ": (0, 4), "u": (0, 4), "ɪ": (1, 0.6), "ʏ": (1, 0.6), "ʊ": (1, 3.4),
              "e": (2, 0.33), "ø": (2, 0.33), "ɘ": (2, 2.1), "ɵ": (2, 2.1), "ɤ": (2, 4), "o": (2, 4), "ə": (3, 2.2),
              "ɛ": (4, 0.67), "œ": (4, 0.67), "ɜ": (4, 2.3), "ɞ": (4, 2.3), "ʌ": (4, 4), "ɔ": (4, 4), "æ": (5, 0.85), "ɐ": (5, 2.4),
              "a": (6, 1), "ɶ": (6, 1), "ɑ": (6, 4), "ɒ": (6, 4)}

    def base_vowel(p):
        L = [ch for ch in p if ch in VCELLS]
        return L[0] if len(L) == 1 else None   # diphthongs (two vowel letters) are not placed

    chart = {"places": PLACES, "rows": [[m, cells] for m, cells in CH], "vowels": [[v, h, b] for v, (h, b) in VCELLS.items()], "langs": {}}
    for key in ("eng", "hind"):
        L = LANG[key]
        cells, vows, diph = defaultdict(list), defaultdict(list), []
        for p, m in L["inv"].items():
            seg = ph[(ph.InventoryID == L["id"]) & (ph.Phoneme == p)].SegmentClass.iloc[0]
            if seg == "consonant":
                cells[base_cons(p)].append(p + ("*" if m else ""))
            elif seg == "vowel":
                b = base_vowel(p)
                if b:
                    vows[b].append(p + ("*" if m else ""))
                else:
                    diph.append(p + ("*" if m else ""))
        known = {s for _, cs in CH for c in cs if c for s in c}
        other = sorted(s for s in cells if s not in known)
        chart["langs"][key] = {"name": L["name"], "id": L["id"], "src": L["src"], "dial": L["dial"], "c": dict(cells), "v": dict(vows),
                               "diph": sorted(diph), "other": other}
        I[f"{key}_chart_ncells"] = len([s for s in cells if s in known])
        print(f"  chart {key}: {len(cells)} consonant cells, unplaced {other}, vowels {sorted(vows)}, diphthongs {diph}")

    # ------------------------------------------------------------------ PHOIBLE feature vectors
    print("PHOIBLE feature vectors")
    FEATS = [("syllabic", "syllabic"), ("consonantal", "consonantal"), ("sonorant", "sonorant"), ("continuant", "continuant"),
             ("delayedRelease", "delayed release"), ("approximant", "approximant"), ("tap", "tap"), ("trill", "trill"),
             ("nasal", "nasal"), ("lateral", "lateral"), ("labial", "labial"), ("round", "round"), ("labiodental", "labiodental"),
             ("coronal", "coronal"), ("anterior", "anterior"), ("distributed", "distributed"), ("strident", "strident"),
             ("dorsal", "dorsal"), ("high", "high"), ("low", "low"), ("front", "front"), ("back", "back"), ("tense", "tense"),
             ("periodicGlottalSource", "voice"), ("spreadGlottis", "spread glottis"), ("constrictedGlottis", "constricted glottis"),
             ("long", "long")]
    SYMS = ["p", "pʰ", "b", "b̤", "t", "tʰ", "d", "t̪", "t̪ʰ", "d̪", "d̪̤", "ʈ", "ʈʰ", "ɖ", "ɖ̤", "k", "kʰ", "ɡ", "ɡ̤", "t̠ʃ", "d̠ʒ",
            "m", "n", "n̪", "ɳ", "ŋ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʂ", "h", "ɦ", "l", "ɭ", "r", "ɾ", "ɽ", "ɹ", "j", "w", "ʋ",
            "i", "iː", "ɪ", "e", "ɛ", "æ", "a", "aː", "ə", "ɑ", "ɔ", "o", "ʊ", "u", "uː"]
    cols = [f for f, _ in FEATS]
    F = {}
    sub = ph[ph.Phoneme.isin(SYMS)]
    for s, d in sub.groupby("Phoneme"):
        vec = Counter(tuple(r) for r in d[cols].astype(str).values)
        best, n = vec.most_common(1)[0]
        F[s] = {"v": [v.replace("-", "−") for v in best], "n": int(len(d)), "agree": round(100 * n / len(d), 1)}
    missing = [s for s in SYMS if s not in F]
    assert not missing, missing
    eng_set = {base for base in LANG["eng"]["inv"]}
    hind_set = {base for base in LANG["hind"]["inv"]}
    feat = {"feats": [lab for _, lab in FEATS], "keys": cols,
            "syms": [{"s": s, "v": F[s]["v"], "n": F[s]["n"], "agree": F[s]["agree"], "en": s in eng_set, "hi": s in hind_set} for s in SYMS]}
    I["feat_n_syms"] = len(SYMS)
    I["feat_n_feats"] = len(FEATS)
    I["feat_n_phoible"] = len(ph.columns) - list(ph.columns).index("tone")
    I["feat_min_agree"] = min(F[s]["agree"] for s in SYMS)

    def ham(a, b):
        va, vb = F[a]["v"], F[b]["v"]
        return sum(1 for x, y in zip(va, vb) if x != y), sum(1 for x, y in zip(va, vb) if x != y and "0" not in (x, y))
    for a, b in [("p", "b"), ("p", "pʰ"), ("t̪", "ʈ"), ("p", "k"), ("s", "ʃ"), ("i", "ɪ"), ("p", "a"), ("b", "b̤"), ("m", "b"), ("v", "w"), ("n", "ŋ")]:
        h, hs = ham(a, b)
        I[f"ham_{a}_{b}"] = h
        print(f"  {a} vs {b}: {h} features differ ({hs} as +/- flips): " +
              ", ".join(lab for (k, lab), x, y in zip(FEATS, F[a]["v"], F[b]["v"]) if x != y))
    print("  lowest per-symbol agreement with the modal vector:", sorted((F[s]["agree"], s) for s in SYMS)[:5])

    # ------------------------------------------------------------------ write intro_data.js
    out = {"mp": MP, "quiz": Q, "chart": chart, "feat": feat,
           "inv": {k: {"id": L["id"], "src": L["src"], "dial": L["dial"], "n": len(L["all"])} for k, L in LANG.items()}}
    with open(os.path.join(HERE, "intro_data.js"), "w", encoding="utf8") as fh:
        fh.write("// Data for the opening half of sounds_language.html, written by make_intro.py: minimal pairs and VOT examples from\n")
        fh.write("// CMU ARCTIC ksp (audio in audio/), the same-token quiz, the IPA chart and feature vectors from PHOIBLE 2.0 (CC BY 4.0).\n")
        fh.write("window.INTRO = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    I["data_kb"] = round(os.path.getsize(os.path.join(HERE, "intro_data.js")) / 1024)
    I["n_utt"] = NUM_UTT

    for k, v in I.items():
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, (np.integer,)):
            v = int(v)
        NUM["i_" + k] = v
    if standalone:
        json.dump(NUM, open(os.path.join(HERE, "numbers.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
        print("merged", len(I), "numbers into numbers.json")
    return NUM


if __name__ == "__main__":
    main()
