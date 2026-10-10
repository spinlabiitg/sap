"""
Figures, quoted numbers and the live-demo data for ../sounds_language.html (Module 2, page 2: phoneme inventories, India and the world).

Run from this folder:   python make_figures.py      (then python build_page.py)
Outputs:  fig/*.png   numbers.json   inventories.js (compact inventories for the live demo)
          and, through make_intro.py (opening half, sections 01-03): sentence.js, intro_data.js, audio/*.wav

Data (kept OUTSIDE the repo; only derived numbers/figures are published):
  PHOIBLE 2.0  (Moran & McCloy eds. 2019, CC BY 4.0)          research/dBase/PHOIBLE/phoible.csv
  Glottolog    (Hammarstroem, Forkel, Haspelmath & Bank, CC BY 4.0)  research/dBase/GLOTTOLOG/languages.csv

Methodological choices (all stated on the page):
  * Language = Glottolog language-level languoid. PHOIBLE inventories coded with a dialect Glottocode are rolled up to
    their parent language; the few coded with a family-level Glottocode (or none) are left out.
  * Languages of India = Glottolog languages whose Countries field contains "IN" (this includes cross-border languages).
  * One inventory per language: the "median inventory", i.e. the inventory whose consonant + vowel count is closest to
    the median of that language's inventories; ties broken by a source preference order (ph, ea, ra, upsid, spa,
    saphon, aa, er), then by InventoryID. Sensitivity: preferred-source pick, random pick (200 draws), all inventories.
  * Sizes count every listed segment (as PHOIBLE does, marginal ones included); "has feature X" uses non-marginal
    phonemes only.
"""
import json
import os
import random
import unicodedata as ud
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)
DB = os.path.join(HERE, "..", "..", "..", "..", "..", "research", "dBase")
if not os.path.isdir(DB):
    DB = r"C:/Users/IITG/Documents/tommy_assistant/research/dBase"
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
    "axes.axisbelow": True,
})
FAMCOL = {"Indo-Aryan": BLUE, "Dravidian": RED, "Austroasiatic": GOLD, "Sino-Tibetan": TEAL, "Other": "#7B5EA7"}
FAMS = list(FAMCOL)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig", name)


def pct(a):
    return round(100.0 * float(np.mean(a)), 1)


# ====================================================================== load
ph = pd.read_csv(os.path.join(DB, "PHOIBLE", "phoible.csv"), low_memory=False, keep_default_na=False, na_values=["NA"])
gl = pd.read_csv(os.path.join(DB, "GLOTTOLOG", "languages.csv"), keep_default_na=False, na_values=[""])
g = gl.set_index("Glottocode")
ph["marg"] = ph.Marginal.astype(str).eq("True")
NUM["n_rows"] = len(ph)
NUM["n_inv_all"] = ph.InventoryID.nunique()
NUM["n_iso_all"] = ph.ISO6393.nunique()
NUM["n_sources"] = ph.Source.nunique()
NUM["n_marginal_rows"] = int(ph.marg.sum())

inv = ph.drop_duplicates("InventoryID")[["InventoryID", "Glottocode", "LanguageName", "SpecificDialect", "Source"]].copy()
inv["lvl"] = inv.Glottocode.map(g.Level)


def to_lang(r):
    if r.lvl == "language":
        return r.Glottocode
    if r.lvl == "dialect":
        return g.Language_ID.get(r.Glottocode)
    return None


inv["lang"] = inv.apply(to_lang, axis=1)
NUM["n_inv_dialect_rolled"] = int((inv.lvl == "dialect").sum())
NUM["n_inv_dropped"] = int(inv.lang.isna().sum())
inv = inv[inv.lang.notna()].copy()
cnt = ph.groupby(["InventoryID", "SegmentClass"]).size().unstack(fill_value=0)
inv["C"] = inv.InventoryID.map(cnt.get("consonant")).fillna(0).astype(int)
inv["V"] = inv.InventoryID.map(cnt.get("vowel")).fillna(0).astype(int)
inv["T"] = inv.InventoryID.map(cnt.get("tone")).fillna(0).astype(int)
inv["CV"] = inv.C + inv.V
NUM["n_inv"] = len(inv)
NUM["n_lang"] = inv.lang.nunique()
NUM["n_glotto_lang"] = int((gl.Level == "language").sum())
NUM["coverage_pct"] = round(100 * NUM["n_lang"] / NUM["n_glotto_lang"], 1)
NUM["n_lang_multi"] = int((inv.groupby("lang").size() > 1).sum())
src_n = inv.groupby("Source").size().sort_values(ascending=False)
for s, n in src_n.items():
    NUM[f"src_{s}"] = int(n)
print("languages", NUM["n_lang"], "inventories", NUM["n_inv"])

# family / group
famname = g.Name


def family_of(gc):
    fid = g.Family_ID.get(gc)
    return famname.get(fid) if isinstance(fid, str) else "isolate"


# Indo-European languages of India that are not Indo-Aryan: Northern Pashto (Iranian) and the Portuguese creoles
NOT_IA = {"nort2646"} | set(gl[gl.Name.str.contains("Portuguese", na=False)].Glottocode)


def group_of(gc):
    f = family_of(gc)
    if f == "Indo-European":
        return "Other" if gc in NOT_IA else "Indo-Aryan"
    return f if f in ("Dravidian", "Austroasiatic", "Sino-Tibetan") else "Other"


# India
gl_lang = gl[gl.Level == "language"].copy()
# Glottolog pseudo-families that are not spoken languages (kept out of coverage denominators)
PSEUDO = {"Bookkeeping", "Sign Language", "Unattested", "Artificial Language", "Speech Register"}
gl_lang["fam"] = gl_lang.Glottocode.map(family_of)
gl_lang["ctry"] = gl_lang.Countries.fillna("").str.split(";")
india = set(gl_lang[gl_lang.ctry.apply(lambda l: "IN" in l)].Glottocode)
india_only = set(gl_lang[gl_lang.Countries.fillna("") == "IN"].Glottocode)
NUM["n_india_glotto"] = len(india)
NUM["n_india_only_glotto"] = len(india_only)
spoken = set(gl_lang[~gl_lang.fam.isin(PSEUDO)].Glottocode)
NUM["n_india_spoken"] = len(india & spoken)
NUM["n_india_pseudo"] = len(india - spoken)
NUM["n_glotto_spoken"] = len(spoken)
NUM["coverage_spoken_pct"] = round(100 * len(set(inv.lang) & spoken) / len(spoken), 1)
inv["india"] = inv.lang.isin(india)
NUM["n_india_lang"] = inv[inv.india].lang.nunique()
NUM["india_coverage_spoken_pct"] = round(100 * len(set(inv[inv.india].lang) & spoken) / len(india & spoken), 1)
NUM["n_india_lang_direct"] = inv[inv.Glottocode.isin(india)].Glottocode.nunique()  # without the dialect roll-up
NUM["n_india_inv_direct"] = int(inv.Glottocode.isin(india).sum())
NUM["n_india_inv"] = int(inv.india.sum())
NUM["india_coverage_pct"] = round(100 * NUM["n_india_lang"] / NUM["n_india_glotto"], 1)
NUM["n_india_only_lang"] = inv[inv.lang.isin(india_only)].lang.nunique()
for s, n in inv[inv.india].groupby("Source").size().items():
    NUM[f"india_src_{s}"] = int(n)
NUM["n_india_multi"] = int((inv[inv.india].groupby("lang").size() > 1).sum())
NUM["india_max_inv"] = int(inv[inv.india].groupby("lang").size().max())
ra_all = inv[inv.Source == "ra"]
NUM["n_ra_total"] = len(ra_all)
NUM["n_ra_india"] = int(ra_all.india.sum())

# ====================================================================== segment classifier (place x manner)
MOD_KEEP = set()
CLICKS = set("ǀǃǁǂʘ‼")
RETRO = set("ʈɖɳɭʂʐɽɻᶑ")
PLACES = ["Labial", "Dental", "Alveolar", "Post-alv.", "Retroflex", "Palatal", "Velar", "Uvular", "Phar./Glottal"]
MANNERS = ["Stop", "Affricate", "Nasal", "Fricative", "Tap/trill", "Lateral", "Approximant"]
B = {}
for chs, pl, mn in [
    ("pbɓ", "Labial", "Stop"), ("mɱ", "Labial", "Nasal"), ("ɸβfv", "Labial", "Fricative"), ("ʋwʍ", "Labial", "Approximant"),
    ("ⱱʙ", "Labial", "Tap/trill"),
    ("tdɗ", "Alveolar", "Stop"), ("n", "Alveolar", "Nasal"), ("sz", "Alveolar", "Fricative"), ("θð", "Dental", "Fricative"),
    ("rɾR", "Alveolar", "Tap/trill"), ("ɹ", "Alveolar", "Approximant"), ("lɫ", "Alveolar", "Lateral"),
    ("ɬɮ", "Alveolar", "Fricative"), ("ɺ", "Alveolar", "Tap/trill"),
    ("ʃʒʆʓ", "Post-alv.", "Fricative"),
    ("ʈɖᶑ", "Retroflex", "Stop"), ("ɳ", "Retroflex", "Nasal"), ("ʂʐ", "Retroflex", "Fricative"), ("ɽ", "Retroflex", "Tap/trill"),
    ("ɻ", "Retroflex", "Approximant"), ("ɭ", "Retroflex", "Lateral"),
    ("cɟʄȶ", "Palatal", "Stop"), ("ɲȵ", "Palatal", "Nasal"), ("çʝɕʑ", "Palatal", "Fricative"), ("jɥ", "Palatal", "Approximant"),
    ("ʎȴ", "Palatal", "Lateral"),
    ("kɡɠ", "Velar", "Stop"), ("ŋ", "Velar", "Nasal"), ("xɣ", "Velar", "Fricative"), ("ɰ", "Velar", "Approximant"), ("ʟ", "Velar", "Lateral"),
    ("qɢʛ", "Uvular", "Stop"), ("ɴ", "Uvular", "Nasal"), ("χʁ", "Uvular", "Fricative"), ("ʀ", "Uvular", "Tap/trill"),
    ("ʔʡ", "Phar./Glottal", "Stop"), ("hɦħʕʜʢ", "Phar./Glottal", "Fricative"),
]:
    for ch in chs:
        B[ch] = (pl, mn)
VOW = {}
for chs, h, b in [("iy", 0, 0), ("ɨʉ", 0, 1), ("ɯu", 0, 2), ("ɪʏ", 1, 0), ("ʊ", 1, 2), ("eø", 2, 0), ("ɘɵ", 2, 1), ("ɤo", 2, 2),
                  ("əɚ", 3, 1), ("ɛœ", 4, 0), ("ɜɞ", 4, 1), ("ʌɔ", 4, 2), ("æ", 5, 0), ("ɐ", 5, 1), ("aɶ", 6, 0), ("ɑɒ", 6, 2)]:
    for ch in chs:
        VOW[ch] = (h, b)


def letters(p):
    """base letters of a phoneme: drop combining marks (Mn) and modifier letters (Lm, Sk) such as ʰ ʱ ʷ ʲ ː ˤ ʼ ⁿ."""
    return [ch for ch in p if ud.category(ch) not in ("Mn", "Lm", "Sk") and ch != "|"]


def classify_cons(p):
    """(manner, place) for a consonant glyph, or None (clicks, double articulations, oddities)."""
    q = p.split("|")[0]  # 't̪|t' (dental-or-alveolar, UPSID convention): classify the first variant
    L = letters(q)
    if not L or any(ch in CLICKS for ch in L):
        return None
    dental = "\u032a" in q
    if len(L) == 1:
        if L[0] not in B:
            return None
        pl, mn = B[L[0]]
        if mn == "Fricative" and "\u031e" in q:  # lowered fricative = approximant (β̞)
            mn = "Approximant"
        if pl == "Alveolar" and dental:
            pl = "Dental"
        return mn, pl
    if len(L) >= 2 and L[0] in B and B[L[0]][1] == "Nasal":
        # prenasalised stop/affricate/fricative (nd, n̠d̠ʒ, ɳɖʐ, ndz, ɱv): classify what follows the nasal
        k = q.index(L[1], q.index(L[0]) + 1)
        rest = classify_cons(q[k:])
        if rest and rest[0] in ("Stop", "Affricate", "Fricative"):
            return rest
    if len(L) == 2:
        a, b = L
        if a in B and b in B:
            (pa, ma), (pb, mb) = B[a], B[b]
            if ma == "Stop" and mb == "Fricative":  # affricate: place from the fricative part
                pl = pb
                if pl == "Alveolar" and dental:
                    pl = "Dental"
                return "Affricate", pl
            if ma == "Nasal" and mb == "Stop" and pa == pb:  # prenasalised stop (mb, nd, ŋɡ)
                return "Stop", ("Dental" if pb == "Alveolar" and dental else pb)
    return None


def classify_vowel(p):
    L = letters(p)
    if len(L) == 1 and L[0] in VOW:
        return VOW[L[0]]
    return None  # diphthongs and odd glyphs


STOPLIKE = ("Stop", "Affricate")
VD = {"b": "p", "d": "t", "ɖ": "ʈ", "ɟ": "c", "ɡ": "k", "ɢ": "q", "ʒ": "ʃ", "ʑ": "ɕ", "ʝ": "ç", "z": "s", "ð": "θ", "ɮ": "ɬ", "v": "f", "β": "ɸ"}


def lar_key(p):
    """(place key, laryngeal category) for a stop/affricate: voiced letters mapped to voiceless ones."""
    base = "".join(VD.get(ch, ch) for ch in p if ch not in "ʰʱ\u0324ː")
    voiced = any(ch in VD for ch in p)
    asp = "ʰ" in p
    breathy = ("ʱ" in p) or ("\u0324" in p)
    if voiced and breathy:
        cat = "vd_breathy"
    elif voiced:
        cat = "vd"
    elif asp:
        cat = "vl_asp"
    else:
        cat = "vl"
    return base, cat


# per-inventory segment table with flags
seg = ph[["InventoryID", "Phoneme", "SegmentClass", "marg", "coronal", "anterior", "distributed"]].copy()
seg = seg[seg.InventoryID.isin(inv.InventoryID)]
cls = {p: classify_cons(p) for p in seg[seg.SegmentClass == "consonant"].Phoneme.unique()}
seg["manner"] = seg.Phoneme.map(lambda p: cls.get(p, (None, None))[0] if cls.get(p) else None)
seg["place"] = seg.Phoneme.map(lambda p: cls.get(p)[1] if cls.get(p) else None)
isC = seg.SegmentClass.eq("consonant")
isV = seg.SegmentClass.eq("vowel")
seg["retro"] = isC & seg.Phoneme.map(lambda p: (any(ch in RETRO for ch in p) or "˞" in p) and not any(ch in CLICKS for ch in p))
seg["retro_feat"] = isC & seg.coronal.eq("+") & seg.anterior.eq("-") & seg.distributed.eq("-")
seg["stoplike"] = isC & seg.manner.isin(STOPLIKE)
seg["asp"] = seg.stoplike & seg.Phoneme.str.contains("ʰ")
seg["breathy"] = seg.stoplike & (seg.Phoneme.str.contains("ʱ") | seg.Phoneme.str.contains("\u0324"))
seg["nasalV"] = isV & seg.Phoneme.str.contains("\u0303")
seg["longV"] = isV & seg.Phoneme.str.contains("ː")
seg["dental"] = isC & seg.Phoneme.str.contains("\u032a") & ~seg.Phoneme.str.contains("|", regex=False)
NUM["n_unclassified_cons_glyphs"] = sum(1 for v in cls.values() if v is None)
NUM["n_cons_glyphs"] = len(cls)
# share of consonant ROWS that the chart classifier places (weighted by use)
NUM["chart_cover_pct"] = pct(seg[isC].manner.notna())


def inv_features(d, use_marg=False):
    if not use_marg:
        d = d[~d.marg]
    C = d[d.SegmentClass == "consonant"]
    V = d[d.SegmentClass == "vowel"]
    st = C[C.stoplike]
    keys = defaultdict(set)
    for p in st.Phoneme:
        k, c = lar_key(p)
        keys[k].add(c)
    four = any({"vl", "vl_asp", "vd", "vd_breathy"} <= s for s in keys.values())
    vs = set(V.Phoneme)
    lenc = any(p.endswith("ː") and p[:-1] in vs for p in vs)
    cons = set(C.Phoneme)
    dent_alv = False
    for p in cons:
        if "\u032a" in p and "|" not in p:
            q = p.replace("\u032a", "")
            L = letters(q)
            if q in cons and L and L[-1] in "tdnlrɾsz":
                dent_alv = True
    return dict(retro=bool(C.retro.any()), retro_feat=bool(C.retro_feat.any()), n_retro=int(C.retro.sum()),
                asp=bool(st.asp.any()), breathy=bool(st.breathy.any()), four=bool(four),
                nasalV=bool(V.nasalV.any()), lenV=bool(lenc), dent_alv=bool(dent_alv),
                tone=bool((d.SegmentClass == "tone").any()), n_stop=len(st))


feat = {iid: inv_features(d) for iid, d in seg.groupby("InventoryID")}
feat_m = {iid: inv_features(d, use_marg=True) for iid, d in seg.groupby("InventoryID")}
FEATS = ["retro", "asp", "breathy", "four", "nasalV", "lenV", "dent_alv", "tone"]
FLAB = {"retro": "retroflex consonant(s)", "asp": "aspirated stop(s)", "breathy": "breathy-voiced stop(s)",
        "four": "four-way p pʰ b bʱ contrast", "nasalV": "nasalised vowel(s)", "lenV": "vowel length contrast",
        "dent_alv": "dental–alveolar contrast", "tone": "tones listed"}
for f in FEATS + ["retro_feat", "n_retro", "n_stop"]:
    inv[f] = inv.InventoryID.map(lambda i: feat[i][f])

# ====================================================================== one inventory per language
PREF = ["ph", "ea", "ra", "upsid", "spa", "saphon", "aa", "er"]
inv["pref"] = inv.Source.map(lambda s: PREF.index(s) if s in PREF else 99)


def pick_median(d):
    m = d.CV.median()
    d = d.assign(dist=(d.CV - m).abs()).sort_values(["dist", "pref", "InventoryID"])
    return d.InventoryID.iloc[0]


def pick_pref(d):
    return d.sort_values(["pref", "InventoryID"]).InventoryID.iloc[0]


groups = dict(tuple(inv.groupby("lang")))
chosen = {L: pick_median(d) for L, d in groups.items()}
chosen_pref = {L: pick_pref(d) for L, d in groups.items()}
I = inv.set_index("InventoryID")


def lang_table(choice):
    t = I.loc[list(choice.values())].copy()
    t["InventoryID"] = t.index
    t = t.reset_index(drop=True)
    t["name"] = t.lang.map(g.Name)
    t["group"] = t.lang.map(group_of)
    t["family"] = t.lang.map(family_of)
    t["lat"] = t.lang.map(g.Latitude)
    t["lon"] = t.lang.map(g.Longitude)
    t["macro"] = t.lang.map(g.Macroarea)
    return t


LT = lang_table(chosen)
LP = lang_table(chosen_pref)
W, IN_, ROW = LT, LT[LT.india], LT[~LT.india]
NUM["n_row_lang"] = len(ROW)
for k in ["C", "V", "T", "CV"]:
    NUM[f"world_med_{k}"] = float(np.median(W[k]))
    NUM[f"india_med_{k}"] = float(np.median(IN_[k]))
    NUM[f"row_med_{k}"] = float(np.median(ROW[k]))
NUM["world_mean_C"] = round(W.C.mean(), 1)
NUM["world_mean_V"] = round(W.V.mean(), 1)
NUM["world_tone_pct"] = pct(W["T"] > 0)
NUM["world_q1_C"], NUM["world_q3_C"] = [float(x) for x in np.percentile(W.C, [25, 75])]
NUM["world_q1_V"], NUM["world_q3_V"] = [float(x) for x in np.percentile(W.V, [25, 75])]
NUM["world_min_C"], NUM["world_max_C"] = int(W.C.min()), int(W.C.max())
NUM["world_min_V"], NUM["world_max_V"] = int(W.V.min()), int(W.V.max())
NUM["world_max_T"] = int(W["T"].max())
# percentile of India medians within the world distribution
NUM["india_C_pctile"] = pct(W.C < NUM["india_med_C"])


def extremes(col, n=6):
    a = W.sort_values(col)[["name", "C", "V", "T", "Source", "lang"]]
    return a.head(n), a.tail(n)


for col in ["C", "V", "CV"]:
    lo, hi = extremes(col)
    print("lowest", col, lo.values.tolist())
    print("highest", col, hi.values.tolist())


def lang_row(name_gc):
    r = W[W.lang == name_gc]
    return r.iloc[0] if len(r) else None


for key, gc in [("taa", "huaa1248"), ("roto", "roto1249"), ("pira", "pira1253"), ("hawa", "hawa1245"), ("eng", "stan1293"),
                ("hind", "hind1269"), ("mala", "mala1464"), ("tami", "tami1289"), ("assa", "assa1263"), ("kild", "kild1236"), ("sedu", "juho1239"),
                ("mand", "mand1415"), ("japa", "nucl1643")]:
    r = lang_row(gc)
    if r is None:
        print("missing", key, gc)
        continue
    NUM[f"{key}_C"], NUM[f"{key}_V"], NUM[f"{key}_T"] = int(r.C), int(r.V), int(r["T"])
    allc = inv[inv.lang == gc]
    NUM[f"{key}_ninv"] = len(allc)
    NUM[f"{key}_Cmin"], NUM[f"{key}_Cmax"] = int(allc.C.min()), int(allc.C.max())
    NUM[f"{key}_Vmin"], NUM[f"{key}_Vmax"] = int(allc.V.min()), int(allc.V.max())
    NUM[f"{key}_CVmin"], NUM[f"{key}_CVmax"] = int(allc.CV.min()), int(allc.CV.max())
    print(key, r["name"], r.C, r.V, r["T"], r.Source, "inventories:", allc[["Source", "C", "V", "T"]].values.tolist())

# ====================================================================== feature frequencies + sensitivity
def rates(t):
    return {f: pct(t[f]) for f in FEATS}


R = {"india": rates(IN_), "row": rates(ROW), "world": rates(W)}
for grp in FAMS:
    R[grp] = rates(IN_[IN_.group == grp])
    NUM[f"n_{grp.replace('-', '_')}"] = int((IN_.group == grp).sum())
for k, v in R.items():
    for f, x in v.items():
        NUM[f"{k.replace('-', '_')}_{f}"] = x
# tones: ea, upsid, saphon and er never list tones, so compare only inventories from sources that do (ph, aa, spa, ra)
TONE_SRC = {"ph", "aa", "spa", "ra"}
for k, t in [("india", IN_), ("row", ROW), ("ST", IN_[IN_.group == "Sino-Tibetan"])]:
    tt = t[t.Source.isin(TONE_SRC)]
    NUM[f"{k}_tone_tsrc"] = pct(tt.tone)
    NUM[f"{k}_n_tsrc"] = len(tt)
for s in ["ea", "upsid", "saphon", "er", "ph", "aa", "spa", "ra"]:
    NUM[f"tone_listed_{s}"] = pct(inv[inv.Source == s]["T"] > 0)
NUM["india_only_retro"] =pct(IN_[IN_.lang.isin(india_only)].retro)
NUM["india_only_breathy"] = pct(IN_[IN_.lang.isin(india_only)].breathy)
NUM["india_only_asp"] = pct(IN_[IN_.lang.isin(india_only)].asp)
NUM["retro_feat_agree_pct"] = pct(W.retro == W.retro_feat)
NUM["n_retro_disagree"] = int((W.retro != W.retro_feat).sum())
print("retro glyph vs feature disagreements:", W[W.retro != W.retro_feat][["name", "Source", "retro", "retro_feat"]].values.tolist()[:20])
NUM["india_mean_n_retro"] = round(IN_[IN_.retro].n_retro.mean(), 1)
NUM["row_mean_n_retro"] = round(ROW[ROW.retro].n_retro.mean(), 1)
# with marginal phonemes counted
for f in ["retro", "breathy", "asp"]:
    NUM[f"india_{f}_withmarg"] = pct(IN_.InventoryID.map(lambda i: feat_m[i][f]))
    NUM[f"row_{f}_withmarg"] = pct(ROW.InventoryID.map(lambda i: feat_m[i][f]))
# retroflex by macroarea (rest of world)
for m, d in ROW[ROW.macro.isin(["Africa", "Eurasia", "Papunesia", "Australia", "North America", "South America"])].groupby("macro"):
    NUM[f"row_retro_{m.replace(' ', '_')}"] = pct(d.retro)
    NUM[f"row_n_{m.replace(' ', '_')}"] = len(d)
# Indian languages share among all retroflex languages
NUM["india_share_of_retro_langs"] = round(100 * IN_.retro.sum() / W.retro.sum(), 1)
NUM["n_world_retro"] = int(W.retro.sum())
NUM["n_india_retro"] = int(IN_.retro.sum())
NUM["n_india_breathy"] = int(IN_.breathy.sum())
NUM["n_world_breathy"] = int(W.breathy.sum())
NUM["india_share_of_breathy_langs"] = round(100 * IN_.breathy.sum() / max(1, W.breathy.sum()), 1)
NUM["n_world_four"] = int(W.four.sum())
NUM["n_india_four"] = int(IN_.four.sum())
NUM["india_share_of_four"] = round(100 * IN_.four.sum() / max(1, W.four.sum()), 1)

# sensitivity: preferred-source pick, random pick, all inventories
rng = random.Random(1)
sens = {}
IP, RP = LP[LP.india], LP[~LP.india]
sens["pref"] = {"india": rates(IP), "row": rates(RP), "C_world": float(np.median(LP.C)), "V_world": float(np.median(LP.V)),
                "C_india": float(np.median(IP.C)), "V_india": float(np.median(IP.V))}
ALL_I, ALL_R = inv[inv.india], inv[~inv.india]
sens["all"] = {"india": rates(ALL_I), "row": rates(ALL_R), "C_world": float(np.median(inv.C)), "V_world": float(np.median(inv.V)),
               "C_india": float(np.median(ALL_I.C)), "V_india": float(np.median(ALL_I.V))}
draws = defaultdict(list)
lang_ids = {L: d.InventoryID.tolist() for L, d in groups.items()}
for _ in range(200):
    ch = {L: rng.choice(ids) for L, ids in lang_ids.items()}
    t = I.loc[list(ch.values())]
    ti, tr = t[t.india], t[~t.india]
    for f in FEATS:
        draws["india_" + f].append(pct(ti[f]))
        draws["row_" + f].append(pct(tr[f]))
    draws["C_world"].append(float(np.median(t.C)))
    draws["V_world"].append(float(np.median(t.V)))
    draws["C_india"].append(float(np.median(ti.C)))
    draws["V_india"].append(float(np.median(ti.V)))
sens["random"] = {k: (round(float(np.min(v)), 1), round(float(np.max(v)), 1)) for k, v in draws.items()}
for f in FEATS:
    lo = min(R["india"][f], sens["pref"]["india"][f], sens["all"]["india"][f], sens["random"]["india_" + f][0])
    hi = max(R["india"][f], sens["pref"]["india"][f], sens["all"]["india"][f], sens["random"]["india_" + f][1])
    NUM[f"sens_india_{f}_lo"], NUM[f"sens_india_{f}_hi"] = lo, hi
    lo = min(R["row"][f], sens["pref"]["row"][f], sens["all"]["row"][f], sens["random"]["row_" + f][0])
    hi = max(R["row"][f], sens["pref"]["row"][f], sens["all"]["row"][f], sens["random"]["row_" + f][1])
    NUM[f"sens_row_{f}_lo"], NUM[f"sens_row_{f}_hi"] = lo, hi
for k in ["C_world", "V_world", "C_india", "V_india"]:
    NUM[f"sens_{k}_pref"] = sens["pref"][k]
    NUM[f"sens_{k}_all"] = sens["all"][k]
    NUM[f"sens_{k}_rlo"], NUM[f"sens_{k}_rhi"] = sens["random"][k]
print("sensitivity", json.dumps(sens, ensure_ascii=False)[:1500])

# ====================================================================== most common phonemes
segc = seg[seg.InventoryID.isin(W.InventoryID)]
nW = len(W)
freq = segc.groupby("Phoneme").InventoryID.nunique().sort_values(ascending=False) / nW * 100
segi = seg[seg.InventoryID.isin(IN_.InventoryID)]
freq_in = segi.groupby("Phoneme").InventoryID.nunique() / len(IN_) * 100
top = freq.head(25)
for i, (p, v) in enumerate(freq.head(12).items()):
    NUM[f"top{i + 1}_ph"] = p
    NUM[f"top{i + 1}_pct"] = round(v, 1)
NUM["n_distinct_phonemes_all"] = int(seg.Phoneme.nunique())
NUM["n_distinct_phonemes"] = int(segc.Phoneme.nunique())
once = (seg[seg.InventoryID.isin(W.InventoryID)].groupby("Phoneme").InventoryID.nunique() == 1)
NUM["n_phonemes_once"] = int(once.sum())
NUM["pct_phonemes_once"] = round(100 * once.mean(), 1)
for p in ["t", "t̪", "t̪|t", "d", "d̪", "n", "n̪", "n̪|n", "ʈ", "ɖ", "ɳ", "ɭ", "ɽ", "ʂ", "ŋ", "h", "ʔ", "s", "f", "ʃ", "r", "ɾ", "l", "a", "e", "o", "ə", "ɛ", "ɔ", "kʰ", "pʰ", "b̤", "ɡ̤", "θ", "ð", "y", "ɻ", "ɲ"]:
    NUM[f"f_{p}"] = round(float(freq.get(p, 0)), 1)
    NUM[f"fin_{p}"] = round(float(freq_in.get(p, 0)), 1)
# any coronal (dental/alveolar/undetermined) voiceless plain t
tcor = segc[segc.Phoneme.isin(["t", "t̪", "t̪|t", "t̺", "t̻"])].InventoryID.nunique() / nW * 100
NUM["f_t_any"] = round(tcor, 1)
print("top", [(p, round(v, 1)) for p, v in top.items()])

# ====================================================================== FIGURES
# background points for maps: every Glottolog language
bg = gl_lang[gl_lang.Latitude.notna()]


def bgdots(ax, box=None, s=1.2):
    b = bg if box is None else bg[(bg.Longitude.between(box[0], box[1])) & (bg.Latitude.between(box[2], box[3]))]
    ax.scatter(b.Longitude, b.Latitude, s=s, color="#C9D2DE", lw=0, zorder=1, rasterized=True)


# ---- Fig 1: sample map + coverage by macroarea
fig = plt.figure(figsize=(10, 4.3))
ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
bgdots(ax)
ax.scatter(W.lon, W.lat, s=5, color=BLUE, lw=0, zorder=2, label=f"PHOIBLE language (n = {nW})")
ax.set_xlim(-180, 190); ax.set_ylim(-58, 80); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.scatter([], [], s=6, color="#C9D2DE", label="every Glottolog language (grey)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=8, markerscale=2.5)
ax.set_title("Where PHOIBLE's languages are (one dot per language)", loc="left")
ax2 = fig.add_axes([0.73, 0.14, 0.26, 0.74])
macros = ["Africa", "Eurasia", "Papunesia", "Australia", "North America", "South America"]
cov = []
for m in macros:
    ng = int(((gl_lang.Macroarea == m) & gl_lang.Glottocode.isin(spoken)).sum())
    npb = int((W.macro == m).sum())
    cov.append((m, ng, npb, 100 * npb / ng))
    NUM[f"cov_{m.replace(' ', '_')}"] = round(100 * npb / ng, 1)
yy = np.arange(len(cov))[::-1]
ax2.barh(yy, [c[3] for c in cov], color=BLUE, height=0.62)
for y, c in zip(yy, cov):
    ax2.text(c[3] + 1.5, y, f"{c[3]:.0f}%  ({c[2]}/{c[1]})", va="center", fontsize=7.8, color=INK)
ax2.set_yticks(yy); ax2.set_yticklabels([c[0] for c in cov])
ax2.set_xlim(0, 100); ax2.set_xlabel("% of Glottolog languages\nwith a PHOIBLE inventory")
ax2.grid(axis="y", visible=False)
ax2.set_title("Coverage by macroarea", loc="left")
save(fig, "01_sample_map.png")

# ---- Fig 2: size distributions
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"wspace": 0.32, "width_ratios": [1.5, 1, 0.8]})
ax = axs[0]
bins = np.arange(0, W.C.max() + 3, 2) - 0.5
ax.hist(W.C, bins=bins, color=BLUE, alpha=0.9)
mC = NUM["world_med_C"]
ax.axvline(mC, color=RED, lw=1.4)
ax.text(mC + 2, ax.get_ylim()[1] * 0.92, f"median {mC:g}", color=RED, fontsize=8.5)
ax.set_xlabel("consonants in the inventory"); ax.set_ylabel("number of languages")
ax.set_title("Consonants", loc="left")
ytop = ax.get_ylim()[1]
small = [lang_row(gc) for gc in ("roto1249", "hawa1245", "pira1253")]
r = lang_row("huaa1248")
ax.text(42, ytop * 0.55, "fewest: Rotokas {}, Hawaiian {}, Pirahã {}\nmost: East Taa (!Xóõ) {}".format(*[int(q.C) for q in small], int(r.C)),
        fontsize=7.5, color=INK, va="center")
for q in small + [r]:
    ax.plot(q.C, ytop * 0.035, marker="v", color=RED, ms=5)
ax = axs[1]
bins = np.arange(0, W.V.max() + 2, 1) - 0.5
ax.hist(W.V, bins=bins, color=TEAL, alpha=0.9)
mV = NUM["world_med_V"]
ax.axvline(mV, color=RED, lw=1.4)
ax.text(mV + 1, ax.get_ylim()[1] * 0.92, f"median {mV:g}", color=RED, fontsize=8.5)
ax.set_xlabel("vowels in the inventory"); ax.set_title("Vowels", loc="left")
ax = axs[2]
tc = W["T"].clip(upper=8).value_counts().sort_index()
ax.bar(tc.index, tc.values, color=GOLD, width=0.75)
ax.set_xticks(range(0, 9)); ax.set_xticklabels([str(i) for i in range(8)] + ["8+"])
ax.set_xlabel("tones in the inventory"); ax.set_title("Tones", loc="left")
ax.text(0.97, 0.92, f"{NUM['world_tone_pct']:.0f}% of languages\nlist any tone", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
save(fig, "02_size_distributions.png")

# ---- Fig 3: most common phonemes, world vs India
fig, ax = plt.subplots(figsize=(10, 4.0))
x = np.arange(len(top))
ax.bar(x - 0.2, top.values, width=0.4, color=BLUE, label=f"all PHOIBLE languages (n = {nW})")
ax.bar(x + 0.2, [freq_in.get(p, 0) for p in top.index], width=0.4, color=RED, alpha=0.85, label=f"languages of India (n = {len(IN_)})")
ax.set_xticks(x); ax.set_xticklabels(top.index, fontsize=12)
ax.set_ylabel("% of languages with this phoneme"); ax.set_ylim(0, 105)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", ncol=2)
ax.set_title("The 25 most widespread phonemes (exact PHOIBLE symbols, one inventory per language)", loc="left")
save(fig, "03_common_phonemes.png")

# ---- Fig 4: India map by family + family counts
fig = plt.figure(figsize=(10, 5.6))
ax = fig.add_axes([0.0, 0.0, 0.6, 1.0])
box = (58, 104, 0, 46)
bgdots(ax, box, s=4)
for grp in FAMS:
    d = IN_[IN_.group == grp]
    ax.scatter(d.lon, d.lat, s=26, color=FAMCOL[grp], edgecolor="white", lw=0.6, zorder=3, label=f"{grp} ({len(d)})")
for gc, lab, dx, dy in [("hind1269", "Hindi", -6, 0), ("beng1280", "Bengali", -3, -2.6), ("assa1263", "Assamese", -2.5, 3.2),
                        ("tami1289", "Tamil", 0.8, -0.8), ("telu1262", "Telugu", 1.2, 0), ("mala1464", "Malayalam", -6.5, -0.8),
                        ("nucl1305", "Kannada", -6, 0.3), ("mara1378", "Marathi", -6, 0.5), ("sant1410", "Santali", -0.5, -2.2),
                        ("khas1269", "Khasi", -1, -2.0), ("mani1292", "Manipuri", 1.0, -1.4), ("buru1296", "Burushaski", 1.0, 0.8),
                        ("nort2646", "Pashto", -3, 1.6), ("uigh1240", "Uyghur", -2, -1.6), ("kham1282", "Khams Tibetan", -3, 1.2),
                        ("apuc1241", "Great Andamanese", -10.5, 0), ("dhiv1236", "Dhivehi", 0.8, 0)]:
    r = lang_row(gc)
    if r is not None and box[0] <= r.lon <= box[1] and box[2] <= r.lat <= box[3]:
        ax.annotate(lab, (r.lon, r.lat), xytext=(r.lon + dx, r.lat + dy), fontsize=7.5, color=INK, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
    elif r is not None:
        print("outside box", lab, r.lon, r.lat)
ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
ax.legend(loc="lower left", fontsize=8, title="PHOIBLE languages of India", title_fontsize=8)
ax.set_title("Languages of India in PHOIBLE, by family (Glottolog coordinates)", loc="left")
ax2 = fig.add_axes([0.7, 0.18, 0.29, 0.64])
gl_in = gl_lang[gl_lang.Glottocode.isin(india & spoken)].copy()
gl_in["group"] = gl_in.Glottocode.map(group_of)
yy = np.arange(len(FAMS))[::-1]
nall = [int((gl_in.group == f).sum()) for f in FAMS]
nph = [int((IN_.group == f).sum()) for f in FAMS]
for f, a in zip(FAMS, nall):
    NUM[f"glotto_india_{f.replace('-', '_')}"] = a
ax2.barh(yy, nall, color="#D7DEE8", height=0.62, label="all (Glottolog)")
ax2.barh(yy, nph, color=[FAMCOL[f] for f in FAMS], height=0.62, label="with PHOIBLE inventory")
for y, a, b in zip(yy, nall, nph):
    ax2.text(a + 3, y, f"{b} / {a}", va="center", fontsize=8, color=INK)
ax2.set_yticks(yy); ax2.set_yticklabels(FAMS)
ax2.set_xlim(0, max(nall) * 1.3)
ax2.grid(axis="y", visible=False)
ax2.set_xlabel("spoken languages of India in Glottolog")
ax2.set_title("Coverage by family", loc="left")
ax2.legend(loc="lower right", fontsize=7.5)
save(fig, "04_india_map.png")

# ---- Fig 5: retroflex world map
fig, ax = plt.subplots(figsize=(10, 4.6))
bgdots(ax)
no = W[~W.retro]
yes = W[W.retro]
ax.scatter(no.lon, no.lat, s=5, color="#8FA3B8", lw=0, zorder=2, label=f"no retroflex consonant ({len(no)})")
ax.scatter(yes.lon, yes.lat, s=9, color=RED, lw=0, zorder=3, label=f"has a retroflex consonant ({len(yes)})")
ax.set_xlim(-180, 190); ax.set_ylim(-58, 80); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=8.5, markerscale=2)
ax.set_title("Retroflex consonants (ʈ ɖ ɳ ɭ ʂ ɽ …) cluster in South Asia and Australia", loc="left")
save(fig, "05_retroflex_world.png")

# ---- Fig 6: India vs rest of world, by feature
fig, ax = plt.subplots(figsize=(9.5, 4.0))
y = np.arange(len(FEATS))[::-1]
ri = [R["india"][f] for f in FEATS]
rr = [R["row"][f] for f in FEATS]
ax.barh(y + 0.2, ri, height=0.38, color=RED, label=f"languages of India (n = {len(IN_)})")
ax.barh(y - 0.2, rr, height=0.38, color=BLUE, label=f"rest of the world (n = {len(ROW)})")
for f, yy_ in zip(FEATS, y):
    lo, hi = NUM[f"sens_india_{f}_lo"], NUM[f"sens_india_{f}_hi"]
    ax.plot([lo, hi], [yy_ + 0.2, yy_ + 0.2], color=INK, lw=1.2, solid_capstyle="butt")
    lo, hi = NUM[f"sens_row_{f}_lo"], NUM[f"sens_row_{f}_hi"]
    ax.plot([lo, hi], [yy_ - 0.2, yy_ - 0.2], color=INK, lw=1.2, solid_capstyle="butt")
    ax.text(max(R["india"][f], NUM[f"sens_india_{f}_hi"]) + 1.5, yy_ + 0.2, f"{R['india'][f]:.0f}%", va="center", fontsize=8, color=RED)
    ax.text(max(R["row"][f], NUM[f"sens_row_{f}_hi"]) + 1.5, yy_ - 0.2, f"{R['row'][f]:.0f}%", va="center", fontsize=8, color=BLUE)
ax.set_yticks(y); ax.set_yticklabels([FLAB[f] for f in FEATS])
ax.set_xlim(0, 108); ax.set_xlabel("% of languages")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
ax.set_title("India vs the rest of the world (black lines: range over inventory-choice rules)", loc="left")
save(fig, "06_india_vs_world.png")

# ---- Fig 7: heatmap by family within India
fig, ax = plt.subplots(figsize=(9.5, 3.4))
rowsH = FAMS + ["India (all)", "Rest of world"]
M = np.array([[R[r][f] for f in FEATS] for r in FAMS] + [[R["india"][f] for f in FEATS], [R["row"][f] for f in FEATS]])
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8.5, color="white" if M[i, j] > 60 else INK)
ns = [int((IN_.group == f).sum()) for f in FAMS] + [len(IN_), len(ROW)]
ax.set_yticks(range(len(rowsH))); ax.set_yticklabels([f"{r} (n={n})" for r, n in zip(rowsH, ns)])
ax.set_xticks(range(len(FEATS))); ax.set_xticklabels([FLAB[f].replace(" ", "\n", 1) for f in FEATS], fontsize=7.8)
ax.axhline(len(FAMS) - 0.5, color="white", lw=3)
ax.grid(False)
ax.set_title("% of languages with each property, by family (languages of India)", loc="left")
save(fig, "07_family_heatmap.png")

# ---- Fig 8: sizes India by family vs world
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"wspace": 0.25})
for ax, col, lab in [(axs[0], "C", "consonants"), (axs[1], "V", "vowels")]:
    data = [ROW[col].values] + [IN_[IN_.group == f][col].values for f in FAMS]
    labs = ["rest of\nworld"] + [f.replace("-", "-\n") if f == "Sino-Tibetan" else f.replace("Austroasiatic", "Austro-\nasiatic") for f in FAMS]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
    cols = ["#AEB9CB"] + [FAMCOL[f] for f in FAMS]
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    r2 = np.random.default_rng(3)
    for i, (d, c) in enumerate(zip(data, cols)):
        if i == 0:
            continue
        ax.scatter(i + 1 + r2.uniform(-0.18, 0.18, len(d)), d, s=9, color=c, lw=0, zorder=3)
    ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel(f"{lab} in the inventory")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Number of {lab}", loc="left")
save(fig, "08_sizes_by_family.png")
for f in FAMS:
    d = IN_[IN_.group == f]
    k = f.replace("-", "_")
    NUM[f"med_C_{k}"], NUM[f"med_V_{k}"] = float(np.median(d.C)), float(np.median(d.V))
    NUM[f"med_stop_{k}"] = float(np.median(d.n_stop))
NUM["med_stop_row"] = float(np.median(ROW.n_stop))

# ---- Fig 9: same language, different sources
CASE = [("hind1269", "Hindi"), ("beng1280", "Bengali"), ("assa1263", "Assamese"), ("mara1378", "Marathi"), ("tami1289", "Tamil"),
        ("telu1262", "Telugu"), ("nucl1305", "Kannada"), ("mala1464", "Malayalam"), ("sant1410", "Santali"), ("mund1320", "Mundari"),
        ("khas1269", "Khasi"), ("mani1292", "Manipuri (Meitei)"), ("bodo1269", "Bodo")]
SRC_COL = {"ph": BLUE, "ea": TEAL, "ra": RED, "upsid": GOLD, "spa": "#7B5EA7"}
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
yy = np.arange(len(CASE))[::-1]
for ax, col in [(axs[0], "C"), (axs[1], "V")]:
    for y, (gc, lab) in zip(yy, CASE):
        d = inv[inv.lang == gc]
        ax.plot([d[col].min(), d[col].max()], [y, y], color="#D7DEE8", lw=3, zorder=1)
        seen = Counter()
        for _, r in d.iterrows():
            k = seen[r[col]]
            seen[r[col]] += 1
            ax.scatter(r[col], y + 0.22 * k, s=42, color=SRC_COL.get(r.Source, SLATE), edgecolor="white", lw=0.7, zorder=3)
        ci = chosen[gc]
        ax.scatter(I.loc[ci, col], y, s=110, facecolor="none", edgecolor=INK, lw=1.2, zorder=4)
    ax.set_xlabel("consonants" if col == "C" else "vowels")
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(yy); axs[0].set_yticklabels([f"{lab} ({(inv.lang == gc).sum()})" for gc, lab in CASE])
for s, c in SRC_COL.items():
    axs[1].scatter([], [], s=40, color=c, label=s)
axs[1].scatter([], [], s=90, facecolor="none", edgecolor=INK, label="chosen (median)")
fig.legend(*axs[1].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=8.5)
axs[0].set_title("Consonants", loc="left")
axs[1].set_title("Vowels", loc="left")
save(fig, "09_sources_disagree.png")
for gc, lab in CASE:
    d = inv[inv.lang == gc]
    k = lab.split(" ")[0].lower()
    NUM[f"case_{k}_ninv"] = len(d)
    NUM[f"case_{k}_Cmin"], NUM[f"case_{k}_Cmax"] = int(d.C.min()), int(d.C.max())
    NUM[f"case_{k}_Vmin"], NUM[f"case_{k}_Vmax"] = int(d.V.min()), int(d.V.max())
    print("case", lab, d[["InventoryID", "Source", "C", "V", "T"]].values.tolist(), "chosen", chosen[gc])

# ====================================================================== demo data (inventories.js)
SRC_NAME = {"ph": "PHOIBLE contributors (from grammars, Moran 2012 and later)", "ea": "Eurasian Phonological Inventories (Nikolaev et al. 2015)",
            "ra": "Ramaswami 1999, Common Linguistic Features in Indian Languages: Phonetics (CIIL)",
            "upsid": "UPSID (Maddieson 1984; Maddieson & Precoda 1990)", "spa": "Stanford Phonology Archive (Crothers et al. 1979)",
            "saphon": "SAPhon (Michael et al.)", "aa": "Alphabets des langues africaines (Chanard 2006; Hartell 1993)",
            "er": "Australian inventories (Round 2019)"}
WORLD_DEMO = [("stan1293", "English"), ("mand1415", "Mandarin Chinese"), ("nucl1643", "Japanese"), ("stan1318", "Standard Arabic"),
              ("hawa1245", "Hawaiian"), ("roto1249", "Rotokas"), ("huaa1248", "East Taa (!Xóõ)"), ("stan1288", "Spanish"),
              ("swah1253", "Swahili"), ("stan1295", "German"), ("nepa1254", "Nepali"), ("sinh1246", "Sinhala")]
demo_langs = sorted(IN_.lang.tolist(), key=lambda gc: (FAMS.index(group_of(gc)), g.Name.get(gc)))
segd = {iid: d for iid, d in seg.groupby("InventoryID")}


def pack(iid):
    d = segd[iid]
    r = I.loc[iid]
    cons, vow, dip, ton, oth = [], [], [], [], []
    for _, s in d.iterrows():
        m = 1 if s.marg else 0
        if s.SegmentClass == "consonant":
            c = cls.get(s.Phoneme)
            if c:
                cons.append([s.Phoneme, MANNERS.index(c[0]), PLACES.index(c[1]), m])
            else:
                oth.append([s.Phoneme, m])
        elif s.SegmentClass == "vowel":
            v = classify_vowel(s.Phoneme)
            if v:
                vow.append([s.Phoneme, v[0], v[1], m])
            else:
                dip.append([s.Phoneme, m])
        else:
            ton.append([s.Phoneme, m])
    dial = r.SpecificDialect if isinstance(r.SpecificDialect, str) else ""
    ff = feat[iid]
    return {"id": int(iid), "s": r.Source, "n": r.LanguageName, "d": dial, "C": int(r.C), "V": int(r.V), "T": int(r["T"]),
            "f": [int(ff[k]) for k in FEATS], "c": cons, "v": vow, "x": dip, "t": ton, "o": oth}


demo = {"places": PLACES, "manners": MANNERS, "src": SRC_NAME, "feats": [FLAB[k] for k in FEATS], "langs": []}
for gc in demo_langs:
    ids = inv[inv.lang == gc].sort_values(["pref", "InventoryID"]).InventoryID.tolist()
    demo["langs"].append({"g": gc, "name": g.Name.get(gc), "fam": group_of(gc), "family": family_of(gc), "ctry": g.Countries.get(gc),
                          "pick": int(chosen[gc]), "inv": [pack(i) for i in ids]})
for gc, lab in WORLD_DEMO:
    ids = inv[inv.lang == gc].sort_values(["pref", "InventoryID"]).InventoryID.tolist()
    if not ids:
        print("world demo missing", lab)
        continue
    ids = ids[:6]
    pick = chosen[gc] if chosen[gc] in ids else ids[0]
    demo["langs"].append({"g": gc, "name": lab, "fam": "World", "family": family_of(gc), "ctry": g.Countries.get(gc),
                          "pick": int(pick), "inv": [pack(i) for i in ids]})
js = "window.PHOIBLE_DEMO=" + json.dumps(demo, ensure_ascii=False, separators=(",", ":")) + ";\n"
open(os.path.join(HERE, "inventories.js"), "w", encoding="utf8").write(js)
NUM["demo_kb"] = round(len(js.encode("utf8")) / 1024, 0)
NUM["demo_n_lang"] = len(demo["langs"])
NUM["demo_n_inv"] = sum(len(l["inv"]) for l in demo["langs"])
print("inventories.js", NUM["demo_kb"], "KB", NUM["demo_n_lang"], "languages", NUM["demo_n_inv"], "inventories")

# ====================================================================== opening half (sections 01-03): ksp speech, PHOIBLE features
import make_intro  # noqa: E402

NUM = make_intro.main(NUM, ph, picks={gc: int(chosen[gc]) for gc in ("stan1293", "hind1269")})

json.dump(NUM, open(os.path.join(HERE, "numbers.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
print("wrote numbers.json with", len(NUM), "numbers")
