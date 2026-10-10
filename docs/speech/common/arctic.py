"""
Reader for the CMU ARCTIC 'ksp' voice (Indian-English male, 16 kHz), used for every per-utterance speech
example on this site.

Source: CMU ARCTIC databases, Carnegie Mellon University (J. Kominek and A. W. Black). Free for any use under
the CMU licence (copyright notice kept, authors credited, modifications marked). Phone labels were produced
automatically by an HMM labeller (EHMM) and were not hand-corrected, so boundaries can be off by a frame or two.

    from arctic import read_wav, phones, pitchmarks, text, find, IPA

The corpus itself lives outside the repository; set ARCTIC_KSP to override the default path.
"""
import os
import re

import numpy as np
import soundfile as sf

ROOT = os.environ.get("ARCTIC_KSP", r"C:\Users\IITG\Documents\tommy_assistant\research\dBase\CMU_ARCTIC\full\cmu_us_ksp_arctic")
SR = 16000
SPEAKER = "ksp (CMU ARCTIC, Indian-English male)"

# ARPAbet (as used by the Festival labels, lower case) -> IPA
IPA = {
    "aa": "ɑ", "ae": "æ", "ah": "ʌ", "ao": "ɔ", "aw": "aʊ", "ax": "ə", "ay": "aɪ", "eh": "ɛ", "er": "ɚ", "ey": "eɪ",
    "ih": "ɪ", "iy": "i", "ow": "oʊ", "oy": "ɔɪ", "uh": "ʊ", "uw": "u",
    "b": "b", "ch": "tʃ", "d": "d", "dh": "ð", "f": "f", "g": "ɡ", "hh": "h", "jh": "dʒ", "k": "k", "l": "l", "m": "m",
    "n": "n", "ng": "ŋ", "p": "p", "r": "ɹ", "s": "s", "sh": "ʃ", "t": "t", "th": "θ", "v": "v", "w": "w", "y": "j",
    "z": "z", "zh": "ʒ", "pau": "(pause)", "ssil": "(silence)",
}
VOWELS = {"aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}
MONOPHTHONGS = {"aa", "ae", "ah", "ao", "eh", "ih", "iy", "uh", "uw"}
FRICATIVES = {"f", "v", "th", "dh", "s", "z", "sh", "zh", "hh"}
STOPS = {"p", "b", "t", "d", "k", "g"}
NASALS = {"m", "n", "ng"}


def utterances():
    return sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "wav")) if f.endswith(".wav"))


def read_wav(utt):
    x, sr = sf.read(os.path.join(ROOT, "wav", utt + ".wav"), dtype="float64")
    assert sr == SR
    return x


def phones(utt):
    """[(start_s, end_s, phone), ...] from the Festival .lab file (each line gives the END time of a phone)."""
    out, t0, body = [], 0.0, False
    for line in open(os.path.join(ROOT, "lab", utt + ".lab"), encoding="latin-1"):
        if line.strip() == "#":
            body = True
            continue
        if not body or not line.strip():
            continue
        end, _, ph = line.split()[:3]
        end = float(end)
        out.append((t0, end, ph))
        t0 = end
    return out


def pitchmarks(utt):
    """Glottal-closure instants (s) from the .pm track (EST format; voiced and unvoiced marks both present)."""
    ts, body = [], False
    for line in open(os.path.join(ROOT, "pm", utt + ".pm"), encoding="latin-1"):
        if line.startswith("EST_Header_End"):
            body = True
            continue
        if body and line.strip():
            ts.append(float(line.split()[0]))
    return np.array(ts)


_TEXT = None


def text(utt):
    global _TEXT
    if _TEXT is None:
        _TEXT = {}
        for line in open(os.path.join(ROOT, "etc", "txt.done.data"), encoding="latin-1"):
            m = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)', line.strip())
            if m:
                _TEXT[m.group(1)] = m.group(2)
    return _TEXT[utt]


def find(phone, min_dur=0.0, limit=None):
    """All occurrences of a phone: [(utt, start, end), ...], longest first."""
    hits = []
    for u in utterances():
        for s, e, p in phones(u):
            if p == phone and e - s >= min_dur:
                hits.append((u, s, e))
    hits.sort(key=lambda h: h[1] - h[2])
    return hits[:limit] if limit else hits


def segment(utt, start, end, pad=0.0):
    x = read_wav(utt)
    a, b = max(0, int((start - pad) * SR)), min(len(x), int((end + pad) * SR))
    return x[a:b]


CREDIT = ("Speech: CMU ARCTIC speaker ksp (Indian English), Carnegie Mellon University; "
          "excerpts trimmed/resampled for this site.")
