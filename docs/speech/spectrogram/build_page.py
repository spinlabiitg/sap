"""Assemble ../spectrogram.html from page_src.html: shared head/styles from linear_prediction.html, quoted numbers from
numbers.json (written by make_figures.py), figure captions numbered in order of appearance.

Placeholders:  <!--N:key-->        integer-valued numbers as integers, others with one decimal
               <!--N:key:.2f-->    any Python format spec
               <!--N:key:sci-->    a x 10^e (for tiny errors)
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
src = open(os.path.join(HERE, "page_src.html"), encoding="utf8").read()
lp = open(os.path.join(ROOT, "linear_prediction.html"), encoding="utf8").read()
head = lp[lp.index('<link rel="stylesheet" href="../assets/katex'):lp.index('<style>')]
style = lp[lp.index('<style>') + len('<style>'):lp.index('</style>')]
num = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf8"))


def fmt(m):
    v, spec = num[m.group(1)], m.group(2)
    if spec and spec.startswith("abs"):                 # e.g. <!--N:key:abs.0f--> for "33 dB weaker"
        v, spec = abs(v), spec[3:]
    if spec == "sci":
        e = math.floor(math.log10(abs(v)))
        return rf"\({v / 10 ** e:.1f} \times 10^{{{e}}}\)"          # KaTeX (auto-render)
    if spec:
        s = format(v, spec)
    elif float(v).is_integer():
        s = str(int(v))
    else:
        s = f"{v:.1f}"
    return s.replace("-", "−")


sys.path.insert(0, os.path.join(ROOT, "common"))
import arctic  # noqa: E402

out = src.replace("<!--HEAD-->", head.strip()).replace("<!--STYLE-->", style.strip("\n")).replace("<!--CREDIT-->", arctic.CREDIT)
out = re.sub(r"<!--N:(\w+)(?::([^>]+?))?-->", fmt, out)
missing = re.findall(r"<!--N:[^>]*-->", out)
assert not missing, missing
n = [0]


def renumber(m):
    n[0] += 1
    return f"<b>Figure {n[0]}.</b>"


out = re.sub(r"<b>Figure \d+\.</b>", renumber, out)
open(os.path.join(ROOT, "spectrogram.html"), "w", encoding="utf8").write(out)
print(f"wrote spectrogram.html ({n[0]} figures)")
