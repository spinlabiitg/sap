"""Assemble the two Module-2 pages from their templates in this folder:

    page_src.html         -> ../sounds_language.html        (page 1: The sounds of language)
    inventories_src.html  -> ../phoneme_inventories.html    (page 2: Phoneme inventories: India and the world)

Shared head/styles are copied from linear_prediction.html, quoted numbers come from numbers.json (written by
make_figures.py and make_intro.py), and each page numbers its own figure captions from 1 in order of appearance.

Placeholders:  <!--N:key-->    integers print as integers, other numbers with one decimal, strings as they are
               <!--N:key:0-->  force the number of decimals
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PAGES = [("page_src.html", "sounds_language.html"),
         ("inventories_src.html", "phoneme_inventories.html")]

lp = open(os.path.join(ROOT, "linear_prediction.html"), encoding="utf8").read()
head = lp[lp.index('<link rel="stylesheet" href="../assets/katex'):lp.index('<style>')]
style = lp[lp.index('<style>') + len('<style>'):lp.index('</style>')]
num = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf8"))


def fill(m):
    v = num[m.group(1)]
    if m.group(2) is not None:
        return f"{float(v):.{int(m.group(2))}f}"
    if isinstance(v, str):
        return v
    if float(v).is_integer():
        return f"{int(v):,}".replace(",", "&nbsp;") if abs(v) >= 10000 else str(int(v))
    return f"{v:.1f}"


for src_name, out_name in PAGES:
    src = open(os.path.join(HERE, src_name), encoding="utf8").read()
    out = src.replace("<!--HEAD-->", head.strip()).replace("<!--STYLE-->", style.strip("\n"))
    out = re.sub(r"<!--N:([^:>]+?)(?::(\d))?-->", fill, out)
    assert "<!--N:" not in out, f"unfilled number placeholder in {src_name}"
    n = [0]

    def renumber(m):
        n[0] += 1
        return f"<b>Figure {n[0]}.</b>"

    out = re.sub(r"<b>Figure \d+\.</b>", renumber, out)
    open(os.path.join(ROOT, out_name), "w", encoding="utf8").write(out)
    print("wrote", out_name, "with", n[0], "figures")
