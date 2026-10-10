"""Assemble ../decibels.html from page_src.html: shared head/styles from linear_prediction.html, quoted numbers from
numbers.json (written by make_figures.py), figure captions numbered in order of appearance."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
src = open(os.path.join(HERE, "page_src.html"), encoding="utf8").read()
lp = open(os.path.join(ROOT, "linear_prediction.html"), encoding="utf8").read()
head = lp[lp.index('<link rel="stylesheet" href="../assets/katex'):lp.index('<style>')]
style = lp[lp.index('<style>') + len('<style>'):lp.index('</style>')]
num = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf8"))

out = src.replace("<!--HEAD-->", head.strip()).replace("<!--STYLE-->", style.strip("\n"))
out = re.sub(r"<!--N:(\w+)-->", lambda m: f"{num[m.group(1)]:.1f}", out)
n = [0]
def renumber(m):
    n[0] += 1
    return f"<b>Figure {n[0]}.</b>"
out = re.sub(r"<b>Figure \d+\.</b>", renumber, out)
open(os.path.join(ROOT, "decibels.html"), "w", encoding="utf8").write(out)
print("wrote decibels.html")
