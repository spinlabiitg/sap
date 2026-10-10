"""Find maths written as plain text/HTML (outside KaTeX delimiters) in the site's pages.

    python tools/scan_math.py [page.html ...]       (default: every page source under docs/)

Prints, per page, text snippets that look like formulas: <sub>/<sup>, unicode maths symbols or sub/superscript
digits, and "symbol = value" patterns. Script, style, SVG and KaTeX-delimited spans are ignored. Heuristic:
review the hits by eye; units such as "16 kHz" or ranges are not flagged.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")

PAT = re.compile(
    r"<su[bp]>|[×≈≤≥±√∑∏∫∞∂∆Δ·⋅πωθφλρσμτ²³⁴₀₁₂₃₄₅ₙₖₚ]|"            # markup or maths symbols
    r"\b[A-Za-z]{1,3}(?:\[[a-z+\-−0-9]+\])?\s*(?:=|≈)\s*[-−]?\(?[A-Za-z0-9]|"   # x = 1, F0 = 120, e[n] = …
    r"\b(?:log|ln|sin|cos|exp)\s*\(|\b[a-z]\[[a-z][+\-−]?[0-9a-z]*\]"           # functions, x[n]
)


def text_outside_math(s):
    s = re.sub(r"<(script|style|svg|code|pre)\b.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
    s = re.sub(r"\\\(.*?\\\)", " ", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r'\b(?:href|src|class|id|style|alt|title|aria-label|content|data-[\w-]+)="[^"]*"', " ", s)
    return s


def scan(path):
    s = text_outside_math(open(path, encoding="utf8").read())
    hits = []
    for line in s.splitlines():
        plain = re.sub(r"<(?!/?su[bp]>)[^>]+>", " ", line)        # keep <sub>/<sup> visible, drop other tags
        if PAT.search(plain):
            hits.append(re.sub(r"\s+", " ", plain).strip()[:150])
    return hits


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(
        f for f in glob.glob(os.path.join(DOCS, "**", "*.html"), recursive=True)
        if not os.path.exists(os.path.join(os.path.dirname(f), os.path.basename(f)[:-5], "page_src.html")))
    total = 0
    for f in files:
        h = scan(f)
        total += len(h)
        print(f"\n=== {os.path.relpath(f, DOCS)}: {len(h)} line(s)")
        for x in h:
            print("   ", x)
    print(f"\n{total} candidate line(s)")
