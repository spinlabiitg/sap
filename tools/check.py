"""Pre-publish checks for the Speech and Audio Processing site.

    python tools/check.py

1. Broken local links: every href/src in docs/**/*.html that points to a local file must exist.
2. TIMIT gate: TIMIT (LDC) may only be used for aggregate statistics. This lists every published file
   (audio, embedded-signal JS, figures) produced by a generator script that reads TIMIT, and every page
   that still mentions TIMIT, so they can be reviewed before pushing. Files reviewed as safe (ARCTIC,
   synthetic, or TIMIT aggregate-only) are recorded in tools/timit_allow.txt with a content hash, so a
   file that is regenerated with different content is flagged again for review:

       python tools/check.py --approve speech/codecs             # after reviewing that folder by eye

Exit code 1 if anything needs attention.
"""
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
ALLOW_F = os.path.join(ROOT, "tools", "timit_allow.txt")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


allow = {}
if os.path.exists(ALLOW_F):
    for line in open(ALLOW_F, encoding="utf8"):
        if line.strip() and not line.startswith("#"):
            path, _, h = line.strip().partition(" ")
            allow[path] = h.strip()

if len(sys.argv) == 3 and sys.argv[1] == "--approve":
    folder = sys.argv[2].strip("/\\")
    for d, _, fs in os.walk(os.path.join(DOCS, folder)):
        for f in fs:
            if f.endswith((".wav", ".mp3", ".js", ".png", ".json")):
                full = os.path.join(d, f)
                allow[os.path.relpath(full, DOCS).replace("\\", "/")] = sha(full)
    with open(ALLOW_F, "w", encoding="utf8") as fh:
        fh.write("# docs-relative path + sha256 prefix of files reviewed as TIMIT-safe (ARCTIC / synthetic / aggregate-only)\n")
        for k in sorted(allow):
            fh.write(f"{k} {allow[k]}\n")
    print(f"approved {folder}: {len(allow)} entries in timit_allow.txt")
    sys.exit(0)

bad = 0
# ---------------------------------------------------------------- 1. links
for d, _, fs in os.walk(DOCS):
    for f in fs:
        if not f.endswith(".html") or f.endswith("_src.html"):   # build templates are checked through their output
            continue
        p = os.path.join(d, f)
        s = open(p, encoding="utf8").read()
        for url in re.findall(r'(?:href|src)="([^"]+)"', s):
            if re.match(r"^(https?:|mailto:|data:|#|javascript:)", url) or "${" in url:
                continue
            target = os.path.normpath(os.path.join(d, url.split("#")[0].split("?")[0]))
            if not os.path.exists(target):
                print(f"BROKEN  {os.path.relpath(p, DOCS)} -> {url}")
                bad += 1

# ---------------------------------------------------------------- 2. TIMIT
gens = []
for d, _, fs in os.walk(DOCS):
    for f in fs:
        if f.endswith(".py"):
            s = open(os.path.join(d, f), encoding="utf8").read()
            if re.search(r"timit", s, re.I):
                gens.append(os.path.join(d, f))
for g in gens:
    gd = os.path.dirname(g)
    for d, _, fs in os.walk(gd):
        for f in fs:
            if f.endswith((".wav", ".mp3", ".js", ".png", ".json")):
                rp = os.path.relpath(os.path.join(d, f), DOCS).replace("\\", "/")
                if allow.get(rp) != sha(os.path.join(d, f)):
                    print(f"TIMIT?  {rp}   (generator {os.path.relpath(g, DOCS)} reads TIMIT)")
                    bad += 1
for d, _, fs in os.walk(DOCS):
    for f in fs:
        if f.endswith(".html"):
            n = len(re.findall(r"TIMIT", open(os.path.join(d, f), encoding="utf8").read()))
            if n:
                print(f"MENTION {os.path.relpath(os.path.join(d, f), DOCS)}: 'TIMIT' x{n}")
print("OK" if not bad else f"{bad} item(s) need attention")
sys.exit(1 if bad else 0)
