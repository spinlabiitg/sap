"""Build the Speech and Audio Processing site from site.json.

    python tools/build.py

* writes docs/index.html (a minimal home page: one row per module, then updates and credits),
* injects the shared navigation bar, stylesheet link and footer into every page listed in site.json
  (between <!--ai:...--> markers, so re-running replaces rather than duplicates),
* writes docs/.nojekyll so GitHub Pages serves the files as they are.

To add a page: put the HTML file under docs/, add an entry to site.json, run this script.
Pages with "chrome": false (e.g. full-screen slide decks) are listed but left untouched.
"""
import html
import json
import os
import posixpath
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
SITE = json.load(open(os.path.join(ROOT, "site.json"), encoding="utf8"))
esc = html.escape
import hashlib  # noqa: E402
CSS_V = hashlib.sha256(open(os.path.join(DOCS, "assets", "site.css"), "rb").read()).hexdigest()[:8]   # cache-buster


def rel(from_file, to_file):
    """Relative URL from one docs/ file to another."""
    return posixpath.relpath(to_file, posixpath.dirname(from_file) or ".")


def all_pages():
    """(group_label, page) for every page, in reading order."""
    for m in SITE["modules"]:
        for p in m["pages"]:
            yield f"Module {m['n']}", m, p
    for p in SITE["extras"]["pages"]:
        yield SITE["extras"]["title"], None, p


def exists(p):
    return os.path.exists(os.path.join(DOCS, p["file"]))


def readable(p):
    return exists(p) and p.get("status") != "planned"


SEQ = [(g, m, p) for g, m, p in all_pages() if readable(p) and p.get("chrome", True)]


def nav_html(cur):
    i = next(k for k, (_, _, p) in enumerate(SEQ) if p["file"] == cur)
    g, m, p = SEQ[i]
    home = rel(cur, "index.html")
    where = f"{esc(g)} &middot; {esc(m['title'] if m else p['title'])}"
    prev_ = SEQ[i - 1] if i > 0 else None
    next_ = SEQ[i + 1] if i + 1 < len(SEQ) else None
    link = lambda t, arrow, cls: (f'<a class="ai-step {cls}" href="{rel(cur, t[2]["file"])}" title="{esc(t[2]["title"])}">{arrow}</a>'
                                  if t else f'<span class="ai-step {cls} ai-off">{arrow}</span>')
    items = []
    for gg, mm, pp in all_pages():
        if not readable(pp):
            continue
        label = f"{mm['n']}. {pp['title']}" if mm else pp["title"]
        here = ' aria-current="page"' if pp["file"] == cur else ""
        items.append(f'<li><a href="{rel(cur, pp["file"])}"{here}>{esc(label)}</a></li>')
    return (f'<nav class="ai-nav" aria-label="Course navigation"><div class="ai-nav-in">'
            f'<a class="ai-brand" href="{home}" title="{esc(SITE["title"])}"><span class="ai-logo" aria-hidden="true">◖◗</span>'
            f'<span class="ai-brand-full">{esc(SITE["title"])}</span><span class="ai-brand-short">{esc(SITE.get("abbr", SITE["title"]))}</span></a>'
            f'<span class="ai-where">{where}</span>'
            f'<details class="ai-menu"><summary>All pages</summary><ul>{"".join(items)}</ul></details>'
            f'{link(prev_, "← prev", "ai-prev")}{link(next_, "next →", "ai-next")}'
            f'</div></nav>')


def licence_html():
    """One line on every page: content CC BY-NC 4.0, code MIT (details in the repository's LICENSE files)."""
    repo = SITE.get("repo")
    content = '<a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noopener license">CC BY-NC 4.0</a>'
    code = f'<a href="{esc(repo)}/blob/main/LICENSE" target="_blank" rel="noopener">MIT</a>' if repo else "MIT"
    more = f' (<a href="{esc(repo)}/blob/main/LICENSE-content.md" target="_blank" rel="noopener">details</a>)' if repo else ""
    return f'<span class="ai-lic">Text, figures and audio: {content} &middot; code: {code}{more}</span>'


def foot_html(cur):
    ins = SITE.get("instructor")
    who = (f'Instructor: <a href="{esc(ins["url"])}" target="_blank" rel="noopener">{esc(ins["name"])}</a>, {esc(ins["affiliation"])}'
           if ins else esc(SITE["course"]))
    return (f'<footer class="ai-foot"><a href="{rel(cur, "index.html")}">{esc(SITE["title"])}</a> &middot; {who} '
            f'&middot; <a href="{rel(cur, "index.html")}#credits">data credits</a>{licence_html()}</footer>')


def inject(path, key, block, where):
    s = open(path, encoding="utf8").read()
    tag = f"<!--ai:{key}-->", f"<!--/ai:{key}-->"
    new = tag[0] + block + tag[1]
    if tag[0] in s:
        s = re.sub(re.escape(tag[0]) + ".*?" + re.escape(tag[1]), lambda _: new, s, flags=re.S)
    elif where == "head":
        s = s.replace("</head>", new + "\n</head>", 1)
    elif where == "body":
        s = re.sub(r"(<body[^>]*>)", lambda mm: mm.group(1) + "\n" + new, s, count=1)
    else:
        s = s[::-1].replace("</body>"[::-1], ("\n" + new + "\n</body>")[::-1], 1)[::-1]
    open(path, "w", encoding="utf8").write(s)


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def instructor_html(cls="ai-instr"):
    ins = SITE.get("instructor")
    if not ins:
        return ""
    return (f'<p class="{cls}">Instructor: <a href="{esc(ins["url"])}" target="_blank" rel="noopener">{esc(ins["name"])}</a>, '
            f'{esc(ins["affiliation"])}</p>')


def home():
    """Minimal home page: a header, then one panel with a row per module (in module order), updates and credits.
    A new module in site.json simply becomes another row."""
    def note(st):
        return "" if st == "live" else f' <span class="ai-note">({esc(SITE["status_legend"][st]).lower()})</span>'

    def row(num, title, summary, pages):
        pages = [p for p in pages if p.get("home", True)]       # "home": false keeps a page live but off the home page
        live = [p for p in pages if readable(p)]
        href = live[0]["file"] if live else None
        head = f'<a href="{href}">{esc(title)}</a>' if href else f'<span>{esc(title)}</span>'
        sub = ""
        if len(pages) > 1:
            sub = '<p class="ai-row-pages">' + " &middot; ".join(
                (f'<a href="{q["file"]}">{esc(q["title"])}</a>' if readable(q) else esc(q["title"])) + note(q["status"]) for q in pages) + "</p>"
        status = note(pages[0]["status"]) if len(pages) == 1 and pages else ""
        return (f'<li class="ai-row"><span class="ai-num">{num}</span><div><h3>{head}{status}</h3>'
                f'<p>{esc(summary)}</p>{sub}</div></li>')

    mods = sorted(SITE["modules"], key=lambda m: m["n"])
    rows = "".join(row(f'{m["n"]:02d}', m["title"], m.get("short", m["summary"]), m["pages"]) for m in mods)
    ex = SITE["extras"]
    tools = ""
    if ex.get("show_on_home", True):
        trows = "".join(row("&middot;", p["title"], p.get("blurb", ""), [p]) for p in ex["pages"])
        tools = f'<h2 id="toolbox">{esc(ex["title"])}</h2><ul class="ai-panel">{trows}</ul>'
    ups = "".join(f'<li><time>{esc(u["date"])}</time> {esc(u["text"])}</li>' for u in SITE["updates"][:5])
    cred = "".join(f"<li>{esc(c)}</li>" for c in SITE["credits"])
    ins = SITE.get("instructor", {})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(SITE["title"])}</title>
<meta name="description" content="{esc(SITE["tagline"])}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@600&family=Inter:wght@400;500;600&display=swap">
<link rel="stylesheet" href="assets/site.css?v={CSS_V}">
</head>
<body class="ai-home">
<main>
  <header>
    <h1>{esc(SITE["title"])}{f' <span class="ai-abbr">({esc(SITE["abbr"])})</span>' if SITE.get("abbr") else ""}</h1>
    <p class="ai-lede">{esc(SITE["tagline"])}</p>
    {instructor_html()}
  </header>
  <h2 id="modules">Modules</h2>
  <ul class="ai-panel">{rows}</ul>
  {tools}
  <h2 id="updates">Updates</h2>
  <ul class="ai-updates">{ups}</ul>
  <details class="ai-credits-box" id="credits"><summary>Data and credits</summary><ul class="ai-credits">{cred}</ul></details>
</main>
<footer class="ai-foot">{esc(SITE["title"])} &middot; {esc(SITE["course"])} &middot; <a href="{esc(ins.get("url", "#"))}" target="_blank" rel="noopener">{esc(ins.get("name", ""))}</a>{licence_html()}</footer>
<script>/* open the credits panel when arriving via #credits (pages link to it) */
(function () {{ function o() {{ if (location.hash === '#credits') document.getElementById('credits').open = true; }} o(); window.addEventListener('hashchange', o); }})();</script>
</body>
</html>
"""


def main():
    os.makedirs(DOCS, exist_ok=True)
    open(os.path.join(DOCS, ".nojekyll"), "w").close()
    open(os.path.join(DOCS, "index.html"), "w", encoding="utf8").write(home())
    n = 0
    for _, _, p in SEQ:
        f = os.path.join(DOCS, p["file"])
        inject(f, "head", f'<link rel="stylesheet" href="{rel(p["file"], "assets/site.css")}?v={CSS_V}">', "head")
        inject(f, "nav", nav_html(p["file"]), "body")
        inject(f, "foot", foot_html(p["file"]), "foot")
        n += 1
    missing = [p["file"] for _, _, p in all_pages() if p.get("status") != "planned" and not exists(p)]
    print(f"home + {n} pages built" + (f"; MISSING: {missing}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
