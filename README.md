# Speech and Audio Processing

Interactive course notes on speech and audio (IIT Guwahati): intuition first, then figures, listening examples
and live in-browser demos. The site is plain static HTML served by GitHub Pages from `docs/`; no build step runs on GitHub.

## Layout

```
site.json          the manifest: modules, pages, status, updates, credits (edit this to add or reorder pages)
tools/build.py     generates docs/index.html and injects the shared navigation bar and footer into every page
tools/check.py     pre-publish checks: broken local links, and the TIMIT gate (see "Data" below)
tools/scan_math.py    lists maths still written as plain text (outside \( \) / $$ $$)
docs/              the published site
  index.html       generated home page (do not edit by hand)
  assets/site.css  shared navigation, footer and home-page styles (namespaced .ai-*)
  assets/katex/    KaTeX 0.16.11 (MIT), served locally so equations render offline too
  speech/          the speech tutorials; each page has a folder with its figures, audio and generator script
  convolution/     the convolution / room-reverb tutorial
  data/            source recordings used by several generators
```

## Modules

| # | Module | Page(s) |
|---|---|---|
| 1 | Seeing sound: the spectrogram | `speech/spectrogram.html` |
| 2 | The sounds of language | `speech/sounds_language.html`, `speech/phoneme_inventories.html` (not listed on the home page: `"home": false`) |
| 3 | A buzz and a tube: the source–filter model | `speech/source_filter.html`, `speech/vocal_tract.html` |
| 4 | Guessing the next sample: correlation, prediction and AR models | `speech/linear_prediction.html` |
| 5 | Linear prediction for speech | `speech/ar_model.html` |
| 6 | Speech codecs | `speech/codecs.html` |
| – | Toolbox (in the page menu; hidden on the home page via `extras.show_on_home`) | `speech/decibels.html`, `convolution/index.html` |

## Working on the site

```bash
# preview locally
python -m http.server 8940 --directory docs
# after editing site.json or any page
python tools/build.py
python tools/check.py
```

* **Add a page:** put the HTML file (and its asset folder) under `docs/`, add an entry to `site.json` with a status
  (`live`, `update` or `planned`), then run `tools/build.py`. A new module becomes a new row on the home page: give it a number `n`,
  a `title`, a one-line `short` description (shown on the home page) and a longer `summary`, and add a line to `updates`.
  `"home": false` on a page keeps it live and in the page menu but leaves it off the home page.
* **Regenerate a page's figures or audio:** run its generator from its own folder, e.g.
  `cd docs/speech/codecs && python make_assets.py`. Some pages are assembled from a template:
  `python docs/speech/codecs/build_page.py` (and `decibels/build_page.py`) rewrite the page, so run
  `tools/build.py` afterwards to put the navigation back.
* Generators need Python with numpy, scipy, pandas, matplotlib, librosa and soundfile, plus local copies of the datasets below
  (they are not part of the repository). Shared reader for the ARCTIC voice: `docs/speech/common/arctic.py` (set `ARCTIC_KSP` to its folder).

## Data

* **CMU ARCTIC, speaker ksp** (Indian English, Carnegie Mellon University): free for any use under its licence; keep the
  copyright notice (`docs/speech/common/CMU_ARCTIC_COPYING.txt`) and credit the authors; excerpts on this site are trimmed and
  some are processed (marked on each page).
* **PHOIBLE 2.0** (Moran & McCloy, 2019) and **Glottolog** (Hammarström et al.): CC BY 4.0. Raw files stay outside the repository; only derived figures and a compact inventory subset are published.
* **TIMIT** (LDC93S1) is licensed by the LDC and is **not redistributable**. It may only be used for aggregate statistics
  (for example average vowel formants). No TIMIT audio, embedded excerpts or per-utterance figures may be published.
  `tools/check.py` lists every file produced by a generator that reads TIMIT. After reviewing a folder by eye, clear it with
  `python tools/check.py --approve <docs-relative folder>`; approvals store a content hash, so a regenerated file is flagged
  again. **Do not push unless `tools/check.py` ends with OK.**
* `docs/data/real_sample_speech.wav`: an Indian-English recording provided by the course instructor.

## Licence

* **Code** (`tools/`, the `*.py` generators and the pages' JavaScript): [MIT](LICENSE).
* **Content** (text, figures, diagrams, quizzes and course audio, including `docs/data/real_sample_speech.wav`):
  [CC BY-NC 4.0](LICENSE-content.md): reuse and adapt with credit, not for commercial purposes.
* Third-party material keeps its own licence (CMU ARCTIC, PHOIBLE, Glottolog, KaTeX; TIMIT is not included): see
  [LICENSE-content.md](LICENSE-content.md). Every page's footer states the licence.

## Writing equations

All maths is LaTeX rendered by the local KaTeX: inline `\( ... \)`, display `$$ ... $$` (inside `<div class="eq">`).
In SVG diagrams use a `<foreignObject>` containing an XHTML `<div>` with `\( ... \)`. Matplotlib figures use mathtext
(`$...$`, `mathtext.fontset = "cm"`). Check a page with `python tools/scan_math.py <page>`.
