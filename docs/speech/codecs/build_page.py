"""Assemble ../codecs.html from page_src.html: shared head/styles from linear_prediction.html, and the
listening grids filled in from numbers.json (so the quoted SNRs always match the audio files)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
src = open(os.path.join(HERE, "page_src.html"), encoding="utf8").read()
lp = open(os.path.join(ROOT, "linear_prediction.html"), encoding="utf8").read()
head = lp[lp.index('<link rel="stylesheet" href="../assets/katex'):lp.index('<style>')]
style = lp[lp.index('<style>') + len('<style>'):lp.index('</style>')]
num = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf8"))
rows = {r["key"]: r for r in num["rows"]}


def card(key, title, note, tag=None):
    r = rows[key]
    rate = f"{r['kbps']:g} kbit/s" if r["kbps"] else "unquantised parameters"
    snr = f" · segSNR {r['segsnr']:.1f} dB" if r["segsnr"] is not None else ""
    t = f' <span class="tag" style="color:{tag[1]}">{tag[0]}</span>' if tag else ""
    return (f'<div class="card"><h4>{title}{t}</h4><div class="rate">{rate}{snr}</div><p>{note}</p>'
            f'<audio controls preload="none" src="codecs/audio/{key}.wav"></audio></div>')


grids = {
    "pcmGrid": [
        card("orig_16k_16b", "Original", r"16 kHz \(\times\) 16 bits, the reference."),
        card("pcm_16k_8b", "16 kHz, 8 bits", "Faint hiss in quiet parts."),
        card("pcm_16k_4b", "16 kHz, 4 bits", "Loud crackling noise riding on the speech."),
        card("pcm_16k_2b", "16 kHz, 2 bits", "Harsh distortion: four levels are not enough."),
        card("pcm_8k_16b", "8 kHz, 16 bits", "Telephone bandwidth: clean but duller; /s/ weakened."),
        card("pcm_4k_16b", "4 kHz, 16 bits", "Muffled: everything above 2 kHz is gone."),
        card("pcm_2k_16b", "2 kHz, 16 bits", "Only F1 survives: words very hard to follow."),
        card("pcm_8k_8b", "8 kHz, 8 bits, uniform", "64 kbit/s the naive way: audible noise floor."),
    ],
    "adpcmGrid": [
        card("pcm_8k_4b", "4-bit PCM", "No tricks: noise everywhere.", ("baseline", "var(--accent-4)")),
        card("mulaw_8k_4b", r"4-bit \(\mu\)-law", "Levels spread to match loudness.", ("+ companding", "var(--accent)")),
        card("apcm_8k_4b", "4-bit adaptive step", "Step size follows the loudness, no prediction yet.", ("+ adaptation", "var(--accent-3)")),
        card("adpcm_8k_4b", "4-bit ADPCM", "Adaptive step + 10-weight predictor: code only the error.", ("+ prediction", "var(--accent-2)")),
        card("adpcm_8k_3b", "3-bit ADPCM", "24 kbit/s: noisier but clearly usable."),
        card("adpcm_8k_2b", "2-bit ADPCM", "16 kbit/s: a 2-bit coder that still carries the speech."),
    ],
    "vocGrid": [
        card("orig_16k_16b", "Original", "256 kbit/s."),
        card("vocoder_unquantised", "LPC vocoder, exact parameters", "Model only: pulses or noise through the measured filter."),
        card("vocoder_q", "LPC vocoder, 57 bits / 20 ms", "Parameters rounded to the bit budget above."),
    ],
}
out = src.replace("<!--HEAD-->", head.strip()).replace("<!--STYLE-->", style.strip("\n"))
for gid, cards in grids.items():
    tag = f'<div class="agrid" id="{gid}"'
    i = out.index(tag); j = out.index("</div>", i)
    out = out[:j] + "\n    " + "\n    ".join(cards) + "\n  " + out[j:]
import re  # noqa: E402
out = re.sub(r"<!--N:(\w+)-->", lambda m: str(num[m.group(1)]), out)     # numbers quoted in the text come from numbers.json
_n = [0]
def _renumber(m):  # figure captions are numbered in order of appearance
    _n[0] += 1
    return f"<b>Figure {_n[0]}.</b>"
out = re.sub(r"<b>Figure \d+\.</b>", _renumber, out)
open(os.path.join(ROOT, "codecs.html"), "w", encoding="utf8").write(out)
print("wrote codecs.html")
