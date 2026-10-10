"""Reverb examples: a dry CMU ARCTIC (ksp) sentence convolved with synthetic room impulse responses.

Run from this folder:  python make_reverb.py   ->  reverb_small.wav, reverb_medium.wav, reverb_large.wav
Same method as the course notebook room_reverb_convolution.ipynb: each RIR is white noise under an exponential
envelope that falls by 60 dB at t = RT60, normalised to unit energy.
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "speech", "common"))
import arctic  # noqa: E402

UTT = "arctic_a0005"                     # "Will we ever forget it."
SR = arctic.SR
rng = np.random.default_rng(42)


def synthetic_rir(rt60, tail_factor=1.4):
    tau = rt60 / (3 * np.log(10))
    t = np.arange(int(SR * rt60 * tail_factor)) / SR
    h = rng.standard_normal(len(t)) * np.exp(-t / tau)
    return h / np.sqrt(np.sum(h ** 2))


dry = arctic.read_wav(UTT)
for name, rt60 in [("small", 0.2), ("medium", 0.6), ("large", 1.5)]:
    wet = fftconvolve(dry, synthetic_rir(rt60))
    wet = wet / np.max(np.abs(wet)) * 0.95
    sf.write(os.path.join(HERE, f"reverb_{name}.wav"), wet.astype(np.float32), SR, subtype="PCM_16")
    print(name, rt60, f"{len(wet) / SR:.2f} s")
print("dry:", arctic.text(UTT))
