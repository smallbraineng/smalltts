import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from smalltts.assets.ensure import ensure_assets
from smalltts.data.phonemization.phonemes import get_token_ids
from smalltts.infer.onnx import SmallTTS, estimate_duration

if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    ensure_assets(["tryme", "codec", "dmd"])

    text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "hello this is small brain speaking, thanks for trying this model out and have fun"
    )

    print("loading model")
    model = SmallTTS()
    ref_latents = np.load("assets/tryme/latents.npy").astype(np.float32)
    tokens = get_token_ids(text)
    duration = estimate_duration(text)

    print(f"generating ({duration:.1f}s estimated)")
    audio = model.synthesize(ref_latents, tokens, duration)
    sf.write("out/tryme.wav", audio.squeeze(), 24_000, subtype="PCM_16")
    print("out/tryme.wav")
