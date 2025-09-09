import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from smalltts.infer.onnx import SmallTTS

if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    print("loading model")
    model = SmallTTS()
    print("generating")
    y = model(
        [torch.from_numpy(np.load("assets/tryme/latents.npy")).float()],
        [np.load("assets/tryme/tokens.npy").tolist()],
        [
            sys.argv[1]
            if len(sys.argv) > 1
            else "hello this is small brain speaking, thanks for trying this model out and have fun"
        ],
    )[0]
    sf.write("out/tryme.wav", y.squeeze(0).t().numpy(), 24_000, subtype="PCM_16")
    print("out/tryme.wav")
