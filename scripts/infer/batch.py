import json
from pathlib import Path

import soundfile as sf
import torch

from smalltts.assets.ensure import ensure_assets
from smalltts.codec.onnx import Encoder
from smalltts.infer.onnx import SmallTTS
from smalltts.infer.utils import resample_hq


def load_audio(paths):
    xs = []
    for i, p in enumerate(paths):
        print(f"loading {i + 1}/{len(paths)} {p}")
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        x = torch.from_numpy(y).view(1, -1)
        x = resample_hq(x, sr, 24_000)
        xs.append(x.to(torch.float32))
    return xs


if __name__ == "__main__":
    ensure_assets(["codec", "e2e", "test_audio"])
    td = Path("assets/test_audio")
    with open(td / "transcriptions.json") as f:
        items = json.load(f)
    files = [td / it["filename"] for it in items]
    trans = [it["transcription"] for it in items]
    texts = [
        "Hello world, I am small tts, and I am talking!",
        "I can clone any voice and emotion.",
        "I have an ONNX export and run very fast.",
        "Woah, this is awesome I can do any character!",
    ]
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"loading {len(files)} files")
    xs = load_audio([str(p) for p in files])
    enc = Encoder()
    tts = SmallTTS()
    conds = []
    for i, x in enumerate(xs):
        print(f"encoding {i + 1}/{len(xs)} for conditioning")
        lat = enc.encode(x.unsqueeze(0))
        conds.append(lat[0].cpu())
    bs = 4
    for i in range(0, len(conds), bs):
        j = min(len(conds), i + bs)
        print(f"inference bsz={j - i}")
        ys = tts(conds[i:j], trans[i:j], texts[i:j])
        for k, y in enumerate(ys):
            sf.write(
                str(outdir / f"{Path(files[i + k]).stem}_gen.wav"),
                y.cpu().contiguous().squeeze(0).t().numpy(),
                24_000,
                subtype="PCM_16",
            )
            print(str(outdir / f"{Path(files[i + k]).stem}_gen.wav"))
