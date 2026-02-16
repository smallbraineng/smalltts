# smalltts

<p align="left">
  <a href="https://huggingface.co/smallbraineng/smalltts"><img alt="🤗 huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97_huggingface-smallbraineng/smalltts-yellow?logo=huggingface" /></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="python" src="https://img.shields.io/badge/python-3.10-blue" /></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="license" src="https://img.shields.io/badge/license-MIT-green" /></a>
  <a href="https://huggingface.co/smallbraineng/smalltts"><img alt="weights license" src="https://img.shields.io/badge/weights-CC--BY--NA-orange" /></a>
</p>

expressive tts that can clone any voice

- open training code and onnx model
- much faster than realtime on cpu & gpu
- emotive, built for characters

## get started

### install + quick test (2 lines)

```bash
git clone https://github.com/smallbraineng/smalltts && cd smalltts
uv run python src/scripts/tryme.py "Hello from smallTTS!"
```

uses [uv](https://github.com/astral-sh/uv) for setup.

### interactive (realtime on cpu)

```bash
uv run python src/scripts/infer/interactive.py
```

### install as a package

```bash
uv pip install "git+https://github.com/smallbraineng/smalltts"
```

```python
from smalltts import SmallTTS
tts = SmallTTS()
```

### more examples

#### batch inference

```bash
uv run python src/scripts/infer/batch.py
```

#### voice cloning

```bash
uv run python src/scripts/infer/clone.py \
  --wav assets/test_audio/1.wav \
  --transcription "the reference transcription here" \
  --text "what you want it to say"
```

## benchmarks

nvidia t4, rust onnx server, cuda execution provider:

| duration | batch=1 | batch=2 | batch=4 | batch=8 |
| ---: | ---: | ---: | ---: | ---: |
| 2s | 166.5ms (0.083x) | 332.0ms (0.083x) | 666.0ms (0.083x) | 1333.1ms (0.083x) |
| 5s | 217.7ms (0.043x) | 434.8ms (0.043x) | 873.9ms (0.043x) | 1752.6ms (0.043x) |
| 10s | 315.2ms (0.032x) | 627.4ms (0.031x) | 1267.0ms (0.032x) | 2537.5ms (0.032x) |

RTF scales linearly with batch size. run `cargo run --release --bin bench` in `src/server/` to benchmark on your hardware.

## fine-tune or train your own

### papers (context)

[dmd2](https://arxiv.org/abs/2405.14867) · [dmdspeech](https://arxiv.org/abs/2410.11097) · [f5‑tts](https://arxiv.org/abs/2410.06885) · [echo tts](https://jordandare.github.io/blog/2025/echo/)

checkpoints live on huggingface: [smallbraineng/smalltts](https://huggingface.co/smallbraineng/smalltts).

### checkpoints

| name | download |
| --- | --- |
| teacher_checkpoints | [assets/teacher_checkpoints](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/teacher_checkpoints) |
| dmd_checkpoints | [assets/dmd_checkpoints](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/dmd_checkpoints) |
| asr_checkpoints | [assets/asr_checkpoints](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/asr_checkpoints) |
| sv_checkpoints | [assets/sv_checkpoints](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/sv_checkpoints) |

from-scratch training order, in `src/scripts/train`:
1. teacher model (128 steps): `uv run accelerate launch src/scripts/train/teacher.py`
2. dmd2 distillation (4 steps): `uv run accelerate launch src/scripts/train/dmd2/distill.py`
3. speaker verification: `uv run accelerate launch src/scripts/train/dmd2/sv.py`
4. asr: `uv run accelerate launch src/scripts/train/dmd2/asr.py`

### method
- teacher is a diffusion tts model generating audio in 128 sampling steps
- we train on encoded latents, not audio/mels. [microsoft vibevoice](https://github.com/microsoft/VibeVoice) encoder/decoder gives ~3200x compression
- we distill to 4 steps with distribution matching distillation ([dmd2 paper](https://arxiv.org/abs/2405.14867))
- inspired by [dmdspeech](https://arxiv.org/abs/2410.11097), [dmospeech2](https://arxiv.org/abs/2507.14988), and [echo tts](https://jordandare.github.io/blog/2025/echo/)
- during distillation we train an asr and a speaker verification model to keep phonemes and style aligned
- no cfg at inference -- guidance is distilled out

### architecture
- **dit backbone**: 12 blocks, hidden_dim=960, joint attention (self + cross to ref and text)
- **text encoder**: 8-layer transformer, dim=512, 4 heads, RoPE, RMSNorm
- **style encoder**: 12-layer transformer, dim=512, patch_size=1, learnable scale
- **codec**: 64-dim vibevoice latents, hop=3200 (~7.5 frames/sec at 24kHz)
- **distillation**: 128 teacher steps -> 4 student steps via dmd2

### notes

- joint attention over self + reference audio + text in every layer, single fused sdpa call
- 128-step teacher distilled to 4 steps via dmd2, no cfg at inference
- diffusion on 64-dim vibevoice codec latents (~7.5 frames/sec), not mels
- per-block cross-kv projections, precomputed once across all denoiser steps
- 23 built-in spontaneous event tokens: [laughter], [cough], [groan], ...

### data
- the default dataloader is dummy. bring your own
- we have tested [webdataset](https://github.com/webdataset/webdataset)
- checkpoints on huggingface make fine‑tuning easy

## inference

use onnx exports for production inference. models download automatically from huggingface on first run.

| model | path |
| --- | --- |
| condition encoder | [assets/dmd/condition_encoder.onnx](https://huggingface.co/smallbraineng/smalltts/tree/main/dmd) |
| denoiser (4 steps) | [assets/dmd/denoiser.onnx](https://huggingface.co/smallbraineng/smalltts/tree/main/dmd) |
| codec encoder | [assets/codec/encoder.onnx](https://huggingface.co/smallbraineng/smalltts/tree/main/codec) |
| codec decoder | [assets/codec/decoder.onnx](https://huggingface.co/smallbraineng/smalltts/tree/main/codec) |

see `src/smalltts/infer/onnx.py` for the python api, or use the scripts above.

## server

a rust inference server lives in `src/server/`. uses axum + onnx runtime, with [x402](https://x402.org) payment gating.

```bash
cd src/server
PAYMENT_ADDRESS=0x... cargo run --release
```

endpoint: `POST /synthesize?duration=N` with multipart form (audio + text). returns audio/wav. gated by x402 -- clients pay $0.05/min of generated audio in USDC on Base.

a web frontend lives in `src/website/` (vite + react + privy for wallet connection).

```bash
cd src/website
bun install && bun run dev
```

## roadmap

- [x] **09.09.25** 🚀 v0 released, model and ONNX exports on 🤗 Hugging Face
- [ ] 📱 quantized version & mobile example
- [ ] 🏋️‍♂️ larger training run

## licenses

code is licensed under CC-BY-NC. see [LICENSE](LICENSE).

model weights on huggingface are licensed CC‑BY‑NA.

## next steps

1. quantized builds and realtime mobile demo
2. full version with stability upgrades

## citations

thanks to the authors and communities of dmdspeech, nanospeech, f5‑tts, echo tts, and vibevoice — this repo is heavily inspired by their ideas and codebases.

```bibtex
@article{peng2025vibevoice,
    title         = {VibeVoice Technical Report},
    author        = {Zhiliang Peng and Jianwei Yu and Wenhui Wang and Yaoyao Chang and Yutao Sun and Li Dong and Yi Zhu and Weijiang Xu and Hangbo Bao and Zehua Wang and Shaohan Huang and Yan Xia and Furu Wei},
    year          = {2025},
    eprint        = {2508.19205},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
    doi           = {10.48550/arXiv.2508.19205},
    url           = {https://arxiv.org/abs/2508.19205}
}

@article{li2024dmdspeech,
    title         = {DMDSpeech: Distilled Diffusion Model Surpassing the Teacher in Zero-shot Speech Synthesis via Direct Metric Optimization},
    author        = {Yinghao Aaron Li and Rithesh Kumar and Zeyu Jin},
    year          = {2024},
    eprint        = {2410.11097},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2410.11097},
    url           = {https://arxiv.org/abs/2410.11097}
}

@article{yin2024dmd2,
    title         = {Improved Distribution Matching Distillation for Fast Image Synthesis},
    author        = {Tianwei Yin and Micha{\"e}l Gharbi and Taesung Park and Richard Zhang and Eli Shechtman and Fredo Durand and William T. Freeman},
    year          = {2024},
    eprint        = {2405.14867},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    doi           = {10.48550/arXiv.2405.14867},
    url           = {https://arxiv.org/abs/2405.14867}
}

@article{li2025dmospeech2,
    title         = {DMOSpeech 2: Reinforcement Learning for Duration Prediction in Metric-Optimized Speech Synthesis},
    author        = {Yinghao Aaron Li and Xilin Jiang and Fei Tao and Cheng Niu and Kaifeng Xu and Juntong Song and Nima Mesgarani},
    year          = {2025},
    eprint        = {2507.14988},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2507.14988},
    url           = {https://arxiv.org/abs/2507.14988}
}

@article{chen2024f5tts,
    title         = {F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
    author        = {Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
    year          = {2024},
    eprint        = {2410.06885},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2410.06885},
    url           = {https://arxiv.org/abs/2410.06885}
}

@misc{newman2025nanospeech,
    title        = {Nanospeech: A simple, hackable text-to-speech system in PyTorch and MLX},
    author       = {Lucas Newman},
    year         = {2025},
    note         = {GitHub repository},
    howpublished = {\url{https://github.com/lucasnewman/nanospeech}}
}

@article{eskimez2024e2tts,
    title         = {E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS},
    author        = {Sefik Emre Eskimez and Xiaofei Wang and Manthan Thakker and Canrun Li and Chung-Hsien Tsai and Zhen Xiao and Hemin Yang and Zirun Zhu and Min Tang and Xu Tan and Yanqing Liu and Sheng Zhao and Naoyuki Kanda},
    year          = {2024},
    eprint        = {2406.18009},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2406.18009},
    url           = {https://arxiv.org/abs/2406.18009}
}

@misc{darefsky2025echo,
    title        = {Echo: Diffusion-based text-to-speech with fast, high-fidelity voice cloning},
    author       = {Jordan Darefsky},
    year         = {2025},
    howpublished = {\url{https://jordandare.github.io/blog/2025/echo/}}
}

@article{zhu2025zipvoice,
    title         = {ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
    author        = {Han Zhu and Wei Kang and Zengwei Yao and Liyong Guo and Fangjun Kuang and Zhaoqing Li and Weiji Zhuang and Long Lin and Daniel Povey},
    year          = {2025},
    eprint        = {2506.13053},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2506.13053},
    url           = {https://arxiv.org/abs/2506.13053}
}
```
