from functools import lru_cache

import torch
from torchaudio.transforms import Resample


@lru_cache(maxsize=None)
def make_resampler(sr_from: int, sr_to: int) -> Resample:
    return Resample(
        orig_freq=sr_from,
        new_freq=sr_to,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=1024,
        rolloff=0.94,
        beta=14.769656459379492,
    )


def resample_hq(x: torch.Tensor, sr: int, target: int) -> torch.Tensor:
    if sr == target:
        return x
    r = make_resampler(sr, target)
    return r(x)


