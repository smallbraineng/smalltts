import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "smallbraineng/smalltts"


def _repo_type():
    api = HfApi()
    for t in ("model", "dataset"):
        try:
            api.repo_info(REPO, repo_type=t)
            return t
        except Exception:
            pass
    return "model"


def ensure_assets(paths):
    if isinstance(paths, (list, tuple, set)):
        for p in paths:
            ensure_assets(p)
        return
    folder = str(paths).strip("/ ")
    if not folder:
        return
    local_root = Path("assets") / folder
    if local_root.exists():
        return
    rt = _repo_type()
    snapshot_download(
        repo_id=REPO,
        repo_type=rt,
        allow_patterns=[f"{folder}/*"],
        local_dir=str(Path("assets")),
        max_workers=os.cpu_count() or 8,
        tqdm_class=None,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m smalltts.assets.ensure <folder> [<folder>...]")
        sys.exit(1)
    ensure_assets(sys.argv[1:])
