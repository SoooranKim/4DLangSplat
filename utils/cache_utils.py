import os
from pathlib import Path


_CACHE_ROOT = Path("/media/ssd1/users/sooran/.cache")


def get_hf_cache_dir() -> str:
    """
    Return a cache directory under /media/ssd1/users/sooran/.cache.

    Preference order (first existing path wins):
      1) /media/ssd1/users/sooran/.cache/huggingface/hub
      2) /media/ssd1/users/sooran/.cache/huggingface
      3) /media/ssd1/users/sooran/.cache

    If none exist, the base cache directory is created and returned.
    """
    candidates = [
        _CACHE_ROOT / "huggingface" / "hub",
        _CACHE_ROOT / "huggingface",
        _CACHE_ROOT,
    ]

    for p in candidates:
        if p.exists():
            cache_dir = p
            break
    else:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        cache_dir = _CACHE_ROOT

    cache_dir = cache_dir.resolve()

    # Also hint HuggingFace / HF Hub based libs to reuse this cache
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))

    return str(cache_dir)


__all__ = ["get_hf_cache_dir"]


