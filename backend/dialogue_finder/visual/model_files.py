"""Shared download-with-retry mechanics for pinned model files fetched over HTTP (Plan 4).

The YuNet model (`visual/faces.py`) and the LR-ASD weights (`visual/lrasd.py`) are both fetched
the same way: try up to `_DOWNLOAD_ATTEMPTS` times with exponential backoff, write to a `.part`
temp file, hand it to the caller's own verifier, then atomically replace the final path on
success. Verification itself stays with each caller -- they disagree on exception type
(`RuntimeError` vs `WeightsVerificationError`) and message wording -- so only that retry/
backoff/temp-file shell is shared here.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_BACKOFF_S = 1.0


def fetch_verified(
    url: str,
    dest: Path,
    verify: Callable[[Path], None],
    name: str,
    *,
    reraise: tuple[type[BaseException], ...] = (),
) -> None:
    """Download `url` to `dest`, calling `verify(tmp)` on the temp file before the atomic
    rename. Retries network/OS errors `_DOWNLOAD_ATTEMPTS` times with backoff. Exceptions in
    `reraise` (a caller's own verification-failure type, when it subclasses OSError) propagate
    immediately instead of being retried -- a hash mismatch is never treated as a transient
    network error."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            verify(tmp)
            tmp.replace(dest)
            return
        except reraise:
            raise
        except (urllib.error.URLError, OSError) as e:
            last_error = e
            if attempt < _DOWNLOAD_ATTEMPTS - 1:
                time.sleep(_DOWNLOAD_BACKOFF_S * (2 ** attempt))
    raise IOError(
        f"Could not download {name} from {url} to {dest} after "
        f"{_DOWNLOAD_ATTEMPTS} attempts: {last_error}\n"
        f"Manual fallback: curl -L -o {dest} {url}"
    ) from last_error
