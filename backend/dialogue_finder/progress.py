from __future__ import annotations

import sys
from typing import Protocol

from .models import StageEvent


class ProgressReporter(Protocol):
    def emit(self, event: StageEvent) -> None: ...


class NullReporter:
    def emit(self, event: StageEvent) -> None:
        return None


class PrintReporter:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._last_progress_stage = ""

    def emit(self, event: StageEvent) -> None:
        if event.progress is not None and not self.verbose:
            return
        tag = f"[{event.stage}:{event.status}]"
        extra = f" {event.progress:.0%}" if event.progress is not None else ""
        print(f"{tag}{extra} {event.message}", file=sys.stderr)
