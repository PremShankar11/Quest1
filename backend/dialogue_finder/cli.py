from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT, Config
from .progress import PrintReporter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dialogue_finder",
                                description="Find the first frame where a dialogue appears in a video.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="video URL (any yt-dlp supported site)")
    src.add_argument("--local", help="path to a local video file")
    p.add_argument("--text", required=True, help='target dialogue, e.g. "My mind rebels at stagnation"')
    p.add_argument("--mode", choices=["hybrid", "audio", "ocr"], default="hybrid")
    p.add_argument("--occurrence", choices=["first", "last", "all"], default="first")
    p.add_argument("--out", default=None, help="output directory (default: <repo>/output)")
    p.add_argument("--json", action="store_true", help="also print result.json content to stdout")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:               # argparse exits 2 on usage error
        return int(e.code) if e.code is not None else 2
    out_dir = Path(args.out) if args.out is not None else DEFAULT.output_dir
    cfg = Config(output_dir=out_dir, cache_dir=DEFAULT.cache_dir)
    try:
        from .pipeline import PipelineError, run
    except Exception as e:                # broken/missing dependency (e.g. opencv DLL load failure)
        print(f"Error: cannot start (missing or broken dependency): {type(e).__name__}: {str(e)[:200]}",
              file=sys.stderr)
        return 1
    try:
        res = run(args.url or args.local, args.text, cfg=cfg, reporter=PrintReporter(args.verbose),
                  mode=args.mode, occurrence=args.occurrence, local=args.local is not None)
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Error: interrupted", file=sys.stderr)
        return 130
    except Exception as e:                # last line of defence: never a traceback
        print(f"Error: unexpected failure ({type(e).__name__}: {str(e)[:200]})", file=sys.stderr)
        return 1
    print(res.format_block())
    for alt in res.alternatives:
        print(f"Also at   : {alt.frame_index} ({alt.timestamp_s:.3f}s) score {alt.score:.2f}")
    try:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "result.json").write_text(json.dumps(res.to_dict(), indent=2, default=str),
                                                     encoding="utf-8")
    except OSError as e:
        print(f"Error: could not write output: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0
