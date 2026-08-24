"""Run the pipeline (OCR mode) on synthetic variants; write docs/BENCHMARK.md with frame error per variant."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.make_clip import make_clip  # noqa: E402
from dialogue_finder.config import Config  # noqa: E402
from dialogue_finder.text.ocr import RapidOCRExtractor  # noqa: E402
from dialogue_finder.pipeline import run  # noqa: E402

TEXT = "My mind rebels at stagnation"
VARIANTS = {
    "baseline_640x360_24fps_bottom": dict(),
    "top_position": dict(position="top"),
    "center_position": dict(position="center"),
    "fade_12_frames": dict(fade_frames=12),
    "small_text_360p": dict(scale=0.6),
    "hd_1280x720": dict(size=(1280, 720)),
    "30fps": dict(fps=30),
    "60fps": dict(fps=60),
}


def main() -> None:
    out_dir = Path("bench_out"); out_dir.mkdir(exist_ok=True)
    ex = RapidOCRExtractor()
    rows = []
    for name, kw in VARIANTS.items():
        clip = out_dir / f"{name}.mp4"
        truth = make_clip(clip, text=TEXT, appear_s=5.0, **kw)
        cfg = Config(output_dir=out_dir / name, cache_dir=out_dir / "cache")
        t = time.perf_counter()
        try:
            res = run(str(clip), TEXT, cfg=cfg, mode="ocr", local=True, extractor=ex)
            err = res.frame_index - truth["frame"]
            rows.append((name, truth["frame"], res.frame_index, err, res.source, res.confidence, f"{time.perf_counter() - t:.1f}"))
        except Exception as e:
            rows.append((name, truth["frame"], "-", "-", f"error: {e}", "-", "-"))
    md = ["# Benchmark (synthetic ground truth, OCR mode)", "",
          "| variant | truth frame | found frame | error (frames) | source | confidence | seconds |",
          "|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    Path("../docs/BENCHMARK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
