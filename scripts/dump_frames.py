"""Spike: download a video and save frames at the given timestamps (seconds) to bench_out/spike/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import cv2  # noqa: E402
from dialogue_finder.config import DEFAULT  # noqa: E402
from dialogue_finder.downloader import fetch_video, probe  # noqa: E402
from dialogue_finder.progress import PrintReporter  # noqa: E402

url = sys.argv[1]
times = [float(t) for t in sys.argv[2:]]
path = fetch_video(url, DEFAULT, PrintReporter(verbose=True))
info = probe(path)
print(info)
out = Path("bench_out/spike"); out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(path))
for t in times:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if ok:
        p = out / f"t{int(t):05d}.png"; cv2.imwrite(str(p), frame); print("saved", p)
