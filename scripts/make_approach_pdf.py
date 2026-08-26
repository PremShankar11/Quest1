"""Render docs/APPROACH.md into a styled, print-ready docs/APPROACH.pdf.

Markdown -> HTML (python-markdown) -> PDF (headless Chrome). Everything is local: the HTML is
self-contained apart from the images it references out of docs/media/, which resolve through the
<base> tag. Run from the repo root:

    .venv/Scripts/python scripts/make_approach_pdf.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MD = DOCS / "APPROACH.md"
HTML = DOCS / "APPROACH.html"
PDF = DOCS / "APPROACH.pdf"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

# The Phase 8 notes were written with inline LaTeX; plain unicode reads better in both GitHub and print.
LATEX_FIXES = [
    (r"\$t_\{\\text\{start\}\} - 3\\text\{s\} \\to t_\{\\text\{end\}\} \+ 3\\text\{s\}\$", "t_start − 3 s → t_end + 3 s"),
    (r"\$t_0 \\to t_\{\\text\{end\}\}\$", "t₀ → t_end"),
    (r"\$t_\{\\text\{end\}\} \\to t_0\$", "t_end → t₀"),
    (r"\$\\ge ([0-9.]+)\$", r"≥ \1"),
    (r"\$\\pm ([0-9]+)\\text\{s\}\$", r"±\1 s"),
    (r"\$([0-9]+)\\text\{s\}\$", r"\1 s"),
    (r"\$z\\text\{-index: \} ([0-9]+)\$", r"z-index \1"),
]


def clean_markdown(text: str) -> str:
    for pattern, replacement in LATEX_FIXES:
        text = re.sub(pattern, replacement, text)
    return text


CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #1B1E23; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.cover { height: 252mm; display: flex; flex-direction: column; justify-content: center; text-align: center;
         page-break-after: always; }
.cover .eyebrow { font-family: Consolas, monospace; font-size: 10pt; letter-spacing: .22em;
                  text-transform: uppercase; color: #0E7C7B; margin-bottom: 14mm; }
.cover h1 { font-size: 30pt; line-height: 1.15; margin: 0 0 6mm; letter-spacing: -.01em; border: 0; }
.cover .sub { font-size: 13pt; color: #5D636D; margin-bottom: 16mm; }
.cover .rule { width: 42mm; height: 3px; background: #B7791F; margin: 0 auto 16mm; }
.cover .meta { font-family: Consolas, monospace; font-size: 9.5pt; color: #5D636D; line-height: 2; }
h1, h2, h3, h4 { font-weight: 650; color: #12151A; page-break-after: avoid; }
h1 { font-size: 20pt; margin: 0 0 6mm; }
h2 { font-size: 15pt; margin: 0 0 5mm; padding-top: 4mm; border-top: 2px solid #0E7C7B;
     page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 12pt; margin: 7mm 0 2.5mm; color: #0E7C7B; }
h4 { font-size: 10.5pt; margin: 5mm 0 2mm; }
p { margin: 0 0 3.2mm; }
ul, ol { margin: 0 0 3.5mm; padding-left: 6mm; }
li { margin-bottom: 1.4mm; }
strong { color: #12151A; }
a { color: #0E7C7B; text-decoration: none; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #F1EFE9; padding: 0.3mm 1.1mm;
       border-radius: 2px; color: #2A2F38; }
pre { background: #F7F6F2; border: 1px solid #E2DED4; border-left: 3px solid #0E7C7B; border-radius: 3px;
      padding: 3mm 4mm; overflow: hidden; page-break-inside: avoid; margin: 0 0 4mm; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.45; white-space: pre-wrap;
           word-break: break-word; }
blockquote { margin: 0 0 4mm; padding: 2mm 4mm; border-left: 3px solid #B7791F; background: #FDF8EE;
             color: #4A5058; }
table { width: 100%; border-collapse: collapse; margin: 0 0 5mm; font-size: 8.8pt;
        page-break-inside: avoid; }
th { background: #E9F6F5; color: #12151A; text-align: left; font-weight: 650; }
th, td { border: 1px solid #DCD8CE; padding: 1.8mm 2.4mm; vertical-align: top; }
tbody tr:nth-child(even) { background: #FAF9F6; }
td code { font-size: 8pt; }
img, svg { max-width: 100%; height: auto; page-break-inside: avoid; }
p:has(img) { text-align: center; margin: 4mm 0 3mm; }
sub { color: #6C727B; font-size: 8.6pt; }
hr { border: 0; border-top: 1px solid #E2DED4; margin: 6mm 0; }
div[align="center"] h1 { text-align: center; }
"""

COVER = """
<div class="cover">
  <div class="eyebrow">Technical Approach</div>
  <h1>Dialogue&nbsp;Frame&nbsp;Finder</h1>
  <div class="sub">How a video URL and one line of dialogue<br>become one exact frame</div>
  <div class="rule"></div>
  <div class="meta">
    4 search modes &nbsp;·&nbsp; 151 automated tests<br>
    audio + on-screen text + active-speaker verification<br>
    GPU-accelerated, CPU-safe, runs entirely offline
  </div>
</div>
"""


def build_html(md_text: str) -> str:
    # The markdown file opens with a centred title block; the PDF has a cover page instead.
    md_text = re.sub(r'^<div align="center">.*?</div>\s*', "", md_text, count=1, flags=re.S)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists", "md_in_html"],
        output_format="html5",
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<base href='{DOCS.as_uri()}/'>"
        "<title>Dialogue Frame Finder — Approach</title>"
        f"<style>{CSS}</style></head><body>{COVER}{body}</body></html>"
    )


def find_chrome() -> Path:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit("No Chrome or Edge found — install one, or print docs/APPROACH.html by hand.")


def main() -> int:
    md_text = clean_markdown(MD.read_text(encoding="utf-8"))
    MD.write_text(md_text, encoding="utf-8")          # keep the source and the PDF in sync
    HTML.write_text(build_html(md_text), encoding="utf-8")
    chrome = find_chrome()
    result = subprocess.run(
        [str(chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
         f"--print-to-pdf={PDF}", HTML.as_uri()],
        capture_output=True, text=True, timeout=180,
    )
    if not PDF.exists():
        print(result.stderr[-2000:], file=sys.stderr)
        return 1
    print(f"wrote {PDF.relative_to(ROOT)} ({PDF.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
