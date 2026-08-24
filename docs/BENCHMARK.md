# Benchmark (synthetic ground truth, OCR mode)

| variant | truth frame | found frame | error (frames) | source | confidence | seconds |
|---|---|---|---|---|---|---|
| baseline_640x360_24fps_bottom | 120 | 120 | 0 | ocr | HIGH | 24.5 |
| top_position | 120 | 120 | 0 | ocr | HIGH | 27.3 |
| center_position | 120 | 120 | 0 | ocr | HIGH | 30.1 |
| fade_12_frames | 120 | 121 | 1 | ocr | MEDIUM | 23.3 |
| small_text_360p | 120 | 120 | 0 | ocr | HIGH | 23.0 |
| hd_1280x720 | 120 | 120 | 0 | ocr | HIGH | 23.6 |
| 30fps | 150 | 150 | 0 | ocr | HIGH | 23.0 |
| 60fps | 300 | 300 | 0 | ocr | HIGH | 24.1 |
