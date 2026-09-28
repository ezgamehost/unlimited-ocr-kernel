# Frozen synthetic validation pages

These four pages were created after the initial kernels and thread settings were
selected. Their content is independent of the original four-page corpus and is
frozen before their first inference. They remain synthetic documents, not a
representative scanned-document dataset.

| Page | Pixels | Expected local grid | Image tokens |
| --- | --- | --- | ---: |
| Receipt | 480 × 640 | Single global view | 273 |
| Wide field report | 1280 × 640 | 2 × 1 | 483 |
| Bilingual notice | 960 × 960 | 2 × 2 | 693 |
| Dense archive register | 1024 × 1536 | 2 × 3 | 903 |

Expected counts follow the pinned gundam crop policy and separator layout. Both
runtimes must confirm them; expectations are not adjusted merely to make a run
pass. The manifest records exact transcripts, PNG/font hashes, text bounds and
Pillow version. Generation checks font coverage and text clipping. Reproduce
from the repository root with `PYTHONPATH=. .venv/bin/python benchmarks/heldout/generate.py`.

Do not edit text or images in response to OCR output. Retain quality failures,
layout differences and truncation alongside successful results. A correctness
pilot precedes repeated timing. If these pages subsequently guide tuning, say
so instead of claiming they are still unseen validation data.
