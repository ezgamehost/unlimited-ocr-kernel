"""Reuse the frozen rendering-bound scorer for the matched BF16 runs."""
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parent / 'bf16-quality'
for name in ('coordinate_diff', 'layout_score'):
    path = root.parent.parent / '2026-09-28-teacher-logits' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    module.BASELINE = module.BASE = root / 'baseline'
    module.CANDIDATE = module.NEW = root / 'candidate'
    module.main()
