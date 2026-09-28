"""Deterministic full 32-crop portrait input and exact transcription."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, __version__

root = Path(__file__).resolve().parent
font_path = Path('/System/Library/Fonts/Menlo.ttc')
font = ImageFont.truetype(str(font_path), 22)
image = Image.new('RGB', (640, 20480), 'white')
draw = ImageDraw.Draw(image)
lines = []
for section in range(32):
    block = [f'SECTION {section + 1:02d}', 'Memory validation document.', f'Row identity: {section + 1:02d}.']
    for i, line in enumerate(block):
        y = section * 640 + 64 + i * 34
        box = draw.textbbox((40, y), line, font=font)
        assert box[2] < 600 and box[3] < (section + 1) * 640
        draw.text((40, y), line, fill='black', font=font)
    lines.extend(block)
image.save(root / 'portrait32.png')
text = '\n'.join(lines) + '\n'
(root / 'portrait32.txt').write_text(text)
manifest = dict(width=640, height=20480, expected_crop_grid=[1, 32], expected_image_tokens=3793,
    sha256=hashlib.sha256((root / 'portrait32.png').read_bytes()).hexdigest(),
    font=str(font_path), font_sha256=hashlib.sha256(font_path.read_bytes()).hexdigest(),
    font_size=22, pillow_version=__version__, description='Synthetic extreme-aspect memory/ordering check, not a representative document corpus.')
(root / 'input.json').write_text(json.dumps(manifest, indent=2) + '\n')
