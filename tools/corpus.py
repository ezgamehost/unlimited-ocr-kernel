"""Render a small deterministic document corpus with exact transcription."""
import argparse
import hashlib
import json
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


def require_glyphs(font: Path, text: str) -> None:
    with TTFont(font, fontNumber=0) as face:
        supported = face.getBestCmap()
        missing = {c for c in text if not c.isspace() and ord(c) not in supported}
    if missing:
        raise ValueError(f"font lacks characters: {sorted(missing)!r}")


def generate_corpus(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    latin = Path('/System/Library/Fonts/Menlo.ttc')
    chinese = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    english = [
        'MEETING NOTES', '', 'Date: September 27, 2026', '',
        'The research team reviewed the new document recognition system.',
        'Every result must include the original input and a clear transcript.',
        'We measure complete page latency, memory use, and character errors.',
        'Faster processing is useful only when the output remains accurate.', '',
        'Actions for next week:', '1. Check English and Chinese documents.',
        '2. Test tables, small print, and unusual page dimensions.',
        '3. Save failures and incomplete outputs with the successful results.',
        '4. Compare identical inputs on the same computer.', '',
        'Approved by: Research Operations',
    ]
    chinese_lines = [
        '文档识别测试报告', '', '日期：二零二六年九月二十七日', '',
        '本报告用于检查中文文档识别的正确性。',
        '每次测试都应保存原始图片和完整识别结果。',
        '性能指标包括处理时间、内存使用量和字符错误率。',
        '提高运行速度的同时，必须保持文字识别准确。', '',
        '测试要求：', '第一，检查标题和正文的阅读顺序。',
        '第二，检查数字、标点符号和表格内容。',
        '第三，不得忽略失败或未完成的测试。',
        '第四，在相同设备上比较相同输入。', '',
        '负责人：研究团队',
    ]
    table = ['PURCHASE ORDER', '', 'Order: PO-2026-0927', 'Currency: USD', '',
             'Item              Qty    Unit price      Amount',
             'Paper reams        12         6.25       75.00',
             'Ink cartridges      4        32.50      130.00',
             'Archive boxes      18         4.75       85.50',
             'Shipping            1        12.00       12.00', '',
             'Subtotal                               302.50',
             'Tax                                     24.20',
             'Total                                  326.70', '',
             'Payment due: October 27, 2026']
    dense = ['TECHNICAL VALIDATION RECORD', '']
    for i in range(1, 29):
        dense.extend([f'{i:02d}. Sample {1000+i}: measured value {i * 7}.25; status verified.',
                      'The original record is retained for independent checking.'])
    recipes = [('english', english, latin, 18), ('chinese', chinese_lines, chinese, 26),
               ('table', table, latin, 21), ('dense', dense, latin, 15)]
    manifest = {'schema_version': 1, 'pillow_version': pillow_version,
                'description': 'Synthetic correctness/latency corpus; not a representative production dataset.',
                'images': []}
    for name, lines, font_path, size in recipes:
        text = '\n'.join(lines)
        require_glyphs(font_path, text)
        width, height, margin = 1024, 1400, 48
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(font_path), size, index=0)
        spacing = size + 7
        right, bottom = margin, margin
        for row, line in enumerate(lines):
            y = margin + row * spacing
            bbox = draw.textbbox((margin, y), line, font=font)
            right, bottom = max(right, bbox[2]), max(bottom, bbox[3])
            if right > width - margin or bottom > height - margin:
                raise ValueError(f'{name}: text would be clipped at {bbox}')
            draw.text((margin, y), line, fill='black', font=font)
        path = output / (name + '.png')
        image.save(path, optimize=False)
        (output / (name + '.txt')).write_text(text + '\n')
        manifest['images'].append({'id': name, 'path': path.name, 'text': text,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'width': width, 'height': height,
                                  'text_bounds': [margin, margin, right, bottom],
                                  'font': str(font_path), 'font_sha256': hashlib.sha256(font_path.read_bytes()).hexdigest(),
                                  'font_size': size})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/corpus'))
    generate_corpus(parser.parse_args().output)
