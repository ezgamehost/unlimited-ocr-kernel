"""Freeze synthetic pages not used to select the existing kernel optimizations."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__
from tools.corpus import require_glyphs

ROOT = Path(__file__).resolve().parent
FONTS = Path('/System/Library/Fonts')


def main():
    receipt = '''HARBOR STATIONERY
Receipt R-08421
2026-09-28 09:35

Description          Qty    Total
Pocket notebook        3    9.75
Blue ink pen           5    7.50
Document folder        2    4.20

Subtotal                   21.45
Tax                         1.72
Amount paid                23.17
Card ending                 6038

Items may be returned within
30 days with this receipt.
Thank you for your visit.'''
    wide = '''WATER QUALITY FIELD REPORT
Station: Willow Creek East     Survey: WC-1186     Date: 28 September 2026

The morning survey covered the upstream bridge and the southern footpath.
Three sealed samples were collected before the first scheduled discharge.
Sample A: 08:15, temperature 18.4 C, pH 7.2, conductivity 412 uS/cm.
Sample B: 08:27, temperature 18.6 C, pH 7.1, conductivity 409 uS/cm.
Sample C: 08:42, temperature 18.5 C, pH 7.3, conductivity 415 uS/cm.

No visible surface film was observed. The water appeared clear at all locations.
Each sample bottle was labelled, photographed, and placed in the transport case.
Laboratory results are pending; these observations are not a safety certification.

Prepared by: Field Operations     Reviewed by: Laboratory Services'''
    square = '''社区图书馆开放通知
COMMUNITY LIBRARY NOTICE

开放日期：二零二六年十月五日
Opening date: 5 October 2026

周一至周五：上午九点至下午六点
Saturday: 10:00 to 16:00
星期日闭馆。

一楼设有借阅服务台和报刊阅读区。
二楼提供自习座位和小组讨论室。
请保持安静，并将手机调至静音。

Borrow up to six books for three weeks.
Return books at the main entrance desk.
Please bring your library card.

咨询电话：010-5550-2186
服务邮箱：library@example.org
欢迎居民和访客使用图书馆服务。'''
    dense = ['ARCHIVE TRANSFER REGISTER', 'Batch AT-5603 / Storage room B7', '']
    destinations = ['north shelf', 'sealed cabinet', 'review desk', 'map drawer']
    for index in range(1, 27):
        dense.extend([f'{index:02d}. File AX-{7310 + index}: {11 + (index * 13) % 79} sheets; destination: {destinations[index % 4]}.',
                      f'    Seal {92000 + index * 17}; received at 14:{index:02d}; condition checked by clerk {index % 3 + 1}.'])
    recipes = [
        ('receipt', receipt, (480, 640), FONTS / 'Menlo.ttc', 16, 9, 32, [1, 1], 273),
        ('wide_report', wide, (1280, 640), FONTS / 'Supplemental/Georgia.ttf', 23, 12, 40, [2, 1], 483),
        ('bilingual_notice', square, (960, 960), FONTS / 'Supplemental/Arial Unicode.ttf', 25, 14, 40, [2, 2], 693),
        ('archive_register', '\n'.join(dense), (1024, 1536), FONTS / 'Supplemental/Georgia.ttf', 17, 8, 40, [2, 3], 903),
    ]
    manifest = dict(schema_version=1, pillow_version=__version__,
        description='Frozen synthetic validation pages, unused by prior kernel tuning. Not representative production scans. Do not revise content in response to OCR output.', images=[])
    for name, text, size, font_path, font_size, gap, margin, grid, tokens in recipes:
        require_glyphs(font_path, text)
        image = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(font_path), font_size, index=0)
        bounds = []
        for row, line in enumerate(text.splitlines()):
            xy = (margin, margin + row * (font_size + gap))
            box = draw.textbbox(xy, line, font=font)
            if box[0] < 0 or box[1] < 0 or box[2] > size[0] - margin or box[3] > size[1] - margin:
                raise ValueError(f'{name}: clipped text at {box}')
            bounds.append(box)
            draw.text(xy, line, font=font, fill='black')
        path = ROOT / f'{name}.png'
        image.save(path, optimize=False)
        (ROOT / f'{name}.txt').write_text(text + '\n')
        manifest['images'].append(dict(id=name, path=path.name, text=text,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), width=size[0], height=size[1],
            expected_crop_grid=grid, expected_image_tokens=tokens, line_bounds=bounds,
            font=str(font_path), font_sha256=hashlib.sha256(font_path.read_bytes()).hexdigest(), font_size=font_size))
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({x['id']: x['sha256'] for x in manifest['images']}, indent=2))


if __name__ == '__main__':
    main()
