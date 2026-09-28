"""Compare changed OCR boxes against frozen synthetic rendering bounds."""
import hashlib
import json
from pathlib import Path
import re
import statistics

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
from tools.benchmark import validate_image
from tools.score import normalize_ocr

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '2026-09-28-current-q8/quality'
NEW = ROOT / 'quality/q8'
BLOCK = re.compile(r'<\|det\|>([^<\n]*?)\[(\d+), (\d+), (\d+), (\d+)\]<\|/det\|>(.*?)(?=<\|det\|>|\Z)', re.S)


def scores(box, truth):
    intersection = max(0, min(box[2],truth[2])-max(box[0],truth[0])) * max(0,min(box[3],truth[3])-max(box[1],truth[1]))
    area = lambda b: max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union = area(box)+area(truth)-intersection
    assert union > 0
    return dict(iou=intersection/union, mean_absolute_edge_error_pixels=sum(abs(a-b) for a,b in zip(box,truth))/4)


def main():
    assert scores([0,0,10,10],[0,0,10,10]) == dict(iou=1.,mean_absolute_edge_error_pixels=0.)
    assert scores([0,0,10,10],[5,0,15,10]) == dict(iou=1/3,mean_absolute_edge_error_pixels=2.5)
    assert scores([0,0,10,10],[20,0,30,10]) == dict(iou=0.,mean_absolute_edge_error_pixels=10.)
    truths = {}
    manifests = []
    for directory in [Path('benchmarks/corpus'), Path('benchmarks/heldout')]:
        path = directory/'manifest.json'
        manifest = json.loads(path.read_text())
        manifests.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for item in manifest['images']:
            image = (directory/item['path']).resolve()
            validate_image(image,item['sha256'])
            lines = item['text'].splitlines()
            if 'line_bounds' in item:
                bounds = item['line_bounds']
            else:
                # Match the frozen tools.corpus recipe; do not regenerate images.
                assert manifest['pillow_version'] == pillow_version
                font_path = Path(item['font'])
                assert hashlib.sha256(font_path.read_bytes()).hexdigest() == item['font_sha256']
                font = ImageFont.truetype(str(font_path),item['font_size'],index=0)
                draw = ImageDraw.Draw(Image.new('RGB',(1,1)))
                bounds = [draw.textbbox((48,48+i*(item['font_size']+7)),line,font=font) for i,line in enumerate(lines)]
            assert len(lines) == len(bounds)
            truths[str(image)] = (item,[(normalize_ocr(line),box) for line,box in zip(lines,bounds) if line.strip()])
    before = [json.loads(x)['result'] for x in (BASE/'responses.jsonl').read_text().splitlines()]
    after = [json.loads(x)['result'] for x in (NEW/'responses.jsonl').read_text().splitlines()]
    cases = json.loads((BASE/'process.json').read_text())['cases']
    results = []
    for case,old,new in zip(cases,before,after):
        if old['text'] == new['text']:
            continue
        item,lines = truths[case['image']]
        a,b = list(BLOCK.finditer(old['text'])),list(BLOCK.finditer(new['text']))
        assert a and len(a)==len(b)
        cursor, blocks = 0,[]
        for index,(left,right) in enumerate(zip(a,b)):
            text = normalize_ocr(left[6])
            assert text == normalize_ocr(right[6]) and left[1]==right[1]
            ends = [end for end in range(cursor+1,len(lines)+1) if ' '.join(line for line,_ in lines[cursor:end])==text]
            assert len(ends)==1,(case['case'],index,text)
            end=ends[0]
            boxes=[box for _,box in lines[cursor:end]]
            truth=[min(box[0] for box in boxes),min(box[1] for box in boxes),max(box[2] for box in boxes),max(box[3] for box in boxes)]
            # Follow upstream rendering: coordinates divide by 999, then truncate.
            convert=lambda m:[int(int(m[i])/999*dimension) for i,dimension in zip(range(2,6),[item['width'],item['height']]*2)]
            left_box,right_box=convert(left),convert(right)
            blocks.append(dict(block=index,rendered_line_span=[cursor,end],truth=truth,before_box=left_box,
                               after_box=right_box,before=scores(left_box,truth),after=scores(right_box,truth)))
            cursor=end
        assert cursor==len(lines),(case['case'],cursor,len(lines))
        summary={name:{metric:statistics.mean(block[name][metric] for block in blocks)
                       for metric in ['iou','mean_absolute_edge_error_pixels']} for name in ['before','after']}
        results.append(dict(case=case['case'],blocks=len(blocks),mean=summary,details=blocks))
    report=dict(manifests=manifests,coordinate_projection='int(value / 999 * image dimension), matching upstream draw',
        bounds='Rendered typographic text bounds; grouped table regions use unions of contiguous lines.',
        limits='Synthetic text layout only, not a general document layout benchmark. Cases selected because raw outputs changed.',cases=results)
    (ROOT/'layout-scores.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='details'} for row in results],indent=2))


if __name__=='__main__':
    main()
