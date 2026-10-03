"""One picture per visit, for judging it: every fragment, and every boundary question as a
pair. Each tile says which camera it is from in large coloured letters -- the two cameras
see the hall from opposite sides, and the owner's first session went wrong exactly there."""
import json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
CROPS = HERE / 'crops'
OUT = HERE / 'sheets'
OUT.mkdir(exist_ok=True)
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
PLAIN = '/System/Library/Fonts/Supplemental/Arial.ttf'
F_BIG, F_MID, F_SMALL = (ImageFont.truetype(BOLD, 26), ImageFont.truetype(BOLD, 19),
                         ImageFont.truetype(PLAIN, 16))
CAM = {'cam1': ((40, 120, 235), 'КАМЕРА 1'), 'cam2': ((235, 120, 20), 'КАМЕРА 2')}
TILE_H = 250
SHOP = timezone(timedelta(hours=7))


def hms(t):
    return datetime.fromtimestamp(t, SHOP).strftime('%H:%M:%S')


def crop_name(clip, s):
    return '%s_%s_%d_%s.webp' % (clip, s['cam'], s['frame'], '-'.join(map(str, s['box'])))


def tile(clip, shot, caption):
    path = CROPS / crop_name(clip, shot)
    colour, cam_text = CAM[shot['cam']]
    if path.exists():
        img = Image.open(path).convert('RGB')
        img = img.resize((max(1, round(img.width * TILE_H / img.height)), TILE_H))
    else:
        img = Image.new('RGB', (110, TILE_H), (40, 40, 40))
        ImageDraw.Draw(img).text((8, TILE_H // 2), 'нет кадра', font=F_SMALL, fill=(200, 200, 200))
    w = max(img.width, 150)
    out = Image.new('RGB', (w + 8, TILE_H + 62), colour)
    out.paste(img, (4 + (w - img.width) // 2, 34))
    d = ImageDraw.Draw(out)
    d.text((6, 4), cam_text, font=F_MID, fill='white')
    d.rectangle([4, TILE_H + 34, w + 4, TILE_H + 58], fill=(20, 20, 20))
    d.text((8, TILE_H + 37), caption, font=F_SMALL, fill='white')
    return out


def row(title, tiles, width=1900):
    """Tiles left to right under a title, wrapping to further lines if they do not fit."""
    lines, line, x = [], [], 0
    for t in tiles:
        if line and x + t.width + 10 > width:
            lines.append(line); line, x = [], 0
        line.append(t); x += t.width + 10
    if line:
        lines.append(line)
    h = 38 + sum(max(t.height for t in l) + 10 for l in lines)
    w = max([sum(t.width + 10 for t in l) for l in lines] + [700])
    out = Image.new('RGB', (w, h), (24, 26, 30))
    d = ImageDraw.Draw(out)
    d.text((6, 6), title, font=F_MID, fill=(230, 230, 230))
    y = 38
    for l in lines:
        x = 0
        for t in l:
            out.paste(t, (x, y)); x += t.width + 10
        y += max(t.height for t in l) + 10
    return out


def build(day, index):
    q = json.load(open(HERE / ('queue_%s.json' % day)))
    v = q['visits'][index]
    parts = []
    frag_tiles = []
    for i, f in enumerate(v['fragments']):
        for k, s in enumerate(f['shots']):
            frag_tiles.append(tile(f['clip'], s, '№%d · %s · %s' % (i + 1, ['нач', 'сер', 'кон'][k] if len(f['shots']) == 3 else k + 1, hms(f['first'] + (s['frame'] - f['shots'][0]['frame']) / 25))))
    parts.append(row('ФРАГМЕНТЫ ВИЗИТА (%d): %s' % (len(v['fragments']), ' | '.join(
        '№%d %s %s %s–%s' % (i + 1, f['clip'][-6:], f['label'], hms(f['first']), hms(f['last']))
        for i, f in enumerate(v['fragments']))), frag_tiles))
    letter = ord('A')
    for side in ('after', 'before'):
        for c in v[side]['ask']:
            edge = v['fragments'][-1] if side == 'after' else v['fragments'][0]
            ours = edge['shots'][::-1][:2] if side == 'after' else edge['shots'][:2]
            ours_tiles = [tile(edge['clip'], s, 'визит · %s' % ('конец' if side == 'after' else 'начало')) for s in ours]
            theirs = c['shots'] if side == 'after' else c['shots'][::-1]
            their_tiles = [tile(c['clip'], s, 'кандидат %s' % chr(letter)) for s in theirs[:2]]
            gap = Image.new('RGB', (40, TILE_H + 62), (24, 26, 30))
            ImageDraw.Draw(gap).text((6, TILE_H // 2), '↔', font=F_BIG, fill='white')
            parts.append(row('ГРАНИЦА %s: %s через %.1f с · %s %s%s · непохожесть %s' % (
                chr(letter), 'после ухода — появился' if side == 'after' else 'до прихода — пропал',
                c['gap_s'], c['clip'][-6:], c['label'],
                ' · ТА ЖЕ ЗАПИСЬ' if c['same_clip'] else '',
                '—' if c['distance'] is None else '%.3f' % c['distance']),
                ours_tiles + [gap] + their_tiles))
            letter += 1
    head = Image.new('RGB', (900, 44), (10, 10, 12))
    ImageDraw.Draw(head).text((8, 8), '%s · визит %d · %s · %s–%s · %.0f с' % (
        day, index + 1, v['id'][-10:], hms(v['first']), hms(v['last']), v['seconds']), font=F_BIG, fill='white')
    parts.insert(0, head)
    width = max(p.width for p in parts)
    sheet = Image.new('RGB', (width, sum(p.height + 6 for p in parts)), (24, 26, 30))
    y = 0
    for p in parts:
        sheet.paste(p, (0, y)); y += p.height + 6
    if sheet.width > 1900:
        sheet = sheet.resize((1900, round(sheet.height * 1900 / sheet.width)))
    path = OUT / ('%s_%03d.jpg' % (day, index + 1))
    sheet.save(path, quality=86)
    return path


if __name__ == '__main__':
    day = sys.argv[1]
    for i in range(int(sys.argv[2]), int(sys.argv[3])):
        print(build(day, i))
