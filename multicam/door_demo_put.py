import numpy as np


def put(img, labels, banners, footer):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img[:, :, ::-1]); d = ImageDraw.Draw(im)
    def font(n):
        for f in ('arialbd.ttf', 'arial.ttf'):
            try:
                return ImageFont.truetype(f, n)
            except OSError:
                pass
        return ImageFont.load_default()
    f20, f34, f22 = font(20), font(34), font(22)
    for (x, y), t, col in labels:
        l, tp, r, b = d.textbbox((x, y), t, font=f20); d.rectangle([l - 3, tp - 2, r + 3, b + 2], fill=(0, 0, 0))
        d.text((x, y), t, fill=tuple(int(c) for c in col[::-1]), font=f20)
    for i, (t, col) in enumerate(banners):
        y = 16 + 50 * i
        l, tp, r, b = d.textbbox((20, y), t, font=f34); d.rectangle([l - 6, tp - 4, r + 6, b + 4], fill=(0, 0, 0))
        d.text((20, y), t, fill=tuple(int(c) for c in col[::-1]), font=f34)
    l, tp, r, b = d.textbbox((20, 690), footer, font=f22); d.rectangle([l - 4, tp - 3, r + 4, b + 3], fill=(0, 0, 0))
    d.text((20, 690), footer, fill=(255, 255, 255), font=f22)
    return np.asarray(im)[:, :, ::-1].copy()
