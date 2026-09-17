import cv2, numpy as np
from seams import seam_response

def segments(img, min_len=45):
    bh = seam_response(img)
    g = np.clip(bh * 8, 0, 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(g)[0]
    if segs is None:
        return np.empty((0, 4))
    segs = segs.reshape(-1, 4)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[L >= min_len]

if __name__ == '__main__':
    for cam in ('cam1', 'cam2'):
        img = cv2.imread('data/%s_bg.jpg' % cam)
        s = segments(img)
        vis = (img * 0.45).astype(np.uint8)
        for x1, y1, x2, y2 in s:
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            col = tuple(int(c) for c in cv2.applyColorMap(np.uint8([[ang / 180 * 255]]), cv2.COLORMAP_HSV)[0, 0])
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
        cv2.imwrite('data/%s_lsd.jpg' % cam, cv2.resize(vis, (1280, 720)))
        print(cam, len(s), 'segments >= 45 px')
