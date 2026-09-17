"""Tile-seam enhancement and floor mask for a median background image."""
import cv2, numpy as np, sys

def floor_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # tiles: low saturation, mid-high brightness; racks are dark or saturated wood
    m = ((s < 45) & (v > 110) & (v < 225)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 60000:
            keep[lab == i] = 255
    return keep

def seam_response(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    # seams are thin DARK lines: black-hat picks structures darker than surroundings
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return bh

if __name__ == '__main__':
    for cam in ('cam1', 'cam2'):
        img = cv2.imread('data/%s_bg.jpg' % cam)
        m = floor_mask(img)
        bh = seam_response(img)
        bhm = bh * (m > 0)
        vis = img.copy(); vis[m == 0] = (vis[m == 0] * 0.35).astype(np.uint8)
        cv2.imwrite('data/%s_mask.jpg' % cam, cv2.resize(vis, (1280, 720)))
        norm = np.clip(bhm * 6, 0, 255).astype(np.uint8)
        cv2.imwrite('data/%s_seams.jpg' % cam, cv2.resize(norm, (1280, 720)))
        print(cam, 'floor px %.1f%%' % (100 * (m > 0).mean()))
