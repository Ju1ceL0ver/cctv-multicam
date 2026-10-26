"""CPU portrait video of observed customer fragments; missing footage is shown explicitly."""
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np


def portrait(raw, box):
    """Boxes use original camera pixels, never coordinates of a resized preview."""
    import cv2
    height_px, width_px = raw.shape[:2]
    height = min(float(height_px),max(120.,2*(box[3]-box[1])))
    width = min(float(width_px),height*0.75)
    cx,cy = (box[0]+box[2])/2,(box[1]+box[3])/2
    x,y = int(np.clip(cx-width/2,0,width_px-width)), int(np.clip(cy-height/2,0,height_px-height))
    return cv2.resize(raw[y:y+int(height),x:x+int(width)],(360,480))


def render(day, ident, sources, folder, fps=5):
    import cv2
    import imageio_ffmpeg
    import rawsource
    if not sources:
        return {'status': 'missing_detection_provenance'}
    sources = sorted(sources, key=lambda s: s['t'])
    times = np.array([s['t'] for s in sources]); t0, t1 = times[0], times[-1]
    temp = folder / (ident + '.tmp.mp4'); mp4 = folder / (ident + '.mp4'); gif = folder / (ident + '.gif')
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (360, 480))
    if not writer.isOpened():
        raise RuntimeError('VideoWriter failed')
    caps, counts, last_box, last_cam, gaps = {}, {}, None, None, 0
    segments = {c: rawsource.segments(c, day) for c in ('cam1', 'cam2')}
    midnight = datetime.strptime(day, '%Y%m%d')
    try:
        for t in np.arange(t0, t1 + 0.01, 1/fps):
            j = np.searchsorted(times, t)
            nearby = [k for k in range(max(0,j-4), min(len(sources),j+5)) if abs(times[k]-t) <= 0.5]
            frame = None
            if nearby:
                k = max(nearby, key=lambda k: (sources[k]['box'][3]-sources[k]['box'][1]) - 50*abs(times[k]-t))
                source = sources[k]; cam = source['cam']
                wall = midnight + timedelta(seconds=source['raw_t'] + t - source['t'])
                for path, began in segments[cam]:
                    index = int(round((wall-began).total_seconds()*rawsource.FPS))
                    if not 0 <= index < rawsource.SEG*rawsource.FPS:
                        continue
                    if path not in caps:
                        caps[path] = cv2.VideoCapture(path)
                        counts[path] = caps[path].get(cv2.CAP_PROP_FRAME_COUNT)
                    if index >= counts[path]:
                        continue
                    cap = caps[path]
                    position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if 0 <= index-position <= 50:
                        for _ in range(index-position):
                            if not cap.grab():
                                break
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                    ok, raw = cap.read()
                    if not ok:
                        continue
                    # detect_raw stores boxes in the original camera pixels.
                    box = np.array(source['box'],float)
                    if last_box is not None and last_cam == cam:
                        box = 0.8*last_box + 0.2*box
                    last_box, last_cam = box, cam
                    frame = portrait(raw, box)
                    break
            if frame is None:
                frame = np.zeros((480,360,3),np.uint8); gaps += 1
                cv2.putText(frame,'No observation',(55,240),cv2.FONT_HERSHEY_SIMPLEX,.65,(200,200,200),1)
                last_box, last_cam = None,None
            hms = (midnight+timedelta(seconds=float(t))).strftime('%H:%M:%S')
            cv2.putText(frame,hms,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            writer.write(frame)
    finally:
        writer.release()
        for cap in caps.values(): cap.release()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run([ff,'-y','-loglevel','error','-i',str(temp),'-c:v','libx264','-preset','fast','-crf','26',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],check=True,capture_output=True)
        subprocess.run([ff,'-y','-loglevel','error','-i',str(mp4),'-t','15','-vf',
                        'fps=3,scale=180:240:flags=lanczos',str(gif)],check=True,capture_output=True)
    finally:
        temp.unlink(missing_ok=True)
    return {'status':'ready','mp4':mp4.name,'gif':gif.name,'missing_frames':gaps,'fps':fps,
            'start':float(t0),'end':float(t1)}


def prune_media(root, now, days=30):
    """Expire only generated media named by old visit manifests; preserve all labels."""
    from storage import read_json
    import re
    cutoff = (now-timedelta(days=days)).strftime('%Y%m%d')
    removed = 0
    for folder in (Path(root)/'data/visits').glob('*'):
        if not re.fullmatch(r'\d{8}',folder.name) or folder.name >= cutoff:
            continue
        for record in read_json(folder/'index.json',{}).get('visits',[]):
            ident = record.get('id','')
            if not re.fullmatch(r'v_[A-Za-z0-9_]+',ident):
                continue
            for ext in ('mp4','gif'):
                name = record.get('media',{}).get(ext)
                if name == ident+'.'+ext and (folder/name).exists():
                    (folder/name).unlink(); removed += 1
    return removed
