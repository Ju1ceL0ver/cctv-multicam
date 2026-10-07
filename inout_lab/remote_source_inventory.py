# Read-only source availability; isolated names avoid affecting running kernels.
def _inout_inventory_0710():
    import sys,json
    from pathlib import Path
    root=Path(r'C:\Users\ArykovAA\cctv_ai\multicam')
    sys.path.insert(0,str(root))
    import rate,numpy as np
    z=np.load(root/'data/inout/io_cam1.npz');ids=z['ids'];samples=json.loads((root/'data/inout/samples.json').read_text(encoding='utf-8'));frames=rate.index(root)
    available=sum(str(i) in samples and samples[str(i)]['frame'] in frames for i in ids)
    print(json.dumps({'export_rows':len(ids),'available_sources':available,'unique_frames':len({samples[str(i)]['frame'] for i in ids if str(i) in samples}),'source_image_bytes':sum(frames[f][0].stat().st_size for f in {samples[str(i)]['frame'] for i in ids if str(i) in samples} if f in frames)}))
_inout_inventory_0710()
