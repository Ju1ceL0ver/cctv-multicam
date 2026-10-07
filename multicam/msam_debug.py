import sys, numpy as np, torch, cv2
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import micro_sam as MS, sam31_lite_eval as SL
dev = 'cuda'
m = MS.MicroSAM().to(dev); m.load_state_dict(torch.load(R / 'runs' / 'msam_a' / 'last.pt', map_location='cpu', weights_only=False)['model']); m.eval()
d, k0, dens = SL.stretches(2, 96)[0]
T = SL.load_res(R / 'runs' / 's31micro_a' / 'evals' / 'teacher_cache' / ('%s_%s_%d_%d.pkl' % (d.parent.name, d.name, k0, 96)))
cap = cv2.VideoCapture(str(d / 'video.mp4')); cap.set(cv2.CAP_PROP_POS_FRAMES, k0 + 10); ok, f = cap.read()
x = np.zeros((MS.HP, MS.W, 3), np.uint8); x[:MS.H] = cv2.resize(f, (MS.W, MS.H))[:, :, ::-1]
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    g16, g8, g4 = m.features(torch.from_numpy(x).to(dev).permute(2, 0, 1)[None].float() / 255)
    det = m.detect(g16, g8, g4)
sc = det['pred_logits'][0, :, 0].float().sigmoid()
print('teacher people on this frame', len(T.get(10, [])))
print('top scores', [round(float(v), 3) for v in sc.sort(descending=True).values[:10]])
dm = det['pred_masks'][0].float().sigmoid()
top = sc.argsort(descending=True)[:5]
print('mask px >0.5 of top', [int((dm[i][:180] > 0.5).sum()) for i in top], 'max prob', [round(float(dm[i].max()), 3) for i in top])
print('boxes top', det['pred_boxes'][0, top].float().cpu().numpy().round(3).tolist())
