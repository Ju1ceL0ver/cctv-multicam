"""Verify checkpoint/prepared parity and original-image RGB+mask API parity.
No training, calibration or threshold changes. Uses held19.09 source sample#970.
"""
from pathlib import Path
import json,numpy as np,cv2
from dual_view_predictor import DualViewPredictor
root=Path(__file__).parent
if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');saved=np.load(root/'diagnostics/dualview_20260919_predictions.npz');indices=saved['indices'];prob=saved['probabilities'];model=DualViewPredictor(root/'diagnostics/dualview_20260919.pt');probe=[int(indices[0]),970,1149];F=z['F'][probe];X=z['X'][probe];G=z['G'][probe,1:];errors=[];answers={}
 for j,i in enumerate(probe):
  result=model.predict_prepared(F[j],X[j],G[j]);p=np.array(list(result['probabilities'].values()));errors.append(float(np.max(np.abs(p-prob[np.where(indices==i)[0][0]]))));answers[str(i)]=result
 assert max(errors)<1e-5,errors
 rgb=cv2.cvtColor(cv2.imread(str(root/'diagnostics/original_970.jpg')),cv2.COLOR_BGR2RGB);draft=cv2.imread(str(root/'diagnostics/original_970_draft.png'),cv2.IMREAD_UNCHANGED);assert rgb.shape[:2]==draft.shape
 meta=json.loads((root/'diagnostics/source_samples.json').read_text())[str(z['ids'][970])];mask=draft==meta['value'];raw=model.predict_rgb(rgb,mask,box=meta['box'],foot=meta['foot'],floor_px=meta['floor_px']);raw_p=np.array(list(raw['probabilities'].values()));prepared_p=np.array(list(answers['970']['probabilities'].values()));delta=float(np.max(np.abs(raw_p-prepared_p)))
 report={'held_day':model.held_day,'prepared_checkpoint_max_difference':max(errors),'original_rgb_mask_probability_difference':delta,'raw_rgb_mask_answer':raw,'prepared_answers':answers,'original_input_shape':rgb.shape,'target_mask_pixels':int(mask.sum())};(root/'diagnostics/dualview_20260919_api_audit.json').write_text(json.dumps(report,indent=2));print('DUAL_VIEW_API',report,flush=True);assert delta<1e-5,'RGB preparation does not match export; inspect preprocessing before use'
