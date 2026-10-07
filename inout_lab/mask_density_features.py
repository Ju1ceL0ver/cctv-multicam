"""Source-scale area and occupancy cues; no labels or learned depth model."""
import numpy as np

def density_features(rich,head,geometry):
    area=np.maximum(rich[:,9],1e-8)*1280*720
    width=np.maximum(geometry[:,4]*1280,1);height=np.maximum(geometry[:,5]*720,1)
    parts=[np.column_stack([np.log1p(area),np.sqrt(area),area/(width*height),height/width,width,height])]
    for j in range(4):
        h=head[:,j*8:j*8+8];head_area=np.maximum(h[:,2],1e-8)*1280*720
        span=np.maximum((h[:,6]-h[:,4])*1280,1)
        parts.append(np.column_stack([np.log1p(head_area),span,head_area/(span*span),head_area/area,area/(span*span),height/span,width/span,np.sqrt(area)/span]))
    result=np.column_stack(parts).astype(np.float32)
    assert result.shape[1]==38 and np.isfinite(result).all()
    return result
