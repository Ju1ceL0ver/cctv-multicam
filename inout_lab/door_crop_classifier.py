"""Camera 1 classifier with detailed crop appearance as well as global context."""
from pathlib import Path
import numpy as np
from door_classifier import DoorClassifier
from door_features import expanded_features
from crop_feature_search import crop_features

class CropDoorClassifier(DoorClassifier):
    requires_crop_rgb=True
    def __init__(self,model_path=None,threshold=0):
        super().__init__(model_path or Path(__file__).with_name('door_crop_classifier.pkl'),threshold)
    def predict_prepared(self,full,crop,geometry):
        if crop.ndim!=3 or crop.shape[2]<5:
            raise ValueError('Detailed crop must contain RGB, person mask and floor channels')
        return self._predict_inputs(full,crop[:,:,3],geometry,crop_rgb=crop[:,:,:3],crop_floor=crop[:,:,4])
    def _predict_inputs(self,full,crop_mask,geometry,signed=None,crop_rgb=None,crop_floor=None):
        if crop_rgb is None or crop_floor is None:raise ValueError('Detailed crop channels required')
        crop=np.dstack([crop_rgb,crop_mask,crop_floor]);a=np.concatenate([expanded_features(full,crop_mask,geometry,signed),crop_features(crop)])
        return self._predict(a)
