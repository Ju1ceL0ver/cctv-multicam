"""Camera 1: outside/inside/doorway. Same RGB+mask API as InsideOutside."""
from pathlib import Path
from inside_outside import InsideOutside
from door_features import expanded_features

class DoorClassifier(InsideOutside):
    def __init__(self,model_path=None,threshold=.9):
        super().__init__(model_path or Path(__file__).with_name('door_classifier.pkl'),threshold)

    def _features(self,full,crop_mask,geometry,signed=None):
        return expanded_features(full,crop_mask,geometry,signed)

    def _predict(self,a):
        p=self.model.predict_proba(a[None])[0];i=int(p.argmax());names=('outside','inside','doorway')
        uncertain=bool(p[i]<self.threshold)
        return dict(label='unknown' if uncertain else names[i],raw_label=names[i],
                    confidence=float(p[i]),uncertain=uncertain,
                    probabilities={name:float(value) for name,value in zip(names,p)})
