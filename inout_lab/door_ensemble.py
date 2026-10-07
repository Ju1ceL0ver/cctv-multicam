"""Equal-probability ensemble; imported by the trusted local pickle model."""
from threadpoolctl import threadpool_limits

class DoorEnsemble:
    def __init__(self, forest, boost):
        self.forest=forest
        self.boost=boost
        self.classes_=forest.classes_
        self.n_jobs=1

    def predict_proba(self,features):
        self.forest.n_jobs=1
        with threadpool_limits(limits=1, user_api="openmp"):
            return (self.forest.predict_proba(features)+self.boost.predict_proba(features))/2
