from data import to_vector
from model import System


class RegressorInterface:
    def predict(self, system: System):
        raise NotImplementedError


class SKRegressor(RegressorInterface):
    def __init__(self, model=None, model_file=None):
        if model is None and model_file:
            from joblib import load
            self.model = load(model_file)
        elif model:
            self.model = model

    def predict(self, system: System):
        vector, = to_vector(system)
        prediction = self.model.predict(vector)
        return prediction
