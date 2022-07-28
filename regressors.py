from data import to_vector, to_prediction_vector
from model import System


class RegressorInterface:
    def predict(self, system: System):
        raise NotImplementedError


class SKRegressor(RegressorInterface):
    def __init__(self, shape, model=None, model_file=None):
        self.shape = shape
        if model:
            self.model = model
        elif model_file:
            from joblib import load
            self.model = load(model_file)

    def predict(self, system: System):
        vector = to_prediction_vector(system, self.shape)
        prediction = self.model.predict(vector)
        return prediction
