import os
import numpy as np
import xgboost as xgb

_MODEL_PATH = os.path.join('/opt/ml/', 'model')  # Path where all your model(s) live in

class ModelService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = xgb.Booster({'nthread': 4})
            cls.model.load_model(os.path.join(_MODEL_PATH, 'model.joblib'))
            print('Model loaded')
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        clf = cls.get_model()
        return clf.predict(input)


def predict(json_input):
    """
    Prediction given the request input
    :param json_input: [dict], request input
    :return: [dict], prediction
    """
    data = json_input['features']
    print(data)
    data = np.array(data)
    data = xgb.DMatrix(data)
    prediction = ModelService.predict(data)
    print(prediction)
    return {
       "prediction": prediction.tolist()
    }
