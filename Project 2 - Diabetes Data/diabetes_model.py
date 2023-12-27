# diabetes_model.py

import pickle


class DiabetesModel:
    def __init__(self, model_path='model.pickle', scaler_path='scaler.pickle'):
        with open(model_path, 'rb') as f2:
            self.model = pickle.load(f2)

        with open(scaler_path, 'rb') as f1:
            self.scaler = pickle.load(f1)

    def predict(self, features):
        # Add any necessary pre-processing steps
        scaled_features = self.scaler.transform(features)

        prediction = self.model.predict(scaled_features)
        return prediction
