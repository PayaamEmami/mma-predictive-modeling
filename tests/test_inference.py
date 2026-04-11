import unittest
import numpy as np
from sklearn.preprocessing import LabelEncoder

from code.inference import ModelInference


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _features):
        return np.array([self.prediction])

    def predict_proba(self, _features):
        probabilities = np.array([[0.8, 0.2]])
        if self.prediction == 1:
            probabilities = np.array([[0.2, 0.8]])
        return probabilities


class TestInference(unittest.TestCase):
    def test_prediction_decodes_label_encoder_to_fighter_slot(self):
        inference = ModelInference.__new__(ModelInference)
        inference.models = {
            "Fighter1 Model": FakeModel(0),
            "Fighter2 Model": FakeModel(1),
        }
        inference.label_encoder = LabelEncoder().fit(["1", "2"])

        predictions = ModelInference.predict_with_features(
            inference, np.array([[0.0, 1.0]])
        )

        self.assertEqual(predictions["Fighter1 Model"]["winner_label"], "1")
        self.assertEqual(predictions["Fighter1 Model"]["winner"], "Fighter 1")
        self.assertEqual(predictions["Fighter2 Model"]["winner_label"], "2")
        self.assertEqual(predictions["Fighter2 Model"]["winner"], "Fighter 2")


if __name__ == "__main__":
    unittest.main()
