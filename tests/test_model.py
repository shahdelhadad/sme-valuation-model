import os
import sys
import unittest
import numpy as np
import joblib

# Ensure src is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.preprocess import load_data, preprocess_data

class TestModelDeployment(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.preprocessor = preprocess_data(cls.df)
        cls.pipeline = None
        
        if os.path.exists(config.MODEL_PKL_PATH):
            cls.pipeline = joblib.load(config.MODEL_PKL_PATH)

    def test_model_exists(self):
        self.assertTrue(os.path.exists(config.MODEL_PKL_PATH), 
                        "Trained model not found. Run train.py first.")
        self.assertIsNotNone(self.pipeline, "Pipeline failed to load.")

    def test_prediction_shape(self):
        if self.pipeline is None:
            self.skipTest("Model file not found.")
            
        # Take 5 samples
        sample_X = self.X_test.head(5)
        predictions = self.pipeline.predict(sample_X)
        
        self.assertEqual(len(predictions), 5, "Prediction length doesn't match input length.")
        self.assertTrue(isinstance(predictions, np.ndarray), "Predictions should be a numpy array.")

    def test_onnx_export_exists(self):
        self.assertTrue(os.path.exists(config.MODEL_ONNX_PATH), 
                        "ONNX model not found. Run export_onnx.py first.")

if __name__ == '__main__':
    unittest.main()
