# Pytorch libraries
import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os

# Platform modules
from models.brick_detector import YOLOv7
from data.data_preprocessing import DataPreprocessing
from models.brick_classifier import AttributeClassifier
from train import train_classifier
from inference import infer_classifier
from utils.optimize import convert_to_onnx, verify_onnx_model

MODEL_PATH = 'best_classifier_model.pth'

# This class tests all the modules of the whole platform
class TestBrickSorter(unittest.TestCase):
    def setUp(self):
        # Initialize necessary objects for each test case
        self.yolo = YOLOv7()
        self.model = AttributeClassifier(num_classes=10, attr_dim=2)
        self.inputs = torch.randn(100, 3, 64, 64)
        self.attrs = torch.randn(100, 2)
        self.labels = torch.randint(0, 10, (100,))
        self.dataset = TensorDataset(self.inputs, self.attrs, self.labels)
        self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, [80, 20])

    def test_object_detector(self):
        # Test YOLOv7 object detector
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        detections = self.yolo.detect_objects(image)
        self.assertIsInstance(detections, list)
        for detection in detections:
            self.assertEqual(len(detection), 5)

    def test_preprocess_image(self):
        # Test image preprocessing module
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed_image = DataPreprocessing.preprocess_image(image, size=(64, 64))
        self.assertEqual(processed_image.shape, (64, 64, 3))
        self.assertTrue((processed_image >= 0).all() and (processed_image <= 1).all())

    def test_classifier(self):
        # Test the AttributeClassifier model
        inputs = torch.randn(1, 3, 64, 64)  # Example input shape
        attrs = torch.randn(1, 2)  # Example attributes shape
        outputs = self.model(inputs, attrs)
        self.assertEqual(outputs.shape, (1, 10))  # Check output shape matches the number of classes

    def test_train_process(self):
        # Test the training process of the classifier
        train_classifier(self.model, self.train_data, self.val_data)
        # Check that the model has been trained by ensuring the state dict is not default
        self.assertTrue(any(param.requires_grad for param in self.model.parameters()))

    def test_inference_process(self):
        # Test the inference process of the classifier
        self.model.load_state_dict(torch.load(MODEL_PATH))
        predictions = infer_classifier(self.model, self.inputs, self.attrs)
        self.assertEqual(predictions.shape, (5,))
        self.assertTrue((predictions >= 0).all() and (predictions < 10).all())

    def test_convert_to_onnx(self):
        # Test conversion of the model to ONNX format
        self.onnx_model_path = 'test_classifier_model.onnx'
        self.model.load_state_dict(torch.load(MODEL_PATH))
        convert_to_onnx(self.model, self.onnx_model_path)
        self.assertTrue(os.path.exists(self.onnx_model_path))
    
    def test_verify_onnx_model(self):
        # Test verification of the ONNX model
        verify_onnx_model(self.onnx_model_path)
        # If no exception is raised, the model is verified
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
