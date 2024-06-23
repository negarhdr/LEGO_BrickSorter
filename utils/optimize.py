import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from models.brick_classifier import AttributeClassifier

# Configuration
NUM_CLASSES = 10
ATTR_DIM = 2  # Example: [color, size]
MODEL_PATH = 'best_classifier_model.pth'
ONNX_MODEL_PATH = 'classifier_model.onnx'
OPTIMIZED_ONNX_MODEL_PATH = 'optimized_classifier_model.onnx'

def load_model(model_path):
    """
    Load the trained PyTorch model.

    Parameters:
    model_path (str): Path to the trained model.

    Returns:
    model (nn.Module): Loaded PyTorch model.
    """
    model = AttributeClassifier(num_classes=NUM_CLASSES, attr_dim=ATTR_DIM)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def convert_to_onnx(model, onnx_model_path):
    """
    Convert the PyTorch model to ONNX format.

    Parameters:
    model (nn.Module): Trained PyTorch model.
    onnx_model_path (str): Path to save the ONNX model.
    """
    dummy_input = torch.randn(1, 3, 64, 64)  # Example input shape (batch_size, channels, height, width)
    dummy_attrs = torch.randn(1, ATTR_DIM)  # Example attributes shape (batch_size, attr_dim)
    torch.onnx.export(model, (dummy_input, dummy_attrs), onnx_model_path, 
                      input_names=['input', 'attributes'], 
                      output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'attributes': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

def optimize_onnx_model(onnx_model_path, optimized_onnx_model_path):
    """
    Optimize the ONNX model.

    Parameters:
    onnx_model_path (str): Path to the ONNX model.
    optimized_onnx_model_path (str): Path to save the optimized ONNX model.
    """
    session_options = onnxruntime.SessionOptions()
    session_options.optimized_model_filepath = optimized_onnx_model_path
    _ = onnxruntime.InferenceSession(onnx_model_path, session_options)

def verify_onnx_model(onnx_model_path):
    """
    Verify the ONNX model.

    Parameters:
    onnx_model_path (str): Path to the ONNX model.
    """
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

if __name__ == "__main__":
    # Load the trained model
    model = load_model(MODEL_PATH)
    
    # Convert to ONNX format
    convert_to_onnx(model, ONNX_MODEL_PATH)
    print(f"Model has been converted to ONNX format and saved to {ONNX_MODEL_PATH}")
    
    # Optimize the ONNX model
    optimize_onnx_model(ONNX_MODEL_PATH, OPTIMIZED_ONNX_MODEL_PATH)
    print(f"ONNX model has been optimized and saved to {OPTIMIZED_ONNX_MODEL_PATH}")
    
    # Verify the optimized ONNX model
    verify_onnx_model(OPTIMIZED_ONNX_MODEL_PATH)
    print("Optimized ONNX model verification completed successfully.")
