import os
import cv2
import torch
import numpy as np
import argparse
from models.brick_detector import YOLOv7
from models.brick_classifier import AttributeClassifier
from train import preprocess_and_detect


def infer_classifier(model, detected_bricks, attributes):
    """
    Performs inference using the classifier model.

    Parameters:
    model (nn.Module): The classifier model.
    detected_bricks (np.ndarray): Array of detected and preprocessed brick images.
    attributes (np.ndarray): Array of extracted attributes.

    Returns:
    predictions (list): List of predicted classes.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(detected_bricks).float()
        attrs = torch.tensor(attributes).float()
        outputs = model(inputs, attrs)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LEGO Brick Classifier Training')
    parser.add_argument('--data_path', type=str, default='./data/brick_images', help='Path to directory containing test brick images, renderings, or so')
    parser.add_argument('--detector_path', type=str, default='./models/yolov7.pth', help='Path to directory containing pretrained object detector')
    parser.add_argument('--classifier_path', type=str, default='./models/resnet50.pth', help='Path to directory containing pretrained object classifier')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of brick classes')
    parser.add_argument('--num_attr', type=int, default=2, help='Number of brick attributes to consider. Default is 2 [color, size]')

    args = parser.parse_args()

    # Initialize YOLOv7 model
    yolo = YOLOv7()

    # Load the classifier model
    classifier_model = AttributeClassifier(num_classes=args.num_classes, attr_dim=args.num_attr)
    classifier_model.load_state_dict(torch.load(args.classifier_path))

    # Get image paths
    image_paths = [os.path.join(args.data_path, img) for img in os.listdir(args.data_path) if img.endswith('.jpg')]

    for image_path in image_paths:
        # Preprocess and detect bricks
        detected_bricks, attributes, label = preprocess_and_detect(yolo, image_path)
        original_image = cv2.imread(image_path)

        # Predict the classes of the detected bricks
        predictions = infer_classifier(classifier_model, detected_bricks, attributes)
        
        # Draw bounding boxes and predictions on the original image
        for i, detection in enumerate(yolo.detect_objects(original_image)):
            x1, y1, x2, y2, conf, cls = map(int, detection)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_image, f'Class: {predictions[i]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Save or display the result
        cv2.imwrite(f'results/{os.path.basename(image_path)}', original_image)
        cv2.imshow('Detected Bricks', original_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
