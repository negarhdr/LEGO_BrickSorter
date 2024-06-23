'''
Training process for detecting bricks in collected images and train/finetune a classifier 
to sort detected bricks into different classes. 
'''


import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data.data_preprocessing import DataPreprocessing
from data.dataset import LegoBrickDataset
from models.brick_detector import YOLOv7
from models.brick_classifier import SimpleClassifier, ResNetClassifier, AttributeClassifier
from torch import nn, optim
import argparse


def extract_attributes(detection, image):
    """
    Extracts attributes from the detected object.

    Parameters:
    detection (list): Bounding box info of the detection.
    image (np.ndarray): Original image.

    Returns:
    attrs (np.ndarray): Array of extracted attributes.
    """
    x1, y1, x2, y2, conf = map(int, detection)
    crop_img = image[y1:y2, x1:x2]

    # Example attribute extraction (color, size)
    color = cv2.mean(crop_img)[:3]  # Average color
    size = [(x2 - x1) * (y2 - y1)]  # Area as size

    return np.array(color + size)

def preprocess_and_detect(yolo, data_loader):
    """
    Preprocesses images and detects objects using YOLOv7.

    Parameters:
    yolo (YOLOv7): Initialized YOLOv7 object detector.
    image_paths (list): List of paths to images.

    Returns:
    detected_bricks (np.ndarray): Array of detected and preprocessed brick images.
    attributes (np.ndarray): Array of extracted attributes.
    labels (np.ndarray): Array of corresponding labels (mock data in this example).
    """
    detected_bricks = []
    attributes = []
    labels = []

    for batch_idx, (img, lbl) in enumerate(data_loader):
        image = cv2.imread(img)
        detections = yolo.detect_objects(image)
        for detection in detections:
            x1, y1, x2, y2, conf = map(int, detection)
            crop_img = image[y1:y2, x1:x2]
            crop_img = DataPreprocessing.preprocess_image(crop_img, size=(64, 64))
            crop_img = np.transpose(crop_img, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
            detected_bricks.append(crop_img)
            attrs = extract_attributes(detection, image)
            attributes.append(attrs)
            labels.append(lbl)

    detected_bricks = np.array(detected_bricks)
    attributes = np.array(attributes)

    return detected_bricks, attributes, labels

def train_classifier(model, train_dataset, val_dataset, epochs, learning_rate):
    """
    Trains the classifier model.

    Parameters:
    model (nn.Module): The classifier model to train.
    train_dataset (TensorDataset): Training dataset.
    val_dataset (TensorDataset): Validation dataset.
    epochs (int): Number of epochs to train.
    learning_rate (float): Learning rate for optimizer.
    """

    # Create DataLoader instances for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, attrs, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}, Images shape: {inputs.shape}, Labels shape: {labels.shape}")
            optimizer.zero_grad()
            outputs = model(inputs, attrs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, attrs, labels in val_loader:
                outputs = model(inputs, attrs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {correct/total * 100}%')

    torch.save(model.state_dict(), 'best_classifier_model.pth')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LEGO Brick Classifier Training')
    parser.add_argument('--data_path', type=str, default='./data/brick_images', help='Path to directory containing collected brick images, renderings, or so')
    parser.add_argument('--detector_path', type=str, default='./models/yolov7.pth', help='Path to directory containing pretrained object detector')
    parser.add_argument('--classifier_path', type=str, default='./models/resnet50.pth', help='Path to directory containing pretrained object classifier')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of brick classes')
    parser.add_argument('--num_attr', type=int, default=2, help='Number of brick attributes to consider. Default is 2 [color, size]')

    args = parser.parse_args()

    # Initialize YOLOv7 model
    yolo = YOLOv7(model_path=args.detector_path)

    # Define transforms (example: resize and denoise)
    transform = DataPreprocessing.preprocess_image

    # Create dataset instance
    dataset = LegoBrickDataset(args.data_path, transform=transform)

    # Split dataset into train and validation sets based on data size
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

     # Create DataLoader instances for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Preprocess and detect bricks
    train_detected_bricks, train_attributes, train_labels = preprocess_and_detect(yolo, train_loader)
    val_detected_bricks, val_attributes, val_labels = preprocess_and_detect(yolo, val_loader)

    train_data = TensorDataset(train_detected_bricks, train_attributes, train_labels)
    val_data = TensorDataset(val_detected_bricks, val_attributes, val_labels)

    # Choose classifier model (Example: AttributeClassifier)
    model = AttributeClassifier(num_classes=args.num_classes, attr_dim=args.num_attr)
    
    # Train classifier model
    train_classifier(model, train_data, val_data, args.epochs, args.learning_rate)

    torch.save(model.state_dict(), args.classifier_path)
    print(f"Model saved to {args.classifier_path}")
