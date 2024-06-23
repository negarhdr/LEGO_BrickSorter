# LEGO Bricks Detection and Recognition

This project aims to detect and recognize LEGO bricks from images using a YOLOv7 object detector and a classifier model. The system is designed to process images of LEGO bricks, identify each brick, and classify it based on specific attributes such as color, and size.

Note: This is a mockup code to represent the overall platform architecture and components. 


## Table of Contents
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Installing YOLOv7](#installing-yolov7)
- [Training the Classifier](#training-the-classifier)
- [Inference](#inference)
- [Unit Tests](#unit-tests)

## Installation

### Requirements

Ensure you have Python 3.7 or higher installed. You can install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Installing YOLOv7
To install YOLOv7, follow these steps:
1. Clone the YOLOv7 repository:
```bash
git clone https://github.com/WongKinYiu/yolov7.git
```
2. Install the dependencies for YOLOv7:

```bash
cd yolov7
pip install -r requirements.txt
```

3. Follow the instructions in the YOLOv7 repository to set up and prepare the model weights. 

## Training the Classifier

To train the classifier on a dataset of LEGO brick images with annotations, use the train.py script. Ensure that your dataset is organized and annotated correctly. You need to specify the data path and other training parameters in the script or pass them as arguments.

```bash
python train.py --data_path /path/to/your/dataset --detector_path /path/to/pretrained/detector
```

The training script will preprocess the images, detect LEGO bricks using YOLOv7, extract attributes, and trains the classifier.

## Inference

To perform inference on new images, use the inference.py script. This script will load the trained classifier model and perform detection and classification on the provided images.

```bash
python inference.py --image_path /path/to/your/test/images --classifier_path /path/to/pretrained/classifier --detector_path /path/to/pretrained/detector
```
The script will display the results and save the images with bounding boxes and predicted classes.

## Unit Tests

To ensure that each module is functioning correctly, run the unit tests using the unittest framework. The tests cover various parts of the system, including the YOLOv7 detector, data preprocessing, classifier model, training process, inference, and optimization.

To run all the unit tests, use the following command:

```bash
python -m unittest discover
```

This will discover and run all the test cases in the project. However, Continious Integration (CI) is setup through Github Actions, so that with every modification to the code, it tests all the modules. 

