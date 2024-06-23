'''
Object detectors are defined to detect the bounding box of bricks in captured images. 
Default detector is YOLOv7. Other object detectors to be added in the future.
'''

import torch
import cv2
import numpy as np

MODEL_PATH = './yolov7.pt'

class YOLOv7:
    def __init__(self, model_path=MODEL_PATH, device='cuda'):
        self.device = device
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path, force_reload=True, trust_repo=True).to(device)

    def detect_objects(self, image, conf_threshold=0.5):
        self.model.eval()
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf
        return detections[detections[:, 4] >= conf_threshold]

# Example inference usage
if __name__ == "__main__":
    yolo = YOLOv7()
    image = cv2.imread('captured_images/test_image.jpg')
    results = yolo.detect_objects(image)
    print(results)
