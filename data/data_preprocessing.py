'''
Data preprocessing steps to denoise, normalize and resize captured images. 
'''

import cv2
import numpy as np
from torchvision.transforms import ToTensor


class DataPreprocessing:
    @staticmethod
    def denoise_image(image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    @staticmethod
    def normalize_image(image):
        image = image.astype('float32') / 255.0
        return image

    @staticmethod
    def resize_image(image, size=(64, 64)):
        return cv2.resize(image, size)

    @staticmethod
    def to_tensor(image):
        return ToTensor()(image)
    
    @staticmethod
    def preprocess_image(image, size=(64, 64)):
        image = DataPreprocessing.denoise_image(image)
        image = DataPreprocessing.resize_image(image, size)
        image = DataPreprocessing.normalize_image(image)
        image = DataPreprocessing.to_tensor(image)
        return image


    