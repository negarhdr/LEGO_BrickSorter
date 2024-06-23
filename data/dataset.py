'''
Customized Dataset class for Bricks dataset. The function __getitem__ can be modified accordingly.
'''

import os
import cv2
from torch.utils.data import Dataset

class LegoBrickDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(self.data_dir, img) for img in os.listdir(self.data_dir) if img.endswith('.jpg')]
        self.labels = self.load_labels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_labels(self):
        labels_file = os.path.join(self.data_dir, 'labels.txt')
        with open(labels_file, 'r') as f:
            lines = f.readlines()
            labels = [int(line.strip().split()[1]) for line in lines]  # Assuming format: image_name label (Brick_ID / Brick_Type)
        return labels

