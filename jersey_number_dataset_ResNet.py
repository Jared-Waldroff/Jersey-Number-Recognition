from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pandas as pd
import json
from PIL import Image
from torchvision import transforms

'''
############################################################################################################
This script defines the classes for crerating datasets for ResNet testing. This script includes classes: 
JerseyNumberDataset and JerseyNumberMultitaskDataset for loading the jersey number dataset The images are 
transformed with grayscale, autocontrast, color jittering, and sharpnessadjustment. 
############################################################################################################
'''

# Ensure data transformations are defined correctly
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomGrayscale(p=0.2),  # 80% chance of grayscale
        transforms.RandomAutocontrast(p=0.5),  # Automatically enhances contrast
        transforms.ColorJitter(brightness=(0.75, 1.0), contrast=0.5), # Color jittering
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # Slight sharpening 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Create dataset from Crops data
class JerseyNumberDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.img_dir = img_dir

        # Ensure mode is valid
        if mode not in data_transforms:
            raise ValueError(f"Invalid mode '{mode}', expected one of {list(data_transforms.keys())}")

        self.transform = data_transforms[mode]

        # Load mappings from tracklet ID to jersey number
        with open(annotations_file, 'r') as f:
            self.tracklet_to_label = json.load(f)  # { "0": 4, "1": 93, ... }

        self.data = []
        for img_name in os.listdir(img_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                tracklet_id = img_name.split('_')[0]  # Extract tracklet ID (first part of filename)
                
                if tracklet_id in self.tracklet_to_label:
                    jersey_number = self.tracklet_to_label[tracklet_id]
                    
                    if 0 <= jersey_number < 100:  # Ensure valid jersey number (ignore -1)
                        img_path = os.path.join(img_dir, img_name)
                        self.data.append((img_path, jersey_number))

        print(f"Loaded {len(self.data)} samples from {annotations_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

class JerseyNumberMultitaskDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.img_dir = img_dir

        # Ensure mode is valid
        if mode not in data_transforms:
            raise ValueError(f"Invalid mode '{mode}', expected one of {list(data_transforms.keys())}")

        self.transform = data_transforms[mode]

        # Load mappings from folder number to jersey number
        with open(annotations_file, 'r') as f:
            self.folder_to_label = json.load(f)  # Dictionary { "0": 10, "1": 23, ... }

        self.data = []
        for folder, jersey_number in self.folder_to_label.items():
            folder_path = os.path.join(img_dir, folder)  # Full path to the subfolder
            if not os.path.isdir(folder_path):
                continue  # Skip if folder does not exist

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure valid image files
                    if 0 < jersey_number < 100:  # Ensure valid jersey number
                        digit1, digit2 = self.get_digit_labels(jersey_number)
                        self.data.append((img_path, jersey_number, digit1, digit2))

        print(f"Loaded {len(self.data)} samples from {annotations_file}")

    def __len__(self):
        return len(self.data)

    def get_digit_labels(self, label):
        """Extract two-digit jersey numbers."""
        if label < 10:
            return label, 10  # Single-digit numbers get (X, 10) padding
        else:
            return label // 10, label % 10

    def __getitem__(self, idx):
        img_name, label, digit1, digit2 = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  

        return image, label, digit1, digit2