from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pandas as pd
import json
from PIL import Image
from torchvision import transforms

data_transforms = {
    'train': {
        'resnet':
            transforms.Compose([
            transforms.RandomGrayscale(),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Image Net
            #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
            ]),
        'vit':
            transforms.Compose([
                transforms.RandomGrayscale(),
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Image Net
                # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
            ]),
        },

    'val': {
        'resnet':
            transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
           #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
        ]),
        'vit':
            transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
           #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
        ])
    },
    'test': {
        'resnet':
        transforms.Compose([ # same as val
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
        #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
    ]),
        'vit':
        transforms.Compose([ # same as val
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
        #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
    ]),
    }
}

class JerseyNumberDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.transform = data_transforms[mode]
        self.img_labels = pd.read_csv(annotations_file)
        unqiue_ids = np.unique(self.img_labels.iloc[:, 1].to_numpy())
        print(f"Datafile:{annotations_file}, number of labels:{len(self.img_labels)}, unique ids: {len(unqiue_ids)}")
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
'''
class JerseyNumberMultitaskDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.transform = data_transforms[mode]
        self.img_labels = pd.read_csv(annotations_file)
        unqiue_ids = np.unique(self.img_labels.iloc[:, 1].to_numpy())
        print(f"Datafile:{annotations_file}, number of labels:{len(self.img_labels)}, unique ids: {len(unqiue_ids)}")
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def get_digit_labels(self, label):
        if label < 10:
            return label, 10
        else:
            return label // 10, label % 10

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        digit1, digit2 = self.get_digit_labels(label)
        if not (label> 0 and label < 100 and digit1 < 10 and digit1 > 0 and digit2 > -1 and digit2 < 11):
            print(label, digit1, digit2)
        if self.transform:
            image = self.transform(image)
        return image, label, digit1, digit2
'''
# Ensure data transformations are defined correctly
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

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
            image = self.transform(image)  # ✅ FIX: Now correctly applies transformations

        return image, label, digit1, digit2

class UnlabelledJerseyNumberLegibilityDataset(Dataset):
    def __init__(self, image_paths, mode='test', arch='resnet18'):
        if 'resnet' in arch:
            arch = 'resnet'
        self.transform = data_transforms[mode][arch]
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

class TrackletLegibilityDataset(Dataset):
    def __init__(self, annotations_file, parent_dir, mode='test', arch='resnet18'):
        if 'resnet' in arch:
            arch = 'resnet'
        self.transform = data_transforms[mode][arch]
        with open(annotations_file, 'r') as f:
            self.tracklet_labels = json.load(f)
        tracklets = self.tracklet_labels.keys()
        self.image_paths = []
        for track in tracklets:
            tracklet_dir = os.path.join(parent_dir, track)
            images = os.listdir(tracklet_dir)
            for im in images:
                label = int(self.tracklet_labels[track])
                label = 1 if label > 0 else 0
                self.image_paths.append([os.path.join(tracklet_dir, im), track, label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, track, label = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, track, label


class JerseyNumberLegibilityDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train', isBalanced=False, arch='resnet18'):
        if 'resnet' in arch:
            arch = 'resnet'
        self.transform = data_transforms[mode][arch]
        self.img_labels = pd.read_csv(annotations_file)
        if isBalanced:
            legible =self.img_labels[self.img_labels.iloc[:,1]==1]
            count_legible = len(legible)
            illegible = self.img_labels[self.img_labels.iloc[:,1]==0]
            print(count_legible, len(illegible))
            if len(illegible) > count_legible:
                illegible = illegible.sample(n=count_legible)
            self.img_labels = pd.concat([legible, illegible])
            print(f"Balanced dataset: legibles = {count_legible} all = {len(self.img_labels)}")
        else:
            legible = self.img_labels[self.img_labels.iloc[:, 1] == 1]
            count_legible = len(legible)
            print(f"As-is dataset: legibles = {count_legible} all = {len(self.img_labels)}")

        self.img_dir = img_dir


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label, self.img_labels.iloc[idx, 0]

