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
        print("=" * 50)
        print("Dataset Initialization Diagnostics")
        print("=" * 50)

        # Normalize architecture name
        if 'resnet' in arch:
            arch = 'resnet'

        # Transform selection
        self.transform = data_transforms[mode][arch]

        # Read JSON file
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)

            print(f"Loaded JSON with {len(annotations)} entries")

            # Convert JSON to DataFrame
            data = []
            for tracklet, jersey_number in annotations.items():
                # Convert illegible (-1) to 0, legible to 1
                label = 1 if jersey_number != -1 else 0

                # Find images for this tracklet
                tracklet_dir = os.path.join(img_dir, tracklet)
                if os.path.exists(tracklet_dir):
                    for img_name in os.listdir(tracklet_dir):
                        img_path = os.path.join(tracklet, img_name)
                        data.append([img_path, label])

            self.img_labels = pd.DataFrame(data, columns=['image', 'legible'])

            print("\nDataFrame Information:")
            print(f"Total rows: {len(self.img_labels)}")
            print("\nLabel Distribution:")
            print(self.img_labels['legible'].value_counts())

        except Exception as e:
            print(f"Error loading annotations: {e}")
            raise

        # Balancing logic
        if isBalanced:
            print("\nBalancing Dataset:")
            legible = self.img_labels[self.img_labels['legible'] == 1]
            illegible = self.img_labels[self.img_labels['legible'] == 0]

            print(f"Original Legible images: {len(legible)}")
            print(f"Original Illegible images: {len(illegible)}")

            # Balance by downsampling the majority class
            if len(illegible) > len(legible):
                illegible = illegible.sample(n=len(legible))
            elif len(legible) > len(illegible):
                legible = legible.sample(n=len(illegible))

            self.img_labels = pd.concat([legible, illegible])
            print(f"Balanced dataset: Total = {len(self.img_labels)}")
            print(f"Balanced dataset: Legible = {len(legible)}, Illegible = {len(illegible)}")

        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

            # Verbose image loading diagnostics
            print(f"Loading image: {img_path}")
            print(f"Image exists: {os.path.exists(img_path)}")

            # Try to open the image
            image = Image.open(img_path).convert('RGB')

            # Log image details
            print(f"Image mode: {image.mode}")
            print(f"Image size: {image.size}")

            label = self.img_labels.iloc[idx, 1]

            if self.transform:
                image = self.transform(image)

            return image, label, self.img_labels.iloc[idx, 0]

        except FileNotFoundError:
            print(f"ERROR: Image file not found: {img_path}")
            raise
        except IOError as e:
            print(f"ERROR: Unable to open image {img_path}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error loading image: {e}")
            raise
