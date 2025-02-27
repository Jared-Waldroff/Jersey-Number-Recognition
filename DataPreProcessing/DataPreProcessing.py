from enum import Enum
from pathlib import Path
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os
import re
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from Logger import CustomLogger

class DataPaths(Enum):
    ROOT_DATA_DIR = str(Path.cwd().parent / 'data' / 'SoccerNet' / 'jersey-2023' / 'extracted')
    TEST_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'test' / 'images')
    TRAIN_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'train' / 'images')
    VALIDATION_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'challenge' / 'images')
    TEMP_EXPERIMENT_DIR = str(Path.cwd() / 'experiments' / 'temp')

class DataPreProcessing:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging = CustomLogger().get_logger()
        logging.info("DataPreProcessing initialized.")
        logging.info(f"ROOT_DATA_DIR: {DataPaths.ROOT_DATA_DIR.value}")
        logging.info(f"TRAIN_DATA_DIR: {DataPaths.TRAIN_DATA_DIR.value}")
        logging.info(f"TEST_DATA_DIR: {DataPaths.TEST_DATA_DIR.value}")
        logging.info(f"VAL_DATA_DIR: {DataPaths.VALIDATION_DATA_DIR.value}")
        logging.info(f"Using device: {device}")
  
    def get_tracks(self, input_folder):
        # Ignore the .DS_Store files
        tracks = [t for t in os.listdir(input_folder) if not t.startswith('.')]
        logging.info(tracks[0:10])

        # Extract numerical part and convert to integer for comparison
        def extract_number(track):
            match = re.search(r'(\d+)', track)  # Extracts the first sequence of digits
            if match:
                return int(match.group(1))
            return -1  # Provide a default value if no number is found

        # Find min and max tracklets based on the extracted number
        if tracks:
            min_track = min(tracks, key=extract_number)
            max_track = max(tracks, key=extract_number)

            logging.info(f"Min tracklet: {min_track}")
            logging.info(f"Max tracklet: {max_track}")
        else:
            logging.warning("No tracklets found.")
            
        return tracks
  
    def process_single_track(self, track, input_folder, val_transforms, use_cuda=False):
        """
        Process one tracklet (i.e. one directory of images) and return a tuple (track, processed_data)
        where processed_data is either a tensor (if load_only) or a numpy array of features.
        """
        track_path = os.path.normpath(os.path.join(input_folder, track))
        if not os.path.isdir(track_path):
            return None  # Skip non-directory

        images = [img for img in os.listdir(track_path) if not img.startswith('.')]
        track_features = []
        for img_path in images:
            img_full_path = os.path.normpath(os.path.join(track_path, img_path))
            try:
                # Load image using cv2 and convert to PIL format
                img = cv2.imread(img_full_path)
                if img is None:
                    continue
                input_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # Apply transforms
                transformed = val_transforms(input_img)  # returns a tensor

                # Simply store the tensor (add a batch dimension for later concatenation)
                track_features.append(transformed.unsqueeze(0))
            except Exception as e:
                logging.info(f"Error processing {img_full_path}: {e}")
                continue

        if track_features:
            processed = torch.cat(track_features, dim=0)
            return (track, processed)
        return None
      
    def generate_features(self, input_folder, output_folder, num_tracks=1400):
        """
        
        """
        use_cuda = False
        model = None

        # Define validation transforms using torchvision
        val_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Get list of valid track directories (skip hidden files)
        tracks = self.get_tracks(DataPaths.TRAIN_DATA_DIR.value)[0:num_tracks]
        
        processed_data = {}

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.process_single_track, track, input_folder, val_transforms): track for track in tracks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tracks"):
                result = future.result()
                if result is not None:
                    track, features = result
                    processed_data[track] = features

        return processed_data