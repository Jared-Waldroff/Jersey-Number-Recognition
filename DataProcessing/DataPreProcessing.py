from enum import Enum
from pathlib import Path
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
import logging
from DataProcessing.Logger import CustomLogger
from reid.CentroidsReidRepo.datasets.transforms.build import ReidTransforms
from reid.CentroidsReidRepo.train_ctl_model import CTLModel
from reid.CentroidsReidRepo.config.defaults import _C as cfg
#import configuration import as cfg

class ModelUniverse(Enum):
  REID_CENTROID = "REID"
  LEGIBILITY_CLASSIFIER = "LEGIBILITY"
  MOE = "MIXTURE_OF_EXPERTS"
  RAC = "RETRIEVAL_AUGMENTED_CLASSIFICATION"
  IMPROVED_STR = "CLIP4STR"
  DUMMY = "DUMMY"

class DataPaths(Enum):
    ROOT_DATA_DIR = str(Path.cwd().parent.parent / 'data' / 'SoccerNet' / 'jersey-2023' / 'extracted')
    TEST_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'test' / 'images')
    TRAIN_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'train' / 'images')
    CHALLENGE_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'challenge' / 'images')
    PRE_TRAINED_MODELS_DIR = str(Path.cwd().parent.parent / 'data' / 'pre_trained_models')
    REID_PRE_TRAINED = str(Path(PRE_TRAINED_MODELS_DIR) / 'reid')
    REID_MODEL_1 = str(Path(REID_PRE_TRAINED) / 'dukemtmcreid_resnet50_256_128_epoch_120.ckpt')
    REID_MODEL_2 = str(Path(REID_PRE_TRAINED) / 'market1501_resnet50_256_128_epoch_120.ckpt')
    REID_CONFIG_YAML = str(Path(REID_PRE_TRAINED) / 'configs' / '256_resnet50.yml')
    PROCESSED_DATA_OUTPUT_DIR = str(Path.cwd().parent.parent / 'data' / 'SoccerNet' / 'jersey-2023' / 'processed_data')
    PROCESSED_DATA_OUTPUT_DIR_TRAIN = str(Path(PROCESSED_DATA_OUTPUT_DIR) / 'train')
    PROCESSED_DATA_OUTPUT_DIR_TEST = str(Path(PROCESSED_DATA_OUTPUT_DIR) / 'test')
    PROCESSED_DATA_OUTPUT_DIR_CHALLENGE = str(Path(PROCESSED_DATA_OUTPUT_DIR) / 'challenge')

class DataPreProcessing:
    def __init__(self, silence_logs: bool=False):
        self.silence_logs = silence_logs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = (self.device.type == 'cuda')
        logging = CustomLogger().get_logger()
        
        self.ver_to_specs = {}
        
        if not self.silence_logs:
            logging.info("DataPreProcessing initialized. Universe of available data paths:")
            
            for data_path in DataPaths:
                logging.info(f"{data_path.name}: {data_path.value}")
        
    def create_data_dirs(self):
        # For every directory inside data_paths, create that directory if it does not already exist
        for data_path in DataPaths:
            if not os.path.exists(data_path.value):
                os.makedirs(data_path.value)
                logging.info(f"Created directory: {data_path.value}")
                
    def get_specs_from_version(self, model_version):
        conf, weights = self.ver_to_specs[model_version]
        conf, weights = str(conf), str(weights)
        return conf, weights
    
    def pass_through_reid_centroid(self, raw_image, model_version='res50_market'):
        self.ver_to_specs["res50_market"] = (DataPaths.REID_CONFIG_YAML.value, DataPaths.REID_MODEL_1.value)
        self.ver_to_specs["res50_duke"]   = (DataPaths.REID_CONFIG_YAML.value, DataPaths.REID_MODEL_2.value)
        
        CONFIG_FILE, MODEL_FILE = self.get_specs_from_version(model_version)
        cfg.merge_from_file(CONFIG_FILE)
        opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
        cfg.merge_from_list(opts)
        
        #model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)
        
        processed_image = cfg
        
    def single_image_transform_pipeline(self, raw_image, model_version='res50_market'):
        # Step 2: Pass through the centroid model that:
        #         1. Resizes + crops the image
        #         2. Does keyframe identification by applying a light ViT to hone in on the player's back
        # Step 3: Call the enhance_image function from DataAugmentation to further enhance this image
        # All of these steps come from main.py. Add them from there.
        
        # dict used to get model config and weights using model version
        
        # Step 1 tranform the image using the reid centroid model
        processed_image = self.pass_through_reid_centroid(raw_image, model_version)
        
        print(processed_image)

  
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
            
        return tracks, max_track
  
    def process_single_track(self, track, input_folder, val_transforms):
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
                self.logging.info(f"Error processing {img_full_path}: {e}")
                continue

        if track_features:
            processed = torch.cat(track_features, dim=0)
            return (track, processed)
        return None
      
    def generate_features(self, input_folder, output_folder, num_tracks, tracks: bool=None, classic_transform: bool=False):
        """
        Generate preprocessed tensors (features) for each image in the specified tracklets.
        
        Args:
            input_folder (str): Path to the folder containing tracklet subdirectories.
            output_folder (str): Path to the folder where processed results (if any) are stored.
            num_tracks (int): Maximum number of tracklets to process.
            tracks (list, optional): A list of tracklet names (subfolders) to process.
                                    If None, tracks are obtained via self.get_tracks().
        
        Returns:
            dict: A dictionary mapping tracklet name -> torch.Tensor of shape (N, C, H, W),
                where N is the number of images in that tracklet.
        
        Notes:
            - If self.device is 'cuda', this function processes images in a single thread
            to avoid multiple processes contending for the GPU.
            - If self.device is 'cpu', it uses ProcessPoolExecutor for CPU parallelism.
        """
        if classic_transform:
            # Define validation transforms using torchvision
            val_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            transforms_base = ReidTransforms(cfg)
            val_transforms = transforms_base.build_transforms(is_train=False)

        # If no explicit track list is provided, gather from disk
        if tracks is None:
            if not self.silence_logs:
                logging.info("No tracklets provided to generate_features. Getting all tracklets.")
            tracks, max_track = self.get_tracks(input_folder)

        # Limit to num_tracks
        tracks = tracks[:num_tracks]

        processed_data = {}

        if self.use_cuda:
            # Single-process approach on GPU
            if not self.silence_logs:
                logging.info("Using single-process GPU mode to generate features.")
            for track in tqdm(tracks, desc="Loading tracklets (GPU)"):
                result = self.process_single_track(track, input_folder, val_transforms)
                if result is not None:
                    track_name, features = result
                    processed_data[track_name] = features
        else:
            # Multi-process CPU approach
            if not self.silence_logs:
                logging.info("Using CPU parallel mode (ProcessPoolExecutor).")
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self.process_single_track, track, input_folder, val_transforms): track
                    for track in tracks
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading tracklets (CPU)"):
                    result = future.result()
                    if result is not None:
                        track_name, features = result
                        processed_data[track_name] = features

        return processed_data