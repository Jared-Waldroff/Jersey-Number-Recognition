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

class CommonConstants(Enum):
    FEATURE_DATA_FILE_POSTFIX = "_features.npy"

class DataPreProcessing:
    def __init__(self, display_transformed_image_sample: bool=False, num_image_samples: int=1, silence_logs: bool=False):
        self.display_transformed_image_sample = display_transformed_image_sample
        self.num_image_samples = num_image_samples
        self.silence_logs = silence_logs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = (self.device.type == 'cuda')
        logging = CustomLogger().get_logger()

        self.num_images_processed = 0
        self.ver_to_specs = {}
        
        if not self.silence_logs:
            logging.info("DataPreProcessing initialized. Universe of available data paths:")
            
            for data_path in DataPaths:
                logging.info(f"{data_path.name}: {data_path.value}")
        
    def create_data_dirs(self):
        # For every entry in DataPaths, create a directory if it doesn't exist,
        # but skip entries that appear to be files (i.e. have a non-empty suffix).
        for data_path in DataPaths:
            path = Path(data_path.value)
            if path.suffix:  # if there's a file extension, skip creating
                continue
            if not path.exists():
                os.makedirs(path)
                logging.info(f"Created directory: {data_path.value}")
                
    def get_specs_from_version(self, model_version):
        conf, weights = self.ver_to_specs[model_version]
        conf, weights = str(conf), str(weights)
        return conf, weights
    
    def pass_through_reid_centroid(self, raw_image, output_file, model_version='res50_market'):
        """
        Process a raw image (or batch) through the pre-trained centroid model.
        This method:
          - Loads the appropriate model using a config and checkpoint.
          - Ensures input raw_image is 4D (N, C, H, W); if 3D, unsqueezes.
          - Feeds the image through model.backbone and batch-norm layer.
          - Flattens the global features per sample.
          - Appends the feature vector to the output file.
        
        Args:
            raw_image (torch.Tensor): Input tensor (C, H, W) or (N, C, H, W).
            output_file (str): File path to save the features.
            model_version (str): Version key to select model configuration.
        
        Returns:
            np.ndarray: Flattened feature vector(s) with shape (N, d)
        """
        # Update the ver_to_specs dictionary:
        self.ver_to_specs["res50_market"] = (DataPaths.REID_CONFIG_YAML.value, DataPaths.REID_MODEL_1.value)
        self.ver_to_specs["res50_duke"]   = (DataPaths.REID_CONFIG_YAML.value, DataPaths.REID_MODEL_2.value)
        
        CONFIG_FILE, MODEL_FILE = self.get_specs_from_version(model_version)
        cfg.merge_from_file(CONFIG_FILE)
        opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
        cfg.merge_from_list(opts)
        
        model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)
        if self.use_cuda:
            model.to('cuda')
            print("using GPU")
        model.eval()
        
        # Ensure input is 4D
        if raw_image.dim() == 3:
            raw_image = raw_image.unsqueeze(0)
        
        with torch.no_grad():
            input_tensor = raw_image.cuda() if self.use_cuda else raw_image
            _, global_feat = model.backbone(input_tensor)
            global_feat = model.bn(global_feat)
        
        # global_feat shape: (N, d). We keep the batch dimension.
        processed_image = global_feat.cpu().numpy()  # shape: (N, d)
        
        # Append new features to output_file:
        # NOTE: The only time we append is when the image tensor batch sent through ImageBatchPipeline is < count(images_in_tracklet).
        # i.e. this would be the case for just passing 2 images through the pipeline, from the same batch, and appending data for img 2 to img 1.
        if os.path.exists(output_file):
            existing = np.load(output_file, allow_pickle=True)
            combined = np.concatenate([existing, processed_image], axis=0)
            np.save(output_file, combined)
        else:
            np.save(output_file, processed_image)
        logging.info(f"Saved features for tracklet with shape {processed_image.shape}")
            
        return processed_image

    def image_transform_pipeline(self, raw_image, output_file, model_version='res50_market'):
        """
        Process a single raw image through the centroid model pipeline.
        
        Steps:
          1. Accept a raw image (tensor, file path, or PIL Image) and convert it to a normalized tensor.
          2. Pass the tensor through the centroid model for resizing, cropping, and keyframe identification.
          3. Save the processed features to output_file.
        
        Returns:
          np.ndarray: The flattened feature vector for the input image.
        """
        processed_image = self.pass_through_reid_centroid(raw_image, output_file, model_version)
        return processed_image

  
    def get_tracks(self, input_folder):
        # Ignore hidden files
        tracks = [t for t in os.listdir(input_folder) if not t.startswith('.')]
        # Sort tracks numerically using the first sequence of digits in the name
        tracks = sorted(tracks, key=lambda t: int(re.search(r'\d+', t).group()) if re.search(r'\d+', t) else -1)
        logging.info(tracks[0:10])

        # Extract numerical part for min and max calculations
        def extract_number(track):
            match = re.search(r'(\d+)', track)
            return int(match.group(1)) if match else -1

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
                processed_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # Apply transforms
                transformed = val_transforms(processed_image)  # returns a tensor

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