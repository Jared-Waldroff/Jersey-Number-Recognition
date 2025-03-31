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
from reid.CentroidsReidRepo.config.defaults import _C as cfg
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from DataProcessing.DataAugmentation import ImageEnhancement

class ModelUniverse(Enum):
  REID_CENTROID = "REID"
  LEGIBILITY_CLASSIFIER = "LEGIBILITY"
  MOE = "MIXTURE_OF_EXPERTS"
  RAC = "RETRIEVAL_AUGMENTED_CLASSIFICATION"
  IMPROVED_STR = "CLIP4STR"
  DUMMY = "DUMMY"

class DataPaths(Enum):
    ROOT_DATA_DIR = str(Path.cwd().parent.parent / 'data' / 'SoccerNet' / 'jersey-2023' / 'extracted')
    TEST_DATA_GT = str(Path(ROOT_DATA_DIR) / 'test' / 'test_gt.json')
    TRAIN_DATA_GT = str(Path(ROOT_DATA_DIR) / 'train' / 'train_gt.json')
    TEST_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'test' / 'images')
    TRAIN_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'train' / 'images')
    CHALLENGE_DATA_DIR = str(Path(ROOT_DATA_DIR) / 'challenge' / 'images')
    PRE_TRAINED_MODELS_DIR = str(Path.cwd().parent.parent / 'data' / 'pre_trained_models')
    REID_PRE_TRAINED = str(Path(PRE_TRAINED_MODELS_DIR) / 'reid')
    STR_PRE_TRAINED = str(Path(PRE_TRAINED_MODELS_DIR) / 'str')
    STR_MODEL = str(Path(STR_PRE_TRAINED) / 'parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt')
    REID_MODEL_1 = str(Path(REID_PRE_TRAINED) / 'dukemtmcreid_resnet50_256_128_epoch_120.ckpt')
    REID_MODEL_2 = str(Path(REID_PRE_TRAINED) / 'market1501_resnet50_256_128_epoch_120.ckpt')
    REID_CONFIG_YAML = str(Path(REID_PRE_TRAINED) / 'configs' / '256_resnet50.yml')
    RESNET_MODEL = str(Path(PRE_TRAINED_MODELS_DIR) / 'resnet' / 'legibility_resnet34_soccer_20240215.pth')
    VIT_MODEL = str(Path(PRE_TRAINED_MODELS_DIR) / 'ViT' / 'vit_base_patch16_224_in21k_ft_svhn.pth')
    PROCESSED_DATA_OUTPUT_DIR = str(Path.cwd().parent.parent / 'data' / 'SoccerNet' / 'jersey-2023' / 'processed_data')
    PROCESSED_DATA_OUTPUT_DIR_TRAIN = str(Path(PROCESSED_DATA_OUTPUT_DIR) / 'train')
    PROCESSED_DATA_OUTPUT_DIR_TEST = str(Path(PROCESSED_DATA_OUTPUT_DIR) / 'test')
    PROCESSED_DATA_OUTPUT_DIR_CHALLENGE = str(Path(PROCESSED_DATA_OUTPUT_DIR) / 'challenge')
    COMMON_PROCESSED_OUTPUT_DATA_TRAIN = str(Path(PROCESSED_DATA_OUTPUT_DIR_TRAIN) / 'common_data')
    COMMON_PROCESSED_OUTPUT_DATA_TEST = str(Path(PROCESSED_DATA_OUTPUT_DIR_TEST) / 'common_data')
    COMMON_PROCESSED_OUTPUT_DATA_CHALLENGE = str(Path(PROCESSED_DATA_OUTPUT_DIR_CHALLENGE) / 'common_data')
    STREAMLINED_PIPELINE = str(Path.cwd().parent.parent / 'StreamlinedPipelineScripts')
    ENHANCED_STR_ROOT = str(Path(PRE_TRAINED_MODELS_DIR) / 'clip4str')
    ENHANCED_STR_MAIN = str(Path(ENHANCED_STR_ROOT) / 'clip4str_huge_3e942729b1.pt')
    ENHANCED_STR_OPEN_CLIP = str(Path(ENHANCED_STR_ROOT) / 'appleDFN5B-CLIP-ViT-H-14.bin')
    ENHANCED_STR_VIT_L = str(Path(ENHANCED_STR_ROOT) / 'ViT-L-14.pt')

class CommonConstants(Enum):
    FEATURE_DATA_FILE_NAME = "features.npy"

# To avoid issues with serialization
def _worker_fn(args):
    instance, track, input_folder, val_transforms = args
    result = instance.process_single_track(track, input_folder, val_transforms)
    if result is None:
        return None
    track_name, features = result
    # Detach from the graph, move to CPU, and clone to avoid shared memory mapping issues.
    if features is not None:
        features = features.detach().cpu().clone()
    return track_name, features

class DataPreProcessing:
    def __init__(self, display_transformed_image_sample: bool=False, suppress_logging: bool=False):
        self.suppress_logging = suppress_logging
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = (self.device.type == 'cuda')
        logging = CustomLogger().get_logger()

        self.num_images_processed = 0
        
        self.image_enhancement = ImageEnhancement()
        
        if not self.suppress_logging:
            logging.info("DataPreProcessing initialized. Universe of available data paths:")
            
            for data_path in DataPaths:
                logging.info(f"{data_path.name}: {data_path.value}")
        
    def create_data_dirs(self, input_data_path, output_processed_data_path):
        # For every entry in DataPaths, create a directory if it doesn't exist,
        # but skip entries that appear to be files (i.e. have a non-empty suffix).
        for data_path in DataPaths:
            path = Path(data_path.value)
            if path.suffix:  # if there's a file extension, skip creating
                continue
            if not path.exists():
                os.makedirs(path)
                logging.info(f"Created directory: {data_path.value}")
        
        # Step 2: Create the processed tracklet data dirs
        # Call get_tracks from input dir and create a directory for every tracklet inside the output_processed_data_path
        tracks, _ = self.get_tracks(input_data_path)
        for track in tracks:
            if not os.path.exists(os.path.join(output_processed_data_path, track)):
                os.makedirs(os.path.join(output_processed_data_path, track))

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
                img = cv2.imread(img_full_path)
                if img is None:
                    continue

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transforms.ToTensor()(img_rgb)  # (C, H, W), values in [0, 1]

                # Normalize using ImageNet stats
                img_tensor = transforms.Normalize(
                    mean=self.image_enhancement.mean.squeeze(),
                    std=self.image_enhancement.std.squeeze()
                )(img_tensor)

                # Enhance using the custom image enhancement module
                denorm_img = self.image_enhancement.denormalize(img_tensor).clamp(0, 1)
                img_tensor = self.image_enhancement.enhance_image(img_tensor)

                # If val_transforms expects PIL input, convert back from tensor
                img_pil = transforms.ToPILImage()(denorm_img)
                transformed = val_transforms(img_pil)

                track_features.append(transformed.unsqueeze(0))
            except Exception as e:
                logging.info(f"Error processing {img_full_path}: {e}")
                continue

        if track_features:
            processed = torch.cat(track_features, dim=0)
            return (track, processed)
        return None
      
    def generate_features(
        self,
        input_folder,
        output_folder,
        num_tracks,
        tracks: bool=None,
        classic_transform: bool=False,
        cuda_only: bool=False
    ):
        """
        Generate preprocessed tensors (features) for each image in the specified tracklets.

        Args:
            input_folder (str): Path to the folder containing tracklet subdirectories.
            output_folder (str): Path to the folder where processed results (if any) are stored.
            num_tracks (int): Maximum number of tracklets to process.
            tracks (list, optional): A list of tracklet names (subfolders) to process.
                                    If None, tracks are obtained via self.get_tracks().
            classic_transform (bool): Whether to use the "classic" transforms pipeline.
            cuda_only (bool): If True, use a single process with CUDA – no CPU multiprocessing.

        Returns:
            dict: A dictionary mapping tracklet name -> torch.Tensor of shape (N, C, H, W),
                where N is the number of images in that tracklet.
        
        Notes:
            - If `self.use_cuda` and `cuda_only` is True, we process all tracklets on the GPU
            in a single process (no CPU multiprocessing).
            - If `self.use_cuda` and `cuda_only` is False, we use both CPU multiprocessing
            and CUDA batch processing.
            - Otherwise, we use CPU-only parallel processing with ProcessPoolExecutor.
        """

        # Set up the transformation pipeline
        if classic_transform:
            val_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transforms_base = ReidTransforms(cfg)
            val_transforms = transforms_base.build_transforms(is_train=False)

        # Get tracklet names if not provided
        if tracks is None:
            if not self.suppress_logging:
                logging.info("No tracklets provided. Retrieving from input folder.")
            tracks, max_track = self.get_tracks(input_folder)

        tracks = tracks[:num_tracks]  # Limit to num_tracks
        processed_data = {}

        # -----------------------------------------------
        # 1) GPU-only mode (single-process, no CPU pool)
        # -----------------------------------------------
        if self.use_cuda and cuda_only:
            if not self.suppress_logging:
                logging.info("Using single-process GPU mode (no CPU multiprocessing).")

            # Process each tracklet sequentially, but all tensor operations go on GPU
            for track in tqdm(tracks, desc="Processing tracklets (CUDA-Only)"):
                result = self.process_single_track(track, input_folder, val_transforms)
                if result is not None:
                    track_name, features = result
                    # Move features to GPU (if desired) or keep on CPU
                    processed_data[track_name] = features.cuda()

            return processed_data

        # ------------------------------------------------------------------
        # 2) Double parallelization: CPU multiprocessing + CUDA (existing)
        # ------------------------------------------------------------------
        if self.use_cuda:
            if not self.suppress_logging:
                logging.info("Using double parallelization: multiprocessing + CUDA batch processing.")

            # Set up arguments for each worker as a tuple
            worker_args = [(self, track, input_folder, val_transforms) for track in tracks]

            # Use multiprocessing for parallel track processing
            mp.set_start_method('spawn', force=True)  # Ensure safe CUDA multiprocessing
            with mp.Pool(processes=6) as pool:
                results = list(tqdm(pool.imap(_worker_fn, worker_args), total=len(worker_args), desc="Processing tracklets (CUDA + CPU)"))
            
            # Aggregate results
            for result in results:
                if result is not None:
                    track_name, features = result
                    processed_data[track_name] = features

        # --------------------------------------
        # 3) CPU-only mode (ProcessPoolExecutor)
        # --------------------------------------
        else:
            if not self.suppress_logging:
                logging.info("Using CPU parallel mode (ProcessPoolExecutor).")

            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self.process_single_track, track, input_folder, val_transforms): track
                    for track in tracks
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tracklets (CPU)"):
                    result = future.result()
                    if result is not None:
                        track_name, features = result
                        processed_data[track_name] = features

        return processed_data