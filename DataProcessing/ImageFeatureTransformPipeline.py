from DataProcessing.Logger import CustomLogger
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from DataProcessing.DataPreProcessing import DataPaths, CommonConstants
from reid.CentroidsReidRepo.train_ctl_model import CTLModel
from reid.CentroidsReidRepo.config.defaults import _C as cfg
import threading
import configuration as config
import torch
import os
import numpy as np
import subprocess
import json
import cv2
import math

# Limit concurrent GPU calls (example).
# CRUCIAL to prevent too many parallel shipments to our GPU to prevent CUDA-out-of-memory issues
# This will become a bottleneck as we enter series code here, but necessary to avoid exploding GPUs.
GPU_SEMAPHORE = threading.Semaphore(value=1)

class ImageFeatureTransformPipeline:
    def __init__(self,
                 raw_image_batch,
                 current_tracklet_number,
                 output_tracklet_processed_data_path,
                 current_tracklet_images_input_dir,
                 current_tracklet_processed_data_dir,
                 common_processed_data_dir,
                 run_soccer_ball_filter: bool,
                 generate_features: bool,
                 run_filter: bool,
                 model_version='res50_market',
                 suppress_logging: bool=False,
                 use_cache: bool=True,
                 image_batch_size: int = 200):
        self.raw_image_batch = raw_image_batch
        self.output_tracklet_processed_data_path = output_tracklet_processed_data_path
        self.model_version = model_version
        self.suppress_logging = suppress_logging
        self.use_cache = use_cache
        self.current_tracklet_number = current_tracklet_number
        self.run_soccer_ball_filter = run_soccer_ball_filter
        self.generate_features = generate_features
        self.run_filter = run_filter
        self.parallelize = True
        self.image_batch_size = image_batch_size
        
        # AUTOMATIC BATCH SIZE DETERMINATION
        # Determine the number of batches
        num_images_to_process = len(raw_image_batch)

        # Ensure at least one batch, using ceil to cover the remainder images
        # print(f"DEBUG: raw_image_batch length: {num_images_to_process}")
        # print(f"DEBUG: image_batch_size: {self.image_batch_size}")
        self.batch_size = max(1, math.ceil(num_images_to_process / self.image_batch_size))
        
        self.current_tracklet_images_input_dir = current_tracklet_images_input_dir
        self.current_tracklet_processed_data_dir = current_tracklet_processed_data_dir
        self.common_processed_data_dir = common_processed_data_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = (self.device.type == 'cuda')
        
        self.image_enhancer = ImageEnhancement()
        self.ver_to_specs = {}
        self.logger = CustomLogger().get_logger()
        
    def get_specs_from_version(self, model_version):
        conf, weights = self.ver_to_specs[model_version]
        conf, weights = str(conf), str(weights)
        return conf, weights
        
    def pass_through_gaussian_outliers_filter(self):
        self.logger.info("Identifying and removing outliers by calling gaussian_outliers_streamlined.py on feature file")
        
        # DO NOT USE: Cache management now controlled in CentralPipeline.
        # if self.use_cache and os.path.exists(self.current_tracklet_processed_data_dir):
        #     self.logger.info(f"Skipping outlier removal for tracklet {self.current_tracklet_number} as cache exists.")
        #     return
        
        command = [
            "python",
            f"{DataPaths.STREAMLINED_PIPELINE.value}\\gaussian_outliers.py",
            "--current_tracklet", self.current_tracklet_number,
            "--current_tracklet_images_input_dir", self.current_tracklet_images_input_dir,
            "--current_tracklet_processed_data_dir", self.current_tracklet_processed_data_dir,
            "--common_processed_data_dir", self.common_processed_data_dir,
        ]
        if self.suppress_logging:
            command.append("--suppress_logging")
        if self.use_cache:
            command.append("--use_cache")

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.logger.info(result.stdout)  # Log standard output
            self.logger.error(result.stderr)  # Capture and log standard error
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running gaussian_outliers_streamlined.py: {e}")
            self.logger.error(f"STDOUT:\n{e.stdout}")  # Print captured standard output
            self.logger.error(f"STDERR:\n{e.stderr}")  # Print captured standard error
        
        self.logger.info("Done removing outliers")
    
    def pass_through_reid_centroid(self, with_cuda=True):
        """
        Process a raw image (or batch) through the pre-trained centroid model.
        This method:
          - Loads the appropriate model using a config and checkpoint.
          - Ensures input self.raw_image_batch is 4D (N, C, H, W); if 3D, unsqueezes.
          - Splits the image batch into mini-batches to limit GPU memory usage.
          - Feeds each mini-batch through model.backbone and batch-norm layer.
          - Concatenates the flattened features per sample.
          - Saves the final feature vector to the output file.
        
        Args:
            with_cuda (bool): Whether to use CUDA for model inference (always True in our case).
        
        Returns:
            np.ndarray: Flattened feature vector(s) with shape (N, d)
        """
        output_file = os.path.join(self.output_tracklet_processed_data_path,
                                   CommonConstants.FEATURE_DATA_FILE_NAME.value)
        
        # Update ver_to_specs
        self.ver_to_specs["res50_market"] = (DataPaths.REID_CONFIG_YAML.value, DataPaths.REID_MODEL_1.value)
        self.ver_to_specs["res50_duke"]   = (DataPaths.REID_CONFIG_YAML.value, DataPaths.REID_MODEL_2.value)
        
        CONFIG_FILE, MODEL_FILE = self.get_specs_from_version(self.model_version)
        cfg.merge_from_file(CONFIG_FILE)
        opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True,
                "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
        cfg.merge_from_list(opts)
        
        # Only use the semaphore if using CUDA.
        if with_cuda and self.use_cuda:
            with GPU_SEMAPHORE:
                model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)
                model.to('cuda')
        else:
            model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg,
                                                  map_location=torch.device('cpu'))
        
        model.eval()
        
        # Ensure raw_image_batch is 4D.
        if self.raw_image_batch.dim() == 3:
            self.raw_image_batch = self.raw_image_batch.unsqueeze(0)
        
        # Determine total number of images.
        num_images = self.raw_image_batch.size(0)
        feature_list = []
        
        # Process in mini-batches.
        with torch.no_grad():
            # Wrap entire inference loop in the GPU semaphore if using CUDA.
            if with_cuda and self.use_cuda:
                context = GPU_SEMAPHORE
            else:
                # Use a dummy context manager if not using CUDA. This is just to modularize the code a bit
                from contextlib import nullcontext
                context = nullcontext()
            
            with context:
                # For the current tracklet, subset the images into a mini-batch to avoid overloading our GPU and hitting out-of-memory problems.
                for i in range(0, num_images, self.batch_size):
                    batch = self.raw_image_batch[i:i+self.batch_size]
                    input_tensor = batch.cuda() if (with_cuda and self.use_cuda) else batch.cpu()
                    _, global_feat = model.backbone(input_tensor)
                    global_feat = model.bn(global_feat)
                    feature_list.append(global_feat.cpu()) # Bring back to CPU for concatenation.
        
        # Concatenate features from all mini-batches.
        global_features = torch.cat(feature_list, dim=0)
        processed_image = global_features.numpy()  # shape: (N, d)
        np.save(output_file, processed_image)
        self.logger.info(f"Saved features for tracklet with shape {processed_image.shape} to {output_file}")
        
        # Free GPU memory.
        del model
        torch.cuda.empty_cache()
        
        return processed_image
        
    def pass_through_soccer_ball_filter(self):
        self.logger.info("Determine soccer balls in image(s) using pre-trained model.")
        
        # DO NOT USE. Cache management now controlled at root of CentralPipeline.
        # if self.use_cache and os.path.exists(self.current_tracklet_images_input_dir):
        #     self.logger.info(f"Skipping soccer ball filter for tracklet {self.current_tracklet_number} as cache exists.")
        #     return
        
        HEIGHT_MIN = 35
        WIDTH_MIN = 30

        # check 10 random images for each track, mark as soccer ball if the size matches typical soccer ball size
        # NOTE: ball_list will always only ever contain 1 tracklet since this function runs once per tracklet
        ball_list = []

        # Perform the filtering for the current tracklet we are looping over
        # Skip if not a directory (extra safety check)
        if not os.path.isdir(self.current_tracklet_images_input_dir):
            self.logger.warning(f"Skipping tracklet {self.current_tracklet_number} as it is not a directory.")
            return

        # Filter out hidden files when listing images
        image_names = [img for img in os.listdir(self.current_tracklet_images_input_dir) if not img.startswith('.')]

        if not image_names:  # Skip if no images found
            self.logger.warning(f"Skipping tracklet {self.current_tracklet_number} as no images were found.")
            return

        sample = len(image_names) if len(image_names) < 10 else 10
        imgs = np.random.choice(image_names, size=sample, replace=False)
        width_list = []
        height_list = []
        try:
            for img_name in imgs:
                img_path = os.path.join(self.current_tracklet_images_input_dir, img_name)
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                width_list.append(w)
                height_list.append(h)
        except:
            self.logger.error(f"Error reading image {img_path}")
            return
        mean_w, mean_h = np.mean(width_list), np.mean(height_list)
        if mean_h <= HEIGHT_MIN and mean_w <= WIDTH_MIN:
            # this must be a soccer ball
            ball_list.append(self.current_tracklet_number)

        self.logger.info(f"Found {len(ball_list)} balls, Ball list: {ball_list}")
        
        # If the soccet_ball_list is not empty, proceed with writing the results to the file
        if len(ball_list) > 0:
            # Write the results to the soccer_ball_list
            # Sanity check: ensure ball list has only 1 element and output a warning if not (should never happen)
            if len(ball_list) > 1:
                self.logger.warning(f"Found more than one soccer ball in tracklet {self.current_tracklet_number}. This should not happen.")
            
            # Just write
            try:
                ball_file_path = os.path.join(self.current_tracklet_images_input_dir, config.dataset['SoccerNet']['soccer_ball_list'])
                with open(ball_file_path, 'w') as fp:
                    # Dump the json data directly
                    # We will only have the current track in here
                    json.dump({"ball_tracks": [self.current_tracklet_number]}, fp, indent=4)
            except FileNotFoundError:
                self.logger.warning(f"Soccer ball list file {ball_file_path} not found. Creating a new one.")
                with open(ball_file_path, 'w') as fp:
                    json.dump({"ball_tracks": [self.current_tracklet_number]}, fp, indent=4)

            return True
    
    def run_image_transform_pipeline(self):
        """
        Process a single raw image through the centroid model pipeline.
        
        Steps:
          1. Accept a raw image (tensor, file path, or PIL Image) and convert it to a normalized tensor.
          2. Pass the tensor through the centroid model for resizing, cropping, and keyframe identification.
          3. Save the processed features to self.output_tracklet_processed_data_path.
        
        Returns:
          np.ndarray: The flattened feature vector for the input image.
        """
        if self.run_soccer_ball_filter:
            self.pass_through_soccer_ball_filter()
            
        if self.generate_features:
            self.pass_through_reid_centroid()  
        
        if self.run_filter:
            self.pass_through_gaussian_outliers_filter()