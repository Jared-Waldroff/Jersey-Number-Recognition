from DataProcessing.Logger import CustomLogger
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from DataProcessing.DataPreProcessing import DataPaths, CommonConstants
from reid.CentroidsReidRepo.train_ctl_model import CTLModel
from reid.CentroidsReidRepo.config.defaults import _C as cfg
import torch
import os
import numpy as np
import subprocess
import json
import cv2

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
                 use_cache: bool=True):
        self.raw_image_batch = raw_image_batch
        self.output_tracklet_processed_data_path = output_tracklet_processed_data_path
        self.model_version = model_version
        self.suppress_logging = suppress_logging
        self.use_cache = use_cache
        self.current_tracklet_number = current_tracklet_number
        self.run_soccer_ball_filter = run_soccer_ball_filter
        self.generate_features = generate_features
        self.run_filter = run_filter
        
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
            self.logger.info(result.stdout)  # Print logs from gaussian_outliers_streamlined.py
            self.logger.error(result.stderr)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running gaussian_outliers_streamlined.py: {e}")
            # Log the stdout and stderr from the exception (if available)
            self.logger.info(e.stdout)
            self.logger.error(e.stderr)
        
        self.logger.info("Done removing outliers")
    
    def pass_through_reid_centroid(self):
        """
        Process a raw image (or batch) through the pre-trained centroid model.
        This method:
          - Loads the appropriate model using a config and checkpoint.
          - Ensures input self.raw_image_batch is 4D (N, C, H, W); if 3D, unsqueezes.
          - Feeds the image through model.backbone and batch-norm layer.
          - Flattens the global features per sample.
          - Appends the feature vector to the output file.
        
        Args:
            self.raw_image_batch (torch.Tensor): Input tensor (C, H, W) or (N, C, H, W).
            self.output_tracklet_processed_data_path (str): File path to save the features.
            self.model_version (str): Version key to select model configuration.
        
        Returns:
            np.ndarray: Flattened feature vector(s) with shape (N, d)
        """
        # Update the ver_to_specs dictionary:
        self.ver_to_specs["res50_market"] = (DataPaths.REID_CONFIG_YAML.value, DataPaths.REID_MODEL_1.value)
        self.ver_to_specs["res50_duke"]   = (DataPaths.REID_CONFIG_YAML.value, DataPaths.REID_MODEL_2.value)
        
        CONFIG_FILE, MODEL_FILE = self.get_specs_from_version(self.model_version)
        cfg.merge_from_file(CONFIG_FILE)
        opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
        cfg.merge_from_list(opts)
        
        model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)
        if self.use_cuda:
            model.to('cuda')
            print("using GPU")
        model.eval()
        
        # Ensure input is 4D
        if self.raw_image_batch.dim() == 3:
            self.raw_image_batch = self.raw_image_batch.unsqueeze(0)
        
        with torch.no_grad():
            input_tensor = self.raw_image_batch.cuda() if self.use_cuda else self.raw_image_batch
            _, global_feat = model.backbone(input_tensor)
            global_feat = model.bn(global_feat)
        
        # global_feat shape: (N, d). We keep the batch dimension.
        processed_image = global_feat.cpu().numpy()  # shape: (N, d)
        
        # Append new features to self.output_tracklet_processed_data_path:
        # NOTE: The only time we append is when the image tensor batch sent through ImageBatchPipeline is < count(images_in_tracklet).
        # i.e. this would be the case for just passing 2 images through the pipeline, from the same batch, and appending data for img 2 to img 1.
        output_file = os.path.join(self.output_tracklet_processed_data_path, CommonConstants.FEATURE_DATA_FILE_NAME.value)
        if os.path.exists(output_file):
            existing = np.load(self.output_tracklet_processed_data_path, allow_pickle=True)
            combined = np.concatenate([existing, processed_image], axis=0)
            
            np.save(output_file, combined)
        else:
            np.save(output_file, processed_image)
        self.logger.info(f"Saved features for tracklet with shape {processed_image.shape} to {output_file}")
            
        return processed_image        
        
    def pass_through_soccer_ball_filter(self):
        self.logger.info("Determine soccer balls in image(s) using pre-trained model.")
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
        for img_name in imgs:
            img_path = os.path.join(self.current_tracklet_images_input_dir, img_name)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            width_list.append(w)
            height_list.append(h)
        mean_w, mean_h = np.mean(width_list), np.mean(height_list)
        if mean_h <= HEIGHT_MIN and mean_w <= WIDTH_MIN:
            # this must be a soccer ball
            ball_list.append(track)

        self.logger.info(f"Found {len(ball_list)} balls, Ball list: {ball_list}")
        
        # If the soccet_ball_list is not empty, proceed with writing the results to the file
        if len(ball_list) > 0:
            # Write the results to the soccer_ball_list
            # Sanity check: ensure ball list has only 1 element and output a warning if not (should never happen)
            if len(ball_list) > 1:
                self.logger.warning(f"Found more than one soccer ball in tracklet {self.current_tracklet_number}. This should not happen.")
            
            # Open JSON file in read+write mode
            try:
                with open(self.current_tracklet_images_input_dir, 'r+') as fp:
                    try:
                        ball_json = json.load(fp)
                    except json.JSONDecodeError:
                        self.logger.warning(f"JSON file {self.current_tracklet_images_input_dir} is empty or corrupt. Initializing new JSON structure.")
                        ball_json = {"ball_tracks": []}

                    if "ball_tracks" not in ball_json:
                        ball_json["ball_tracks"] = []

                    # Append the tracklet number if it is not already in the list
                    if self.current_tracklet_number not in ball_json["ball_tracks"]:
                        ball_json["ball_tracks"].append(self.current_tracklet_number)

                        # Move cursor to the start of the file before writing
                        fp.seek(0)
                        json.dump(ball_json, fp, indent=4)
                        fp.truncate()  # Ensure no leftover content

            except FileNotFoundError:
                self.logger.warning(f"Soccer ball list file {self.current_tracklet_images_input_dir} not found. Creating a new one.")
                with open(self.current_tracklet_images_input_dir, 'w') as fp:
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