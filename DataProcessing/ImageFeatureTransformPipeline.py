from DataProcessing.Logger import CustomLogger
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from DataProcessing.DataPreProcessing import DataPaths
from reid.CentroidsReidRepo.train_ctl_model import CTLModel
from reid.CentroidsReidRepo.config.defaults import _C as cfg
import torch
import os
import numpy as np
import subprocess

class ImageFeatureTransformPipeline:
    def __init__(self, raw_image_batch, output_feature_data_file, model_version='res50_market'):
        self.raw_image_batch = raw_image_batch
        self.output_feature_data_file = output_feature_data_file
        self.model_version = model_version
        
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
            "StreamlinedPipelineScripts/gaussian_outliers_streamlined.py",
            "--feature_file", self.output_feature_data_file
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.logger.info(result.stdout)  # Print logs from gaussian_outliers_streamlined.py
            self.logger.error(result.stderr)  # Print errors if any
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running gaussian_outliers_streamlined.py: {e}")
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
            self.output_feature_data_file (str): File path to save the features.
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
        
        # Append new features to self.output_feature_data_file:
        # NOTE: The only time we append is when the image tensor batch sent through ImageBatchPipeline is < count(images_in_tracklet).
        # i.e. this would be the case for just passing 2 images through the pipeline, from the same batch, and appending data for img 2 to img 1.
        if os.path.exists(self.output_feature_data_file):
            existing = np.load(self.output_feature_data_file, allow_pickle=True)
            combined = np.concatenate([existing, processed_image], axis=0)
            np.save(self.output_feature_data_file, combined)
        else:
            np.save(self.output_feature_data_file, processed_image)
        self.logger.info(f"Saved features for tracklet with shape {processed_image.shape}")
            
        return processed_image
    
    def run_image_transform_pipeline(self):
        """
        Process a single raw image through the centroid model pipeline.
        
        Steps:
          1. Accept a raw image (tensor, file path, or PIL Image) and convert it to a normalized tensor.
          2. Pass the tensor through the centroid model for resizing, cropping, and keyframe identification.
          3. Save the processed features to self.output_feature_data_file.
        
        Returns:
          np.ndarray: The flattened feature vector for the input image.
        """
        self.pass_through_reid_centroid()  
        self.pass_through_gaussian_outliers_filter()