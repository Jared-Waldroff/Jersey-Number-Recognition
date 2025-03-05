from DataProcessing.Logger import CustomLogger
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement

class ImageFeatureTransformPipeline:
    def __init__(self, raw_image_batch, output_feature_data_file, model_version='res50_market'):
        self.raw_image_batch = raw_image_batch
        self.output_feature_data_file = output_feature_data_file
        self.model_version = model_version
        
        self.image_enhancer = ImageEnhancement()
        
    def pass_through_gaussian_outliers_filter(self):
        logging.info("Identifying and removing outliers, calling gausian_outliers.py script on tracklet feature files")
        command = f"python gaussian_outliers_streamlined.py --tracklets_folder {image_dir} --output_folder {self.output_feature_data_file}"
        success = os.system(command) == 0
        logging.info("Done removing outliers")
    
    def pass_through_reid_centroid(self, raw_image_batch, output_feature_data_file, model_version='res50_market'):
        """
        Process a raw image (or batch) through the pre-trained centroid model.
        This method:
          - Loads the appropriate model using a config and checkpoint.
          - Ensures input raw_image_batch is 4D (N, C, H, W); if 3D, unsqueezes.
          - Feeds the image through model.backbone and batch-norm layer.
          - Flattens the global features per sample.
          - Appends the feature vector to the output file.
        
        Args:
            raw_image_batch (torch.Tensor): Input tensor (C, H, W) or (N, C, H, W).
            output_feature_data_file (str): File path to save the features.
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
        if raw_image_batch.dim() == 3:
            raw_image_batch = raw_image_batch.unsqueeze(0)
        
        with torch.no_grad():
            input_tensor = raw_image_batch.cuda() if self.use_cuda else raw_image_batch
            _, global_feat = model.backbone(input_tensor)
            global_feat = model.bn(global_feat)
        
        # global_feat shape: (N, d). We keep the batch dimension.
        processed_image = global_feat.cpu().numpy()  # shape: (N, d)
        
        # Append new features to output_feature_data_file:
        # NOTE: The only time we append is when the image tensor batch sent through ImageBatchPipeline is < count(images_in_tracklet).
        # i.e. this would be the case for just passing 2 images through the pipeline, from the same batch, and appending data for img 2 to img 1.
        if os.path.exists(output_feature_data_file):
            existing = np.load(output_feature_data_file, allow_pickle=True)
            combined = np.concatenate([existing, processed_image], axis=0)
            np.save(output_feature_data_file, combined)
        else:
            np.save(output_feature_data_file, processed_image)
        logging.info(f"Saved features for tracklet with shape {processed_image.shape}")
            
        return processed_image
    
    def run_image_transform_pipeline(self):
        """
        Process a single raw image through the centroid model pipeline.
        
        Steps:
          1. Accept a raw image (tensor, file path, or PIL Image) and convert it to a normalized tensor.
          2. Pass the tensor through the centroid model for resizing, cropping, and keyframe identification.
          3. Save the processed features to output_feature_data_file.
        
        Returns:
          np.ndarray: The flattened feature vector for the input image.
        """        
        # IMPORTANT: Filter outliers BEFORE passing through the centroid model.
        self.pass_through_gaussian_outliers_filter()
        self.pass_through_reid_centroid(model_version)