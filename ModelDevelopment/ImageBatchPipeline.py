from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations
from DataProcessing.ImageFeatureTransformPipeline import ImageFeatureTransformPipeline
from DataProcessing.Logger import CustomLogger
from enum import Enum
import torch
import matplotlib.pyplot as plt
import numpy as np

class DataLabelsUniverse(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    VALIDATION = "VALIDATION"

class ImageBatchPipeline:
    """
    Pipeline to process either a batch of images (e.g. a tracklet) or a single image.
    This class is dedicated for running models on already pre-processed data.
    Image pre-processing is handled as part of class instantiation by implicilty instantiating an ImageFeatureTransfromPipeline class
    """
    def __init__(self,
                raw_image_tensor_batch,
                output_feature_data_file,
                model: ModelUniverse,
                display_transformed_image_sample: bool=False,
                suppress_logging: bool=False,
                use_cache: bool=True):
        self.display_transformed_image_sample = display_transformed_image_sample
        self.raw_image_tensor_batch = raw_image_tensor_batch  # Either shape (C, H, W) or (N, C, H, W)
        self.output_feature_data_file = output_feature_data_file
        self.use_cache = use_cache
        self.image_feature_transform = ImageFeatureTransformPipeline(
          raw_image_batch=raw_image_tensor_batch,
          output_feature_data_file=output_feature_data_file,
          suppress_logging=suppress_logging,
          use_cache=use_cache)
        self.data_preprocessor = DataPreProcessing(suppress_logging=True) # No need for double logging as CentralPipeline already instantiates it
        self.logger = CustomLogger().get_logger()
        
        # Preprocess the image(s) via the transform pipeline.
        self.image_feature_transform.run_image_transform_pipeline()
        
        # NOTE: This is not the image after it was passed through the image transform pipeline because we cannot visualize that.
        if self.display_transformed_image_sample:
            # For display, we assume raw_image_tensor_batch is a single image (or take first if batch).
            single_img = self.raw_image_tensor_batch if self.raw_image_tensor_batch.dim() == 3 else self.raw_image_tensor_batch[0]
            # Denormalize and convert to numpy image for display.
            single_img = DataAugmentation(self.raw_image_tensor_batch).denormalize(single_img)
            img_np = single_img.detach().cpu().permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            plt.title("Raw Image")
            #plt.axis("off")
            plt.show()

    def pass_through_legibility_classifier(self):
      pass
    
    def pass_through_improved_str(self, layer_moe: bool=False, layer_rac: bool=False):
      pass
    
    def pass_through_outlier_filter(self):
      pass
    
    def get_dataset_labels(self, type: DataLabelsUniverse):
      pass
    
    def run_model_chain(self):
        # This is where we actually layer models after already having pre-processed data.
        # As all image pre-processing is handled as part of instnaitating this class, it is guaranteed that we are good to go at this point.
        self.logger.info("Running model chain on preprocessed image(s).")