from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from DataProcessing.Logger import CustomLogger
from enum import Enum
import torch
import matplotlib.pyplot as plt

class DataLabelsUniverse(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    VALIDATION = "VALIDATION"

class ImageBatchPipeline:
    """
    Pipeline to process either a batch of images (e.g. a tracklet) or a single image.
    This class performs preprocessing (including the centroid model pass) and further operations.
    """
    def __init__(self, raw_image_tensor, output_file, model: ModelUniverse, silence_logs: bool=False, display_transformed_image: bool=False):
        self.display_transformed_image = display_transformed_image
        self.raw_image_tensor = raw_image_tensor  # Either shape (C, H, W) or (N, C, H, W)
        self.output_file = output_file
        self.data_preprocessor = DataPreProcessing(silence_logs=silence_logs)
        self.image_enhancer = ImageEnhancement()
        self.logger = CustomLogger().get_logger()
        
        # Preprocess the image(s) via the transform pipeline.
        # The method single_image_transform_pipeline expects a raw image tensor and returns features.
        self.preprocessed_features = self.data_preprocessor.single_image_transform_pipeline(self.raw_image_tensor, self.output_file)
        
        # Optionally, you can then apply further enhancement.
        # If you're processing a single image, self.preprocessed_features has shape (1, d) or if not, shape (N, d)
        # For display, we might want to convert back to image format.
        if self.display_transformed_image:
            # For display, we assume raw_image_tensor is a single image (or take first if batch).
            single_img = self.raw_image_tensor if self.raw_image_tensor.dim() == 3 else self.raw_image_tensor[0]
            # Denormalize and convert to numpy image for display.
            single_img = DataAugmentation(self.raw_image_tensor).denormalize(single_img)
            img_np = single_img.detach().cpu().permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            plt.title("Transformed Preprocessed Image")
            plt.axis("off")
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
        self.logger.info("Running model chain on preprocessed image(s).")
        # Insert further processing: legibility classifier, improved STR, etc.
        # For now, simply log and return preprocessed features.
        return self.preprocessed_features