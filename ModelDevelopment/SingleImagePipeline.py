from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from DataProcessing.Logger import CustomLogger
from enum import Enum
  
class DataLabelsUniverse(Enum):
  TRAIN = "TRAIN"
  TEST = "TEST"
  VALIDATION = "VALIDATION"

class SingleImagePipeline:
  # The main entrypoint for our project pipeline
  def __init__(self, raw_image, model: ModelUniverse, silence_logs: bool=False):
    self.raw_image = raw_image
    self.data_preprocessor = DataPreProcessing(silence_logs=silence_logs)
    self.image_enhancer = ImageEnhancement()
    
    # Pass the raw_image through the preprocessing pipeline
    self.preprocessed_image = self.data_preprocessor.single_image_transform_pipeline(self.raw_image)
    self.logger = CustomLogger().get_logger()
    
  def pass_through_reid(self):
    pass
  
  def pass_through_legibility_classifier(self):
    pass
  
  def pass_through_improved_str(self, layer_moe: bool=False, layer_rac: bool=False):
    pass
  
  def get_dataset_labels(self, type: DataLabelsUniverse):
    pass
  
  def run_model(self):
    self.logger.info("Running model on single image.")
    pass