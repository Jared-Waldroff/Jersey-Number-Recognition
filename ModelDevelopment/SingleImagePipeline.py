from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from DataProcessing.Logger import CustomLogger
from enum import Enum
from matplotlib import pyplot as plt
  
class DataLabelsUniverse(Enum):
  TRAIN = "TRAIN"
  TEST = "TEST"
  VALIDATION = "VALIDATION"

class SingleImagePipeline:
  # The main entrypoint for our project pipeline
  def __init__(self, raw_image_tensor, output_file, model: ModelUniverse, silence_logs: bool=False, display_transformed_image: bool=False):
    self.display_transformed_image = display_transformed_image
    self.raw_image_tensor = raw_image_tensor
    self.output_file = output_file
    self.data_preprocessor = DataPreProcessing(silence_logs=silence_logs)
    self.image_enhancer = ImageEnhancement()
    
    self.logger = CustomLogger().get_logger()
    
    # Pass the raw_image_tensor through the preprocessing pipeline
    self.preprocessed_image = self.data_preprocessor.single_image_transform_pipeline(self.raw_image_tensor, output_file=self.output_file)
    self.preprocessed_image = self.raw_image_tensor
    self.data_augmentation = DataAugmentation(self.preprocessed_image)
    
    self.preprocessed_image = self.data_augmentation.denormalize(self.preprocessed_image)
    
    if self.display_transformed_image:
      if not silence_logs:
        self.logger.info("Displaying a sample transformed pre-processed image to be passed through the model.")
        
      img = self.preprocessed_image.detach().cpu().permute(1, 2, 0).numpy().astype(float)
      plt.imshow(img)
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
    self.logger.info("Running model on single image.")
    pass