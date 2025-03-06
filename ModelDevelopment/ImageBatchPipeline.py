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
        
        self.legible_tracklets = {}
        self.illegible_tracklets = []
        
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

    def pass_through_legibility_classifier(self, use_filtered=True, filter='gauss', exclude_balls=True):
      self.logger.info("Classifying legibility of image(s) using pre-trained model.")
      self.legible_tracklets, self.illegible_tracklets = get_soccer_net_legibility_results(args, use_filtered=True, filter='gauss', exclude_balls=True)

      root_dir = config.dataset['SoccerNet']['root_dir']
      image_dir = config.dataset['SoccerNet'][args.part]['images']
      path_to_images = os.path.join(root_dir, image_dir)
      tracklets = os.listdir(path_to_images)

      if use_filtered:
          if filter == 'sim':
              path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                    config.dataset['SoccerNet'][args.part]['sim_filtered'])
          else:
              path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                    config.dataset['SoccerNet'][args.part]['gauss_filtered'])
          with open(path_to_filter_results, 'r') as f:
              filtered = json.load(f)

      if exclude_balls:
          updated_tracklets = []
          soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                          config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
          with open(soccer_ball_list, 'r') as f:
              ball_json = json.load(f)
          ball_list = ball_json['ball_tracks']
          for track in tracklets:
              if not track in ball_list:
                  updated_tracklets.append(track)
          tracklets = updated_tracklets

      for directory in tqdm(tracklets):
          track_dir = os.path.join(path_to_images, directory)
          if use_filtered:
              images = filtered[directory]
          else:
              images = os.listdir(track_dir)
          images_full_path = [os.path.join(track_dir, x) for x in images]
          track_results = lc.run(images_full_path, config.dataset['SoccerNet']['legibility_model'], arch=config.dataset['SoccerNet']['legibility_model_arch'], threshold=0.5)
          legible = list(np.nonzero(track_results))[0]
          if len(legible) == 0:
              self.illegible_tracklets.append(directory)
          else:
              legible_images = [images_full_path[i] for i in legible]
              self.legible_tracklets[directory] = legible_images

      # save results
      json_object = json.dumps(self.legible_tracklets, indent=4)
      full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['legible_result'])
      with open(full_legibile_path, "w") as outfile:
          outfile.write(json_object)

      full_illegibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config. dataset['SoccerNet'][args.part]['illegible_result'])
      json_object = json.dumps({'illegible': self.illegible_tracklets}, indent=4)
      with open(full_illegibile_path, "w") as outfile:
          outfile.write(json_object)
    
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