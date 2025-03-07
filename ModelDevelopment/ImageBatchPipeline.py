from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations
from DataProcessing.ImageFeatureTransformPipeline import ImageFeatureTransformPipeline
from DataProcessing.Logger import CustomLogger
from enum import Enum
import torch
import matplotlib.pyplot as plt
import numpy as np
import configuration as config
import os
import legibility_classifier as lc
import json
from tqdm import tqdm

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
                output_tracklet_processed_data_path,
                model: ModelUniverse,
                output_processed_data_path: DataPaths,
                input_data_path: DataPaths,
                tracklets_to_process: list,
                display_transformed_image_sample: bool=False,
                suppress_logging: bool=False,
                use_cache: bool=True):
        self.display_transformed_image_sample = display_transformed_image_sample
        self.raw_image_tensor_batch = raw_image_tensor_batch  # Either shape (C, H, W) or (N, C, H, W)
        self.output_tracklet_processed_data_path = output_tracklet_processed_data_path
        self.input_data_path = input_data_path
        self.use_cache = use_cache
        self.tracklets_to_process = tracklets_to_process
        self.output_processed_data_path = output_processed_data_path
        self.image_feature_transform = ImageFeatureTransformPipeline(
          raw_image_batch=raw_image_tensor_batch,
          output_tracklet_processed_data_path=output_tracklet_processed_data_path,
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
            
    def save_json_results(self, path: str, results, task: str):
        self.logger.info(f"Saving {task} to: {legible_results_path}")
        if not os.path.exists(path):
            os.makedirs(paths)
            
        json_object = json.dumps(results, indent=4)
        with open(path, "w") as outfile:
            outfile.write(json_object)
        
        self.logger.info(f"Saved {task} to: {legible_results_path}")    

    def pass_through_legibility_classifier(self, use_filtered=True, filter='gauss', exclude_balls=True):
        self.logger.info("Classifying legibility of image(s) using pre-trained model.")
        
        if use_filtered:
            if filter == 'sim':
                path_to_filter_results = os.path.join(self.output_tracklet_processed_data_path, config.dataset['SoccerNet']['sim_filtered'])
            else:
                path_to_filter_results = os.path.join(self.output_tracklet_processed_data_path, config.dataset['SoccerNet']['gauss_filtered'])
            with open(path_to_filter_results, 'r') as f:
                filtered = json.load(f)

        if exclude_balls:
            updated_tracklets = []
            soccer_ball_list = os.path.join(self.output_tracklet_processed_data_path, config.dataset['SoccerNet']['soccer_ball_list'])
            
            # Check if the soccer_ball_list even exists first, and if not, skip
            if not os.path.exists(soccer_ball_list):
                self.logger.warning("No soccer ball list found. Skipping exclusion of soccer balls.")
                self.logger.info(f"Path checked: {soccer_ball_list}")
            else:
                with open(soccer_ball_list, 'r') as f:
                    ball_json = json.load(f)
                ball_list = ball_json['ball_tracks']
                for track in self.tracklets_to_process:
                    if not track in ball_list:
                        updated_tracklets.append(track)
                self.tracklets_to_process = self.tracklets_to_process

        # Loop over our subset of the available universe
        # NOTE: This part is not ready yet.
        if use_filtered:
            images = filtered # We maintain one file per tracklet
        else:
            images = self.tracklets_to_process
        images_full_path = [os.path.join(self.input_data_path, str(x)) for x in images]
        print(images_full_path)
        track_results = lc.run(images_full_path, DataPaths.RESNET_MODEL.value, arch=config.dataset['SoccerNet']['legibility_model_arch'], threshold=0.5)
        legible = list(np.nonzero(track_results))[0]
        if len(legible) == 0:
            self.illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            self.legible_tracklets[directory] = legible_images
                
        # Create dir under output_processed_data_path
        legible_results_path = os.path.join(self.output_tracklet_processed_data_path, config.dataset['SoccerNet']['legible_result'])
        illegible_results_path = os.path.join(self.output_tracklet_processed_data_path, config.dataset['SoccerNet']['illegible_result'])
        
        self.save_json_results(self, legible_results_path, self.legible_tracklets, "legible tracklets")
        self.save_json_results(self, illegible_results_path, self.illegible_tracklets, "illegible tracklets")
        self.logger.info("Legibility classification complete.")
    
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
        
        # Step 1: Legibility Classifier
        self.pass_through_legibility_classifier()