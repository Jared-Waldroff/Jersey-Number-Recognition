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
                raw_tracklet_images_tensor,
                output_tracklet_processed_data_path,
                model: ModelUniverse,
                output_processed_data_path: DataPaths,
                input_data_path: DataPaths,
                current_tracklet_number: int,
                tracklets_to_process: list,
                common_processed_data_dir: DataPaths,
                display_transformed_image_sample: bool=False,
                suppress_logging: bool=False,
                use_cache: bool=True):
        self.display_transformed_image_sample = display_transformed_image_sample
        self.raw_tracklet_images_tensor = raw_tracklet_images_tensor  # Either shape (C, H, W) or (N, C, H, W)
        self.output_tracklet_processed_data_path = output_tracklet_processed_data_path
        self.input_data_path = input_data_path
        self.use_cache = use_cache
        self.tracklets_to_process = tracklets_to_process
        self.current_tracklet_number = current_tracklet_number
        self.common_processed_data_dir = common_processed_data_dir
        self.output_processed_data_path = output_processed_data_path
        self.suppress_logging = suppress_logging
        self.image_feature_transform = ImageFeatureTransformPipeline(
          current_tracklet_images_input_dir=os.path.join(self.input_data_path, str(current_tracklet_number)),
          current_tracklet_processed_data_dir=self.output_tracklet_processed_data_path,
          common_processed_data_dir=self.common_processed_data_dir,
          raw_image_batch=self.raw_tracklet_images_tensor,
          output_tracklet_processed_data_path=self.output_tracklet_processed_data_path,
          suppress_logging=self.suppress_logging,
          use_cache=self.use_cache)
        self.data_preprocessor = DataPreProcessing(suppress_logging=True) # No need for double logging as CentralPipeline already instantiates it
        self.logger = CustomLogger().get_logger()
        
        self.legible_tracklets = {}
        self.illegible_tracklets = []
        
        # Preprocess the image(s) via the transform pipeline.
        self.image_feature_transform.run_image_transform_pipeline()
        
        # NOTE: This is not the image after it was passed through the image transform pipeline because we cannot visualize that.
        if self.display_transformed_image_sample:
            # For display, we assume raw_tracklet_images_tensor is a single image (or take first if batch).
            single_img = self.raw_tracklet_images_tensor if self.raw_tracklet_images_tensor.dim() == 3 else self.raw_tracklet_images_tensor[0]
            # Denormalize and convert to numpy image for display.
            single_img = DataAugmentation(self.raw_tracklet_images_tensor).denormalize(single_img)
            img_np = single_img.detach().cpu().permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            plt.title("Raw Image")
            #plt.axis("off")
            plt.show()
            
    def save_json_results(self, path: str, results, task: str):
        self.logger.info(f"Saving {task} to: {path}")

        # Read existing data or initialize it
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        if task == "legible_tracklets":
            # Legible data pattern: {tracklet_number: [list of image numbers]}
            data[self.current_tracklet_number] = results.get(self.current_tracklet_number, [])
        elif task == "illegible_tracklets":
            # Illegible data pattern: {'illegible': [list of tracklet numbers]}
            if 'illegible' not in data:
                data['illegible'] = []
            # Assume 'results' is a tracklet number or a list of them
            if isinstance(results, list):
                data['illegible'].extend(results)
            else:
                data['illegible'].append(results)
        else:
            # Generic save: overwrite entire content with new results
            data = results

        # Write updated data back to the file
        with open(path, "w") as outfile:
            json.dump(data, outfile, indent=4)

        self.logger.info(f"Saved {task} to: {path}")

    def pass_through_legibility_classifier(self, use_filtered=True, filter='gauss', exclude_balls=True):
        self.logger.info("Classifying legibility of image(s) using pre-trained model.")
        
        if use_filtered:
            if filter == 'sim': # Do not use
                path_to_filter_results = os.path.join(self.output_tracklet_processed_data_path, config.dataset['SoccerNet']['sim_filtered'])
            else:
                # Access the params from the config and determine which data file to pull from
                gauss_config = config.dataset['SoccerNet']['gauss_filtered']
                filename = gauss_config['filename']
                threshold = gauss_config['th']
                rounds = gauss_config['r']
                gaussian_filter_lookup_table = f"{filename}_th={threshold}_r={rounds}"
            
                path_to_filter_results = os.path.join(self.output_tracklet_processed_data_path, gaussian_filter_lookup_table)
            
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

        if use_filtered:
            images = filtered[current_tracklet_number]
        else:
            # Otherwise no keep list, just all of them
            images = os.listdir(self.input_data_path)
        
        # images_full_path is either filtered down now or just all images
        images_full_path = [os.path.join(self.input_data_path, str(x)) for x in images]

        # Ship these images over to the legibility classifier
        track_results = lc.run(images_full_path, DataPaths.RESNET_MODEL.value, arch=config.dataset['SoccerNet']['legibility_model_arch'], threshold=0.5)
        legible = list(np.nonzero(track_results))[0]
        
        if len(legible) == 0:
            self.illegible_tracklets.append(self.current_tracklet_number)
        else:
            legible_images = [images_full_path[i] for i in legible]
            self.legible_tracklets[self.current_tracklet_number] = legible_images
                
        # Create dir under output_processed_data_path
        legible_results_path = os.path.join(self.output_tracklet_processed_data_path, config.dataset['SoccerNet']['legible_result'])
        illegible_results_path = os.path.join(self.output_tracklet_processed_data_path, config.dataset['SoccerNet']['illegible_result'])
        
        # NOTE: When saving results, save them to the lookup table at the appropriate key (this tracklet)
        self.save_json_results(self, legible_results_path, self.legible_tracklets, "legible_tracklets")
        self.save_json_results(self, illegible_results_path, self.illegible_tracklets, "illegible_tracklets")
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