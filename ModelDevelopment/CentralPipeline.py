from enum import Enum
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm
import os
import re
import cv2
from PIL import Image
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import configuration as config
import json

from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse, CommonConstants
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from ModelDevelopment.ImageBatchPipeline import ImageBatchPipeline, DataLabelsUniverse
from DataProcessing.Logger import CustomLogger

class CentralPipeline:
  def __init__(self,
               input_data_path: DataPaths,
               output_processed_data_path: DataPaths,
               common_processed_data_dir: DataPaths,
               single_image_pipeline: bool=True,
               display_transformed_image_sample: bool=False,
               num_image_samples: int=1,
               use_cache: bool=True,
               suppress_logging: bool=False
               ):
    self.input_data_path = input_data_path
    self.output_processed_data_path = output_processed_data_path
    self.common_processed_data_dir = common_processed_data_dir
    self.single_image_pipeline = single_image_pipeline
    self.display_transformed_image_sample = display_transformed_image_sample
    self.num_image_samples = num_image_samples
    self.use_cache = use_cache
    self.suppress_logging = suppress_logging
    
    self.data_preprocessor = DataPreProcessing(display_transformed_image_sample=self.display_transformed_image_sample, num_image_samples=self.num_image_samples)
    self.image_enhancer = ImageEnhancement()
    
    # When the pipeline is first instantiated, ensure the use has all the necessary paths
    self.data_preprocessor.create_data_dirs(self.input_data_path, self.output_processed_data_path)
    
    # Check if the input directory exists. If not, tell the user.
    if not os.path.exists(self.input_data_path):
      raise FileNotFoundError(f"Input data path does not exist: {self.input_data_path}")
    
    # Check if the output directory exists. If not, create it.
    if not os.path.exists(self.output_processed_data_path):
      os.makedirs(self.output_processed_data_path)
    
    self.LEGAL_TRANSFORMATIONS = list(LegalTransformations.__members__.keys())
    self.logger = CustomLogger().get_logger()
    
    # Determine if the user has pytorch cuda
    self.use_cuda = True if torch.cuda.is_available() else False
    self.device = torch.device('cuda' if self.use_cuda else 'cpu')
    self.logger.info(f"Using device: {self.device}")
    
    self.track_result = self.data_preprocessor.get_tracks(self.input_data_path)
    self.tracklets = self.track_result[0]
    self.total_tracklets = self.track_result[1]
    
    self.DISP_IMAGE_CAP = 1
    
  def init_gaussian_outliers_data_files(self):
      # Initialize Gaussian Outliers   
      # If the n number of main results jsons have not been created yet, initialize results to be for all global data.
      # Then write the placeholders to the file before even collecting data.
      # Once that is done (or if jsons already exist), default back to results[r] = [] (single tracklet case)
      # At the end, load the JSON and use the current round and tracklet key to index into the result dict, then save it.
      # This is computationally ok because this is light-weight JSON data. Easy numbers.
      
      # Step 1: Access the params from the config
      gauss_config = config.dataset['SoccerNet']['gauss_filtered']
      filename = gauss_config['filename']
      threshold = gauss_config['th']
      rounds = gauss_config['r']
            
      # Preliminary step: create placeholder data files.
      # This is necessary because we are creating the whole lookup table for all tracklets whereas this function runs on a single tracklet.
      # It is ok for this to run 3 times 
      self.logger.info("Creating placeholder data files for Gaussian Outliers.")
      for r in range(rounds):
          # Construct fstub
          result_file_name = f"{filename}_th={threshold}_r={r+1}.json"
          
          # Check if the result_file_name exists in the common_processed_data_dir
          result_file_path = os.path.join(self.common_processed_data_dir, result_file_name)
          
          # If it does not exist, create it with the empty dict
          if not os.path.exists(result_file_path):
              try:
                self.logger.info(f"Initializing data file: {result_file_path}")
                with open(result_file_path, "w") as outfile:
                    json.dump({x: [] for x in self.tracklets_to_process}, outfile)
              except Exception as e:
                self.logger.error(f"Error creating placeholder data file: {result_file_path}")
                self.logger.error(e)
      
  def init_legibility_classifier_data_file(self):
      self.logger.info("Creating placeholder data files for Legibility Classifier.")
      legible_results_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['legible_result'])
      illegible_results_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['illegible_result'])
      
      # Legible data pattern: {tracklet_number: [list of image numbers]}
      for tracklet in self.tracklets_to_process:
          # Allow overwrite if user does not want to use cache
          if not os.path.exists(legible_results_path) or self.use_cache is False:
              with open(legible_results_path, "w") as outfile:
                  json.dump({tracklet: []}, outfile)
          
      # Illegible data pattern: {'illegible': [list of tracklet numbers]}
      if not os.path.exists(legible_results_path) or self.use_cache is False:
          with open(illegible_results_path, "w") as outfile:
              json.dump({'illegible': []}, outfile)
  
  def init_soccer_ball_filter_data_file(self):
    self.logger.info("Creating placeholder data files for Soccer Ball Filter.")
    soccer_ball_list_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['soccer_ball_list'])
    
    # See if the user specified use_cache=False
    if self.use_cache is False and os.path.exists(soccer_ball_list_path):
      os.remove(soccer_ball_list_path)
      
    # Create only a single file to contain all tracklets
    # Check if the file  exists first
    if not os.path.exists(soccer_ball_list_path):
        with open(soccer_ball_list_path, "w") as outfile:
            json.dump({'ball_tracks': []}, outfile)
            
            actions = {"soccer_ball_filter": False,
                       "feat": False,
                       "filter": False,
                       "legible": False,
                       "legible_eval": False,
                       "pose": False,
                       "crops": False,
                       "str": False,
                       "combine": True,
                       "eval": True}
  
  def run_soccernet(self,
                    num_tracklets=None,
                    num_images_per_tracklet=None,
                    run_soccer_ball_filter=True,
                    generate_features=True,
                    run_filter=True,
                    run_legible=True,
                    run_legible_eval=True,
                    run_pose=True,
                    run_crops=True,
                    run_str=True,
                    run_combine=True,
                    run_eval=True
                    ):
      self.logger.info("Running the SoccerNet pipeline.")
      
      if num_tracklets is None:
          num_tracklets = self.total_tracklets
      
      data_dict = self.data_preprocessor.generate_features(self.input_data_path, self.output_processed_data_path, num_tracks=num_tracklets, tracks=self.tracklets)
      
      # This is different than the total possible universe since we impose a cap through the generate_features call
      self.tracklets_to_process = list(data_dict.keys())
      
      # IMPORTANT: These init methods require self.tracklets_to_process to exist, so they are called below.
      self.init_soccer_ball_filter_data_file() # Even if the filter is not used, algo will just ignore the empty file.
      self.init_gaussian_outliers_data_files()
      self.init_legibility_classifier_data_file()
      
      num_images = 0
      for tracklet in tqdm(self.tracklets_to_process, desc="Central Pipeline Progress"):
          images = data_dict[tracklet]
          if num_images_per_tracklet is not None:
              images = images[:num_images_per_tracklet]
              
          tracklet_data_file_stub = f"features.npy"
              
          if not self.use_cache:
            # User does not want to use any cached tracklet feature data.
            # Delete the cached data if it exists before proceeding
            tracklet_feature_file = os.path.join(self.output_processed_data_path, tracklet, tracklet_data_file_stub)
            if os.path.exists(tracklet_feature_file):
              os.remove(tracklet_feature_file)
              self.logger.info(f"Removed cached tracklet feature file (use_cache: False): {tracklet_feature_file}")

          # Process the entire batch of images for the whole tracklet
          pipeline = ImageBatchPipeline(raw_tracklet_images_tensor=images,
                                        current_tracklet_number=tracklet,
                                        output_tracklet_processed_data_path=os.path.join(self.output_processed_data_path, tracklet),
                                        model=ModelUniverse.DUMMY.value,
                                        display_transformed_image_sample=self.display_transformed_image_sample,
                                        suppress_logging=self.suppress_logging,
                                        use_cache=self.use_cache,
                                        input_data_path=self.input_data_path,
                                        output_processed_data_path=self.output_processed_data_path,
                                        tracklets_to_process=self.tracklets_to_process,
                                        common_processed_data_dir=self.common_processed_data_dir,
                                        run_soccer_ball_filter=run_soccer_ball_filter,
                                        generate_features=generate_features,
                                        run_filter=run_filter,
                                        run_legible=run_legible,
                                        run_legible_eval=run_legible_eval,
                                        run_pose=run_pose,
                                        run_crops=run_crops,
                                        run_str=run_str,
                                        run_combine=run_combine,
                                        run_eval=run_eval
                                        )
          pipeline.run_model_chain()