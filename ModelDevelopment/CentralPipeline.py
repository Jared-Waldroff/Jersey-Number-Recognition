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

from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse, CommonConstants
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from ModelDevelopment.ImageBatchPipeline import ImageBatchPipeline, DataLabelsUniverse
from DataProcessing.Logger import CustomLogger

class CentralPipeline:
  def __init__(self,
               input_data_path: DataPaths,
               output_processed_data_path: DataPaths,
               single_image_pipeline: bool=True,
               display_transformed_image_sample: bool=False,
               num_image_samples: int=1,
               use_cache: bool=True,
               suppress_logging: bool=False
               ):
    self.input_data_path = input_data_path
    self.output_processed_data_path = output_processed_data_path
    self.single_image_pipeline = single_image_pipeline
    self.display_transformed_image_sample = display_transformed_image_sample
    self.num_image_samples = num_image_samples
    self.use_cache = use_cache
    self.suppress_logging = suppress_logging
    
    self.data_preprocessor = DataPreProcessing(display_transformed_image_sample=self.display_transformed_image_sample, num_image_samples=self.num_image_samples)
    self.image_enhancer = ImageEnhancement()
    
    # When the pipeline is first instantiated, ensure the use has all the necessary paths
    self.data_preprocessor.create_data_dirs()
    
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
    
  def run_soccernet(self, num_tracklets=None, num_images_per_tracklet=None):
    self.logger.info("Running the SoccerNet pipeline.")
    if num_tracklets is None:
        num_tracklets = self.total_tracklets
    
    data_dict = self.data_preprocessor.generate_features(self.input_data_path, self.output_processed_data_path, num_tracks=num_tracklets, tracks=self.tracklets)
    all_tracklets = list(data_dict.keys())
    
    num_images = 0
    for tracklet in all_tracklets:
        images = data_dict[tracklet]
        if num_images_per_tracklet is not None:
            images = images[:num_images_per_tracklet]
            
        tracklet_data_file_stub = f"{tracklet}{CommonConstants.FEATURE_DATA_FILE_POSTFIX.value}"
            
        if not self.use_cache:
          # User does not want to use any cached tracklet feature data.
          # Delete the cached data if it exists before proceeding
          tracklet_feature_file = os.path.join(self.output_processed_data_path, tracklet_data_file_stub)
          if os.path.exists(tracklet_feature_file):
            os.remove(tracklet_feature_file)
            self.logger.info(f"Removed cached tracklet feature file (use_cache: False): {tracklet_feature_file}")
        
        # For each tracklet, choose to process the entire batch or each image individually
        # Regardless of whether we do tracklet-level processing or image-level processing,
        # There will only be one feature file per tracklet. The difference is if we read into memory the data file n times (n = number of images in tracklet)
        # or if we only do it once (because we pass the whole tracklet batch). This is excellent for decoupling production code versus research.
        # For research, we can afford to read/write n times because n may only be 1 or 2 as we just want to test the pipeline on a few images.
        if self.single_image_pipeline:
            # Process each image separately
            for image in images:
                display_flag = num_images <= 1
                num_images += 1
                
                # NOTE: This ImageBatchPipeline needs to be instantiated twice.
                # The reason for this is because it does pre-processing implicitly in the constructor.
                # If we call it once for every image, pre-processing runs by-image. Otherwise, it is by tracklet.
                pipeline = ImageBatchPipeline(raw_image_tensor_batch=image,
                                              output_feature_data_file=os.path.join(self.output_processed_data_path, tracklet_data_file_stub),
                                              model=ModelUniverse.DUMMY.value,
                                              display_transformed_image_sample=display_flag,
                                              suppress_logging=self.suppress_logging,
                                              use_cache=self.use_cache,
                                              output_processed_data_path=self.output_processed_data_path)
                pipeline.run_model_chain()
        else:
            # Process the entire batch of images for the whole tracklet
            pipeline = ImageBatchPipeline(raw_image_tensor_batch=images,
                                          output_feature_data_file=os.path.join(self.output_processed_data_path, tracklet_data_file_stub),
                                          model=ModelUniverse.DUMMY.value,
                                          display_transformed_image_sample=self.display_transformed_image_sample,
                                          suppress_logging=self.suppress_logging,
                                          use_cache=self.use_cache,
                                          output_processed_data_path=self.output_processed_data_path)
            pipeline.run_model_chain()