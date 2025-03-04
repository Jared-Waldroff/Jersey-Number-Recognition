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

from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from ModelDevelopment.SingleImagePipeline import SingleImagePipeline, DataLabelsUniverse
from DataProcessing.Logger import CustomLogger

class CentralPipeline:
  def __init__(self, input_data_path: DataPaths, output_processed_data_path: DataPaths):
    self.input_data_path = input_data_path
    self.output_processed_data_path = output_processed_data_path
    
    # Check if the input directory exists. If not, tell the user.
    if not os.path.exists(self.input_data_path):
      raise FileNotFoundError(f"Input data path does not exist: {self.input_data_path}")
    
    # Check if the output directory exists. If not, create it.
    if not os.path.exists(self.output_processed_data_path):
      os.makedirs(self.output_processed_data_path)
    
    self.data_preprocessor = DataPreProcessing()
    self.image_enhancer = ImageEnhancement()
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
    
  def run_soccernet_pipeline(self, output_folder, num_tracklets=None, num_images_per_tracklet=None, display_transformed_image: bool=False):
    # Obtain the tracklets
    # Iterate over the tracklets
    # And feed each image to the SingleImagePipeline
    self.logger.info("Running the SoccerNet pipeline.")
    self.logger.info(f"num_tracklets: {num_tracklets}")
    
    # Use all of them
    if num_tracklets is None:
      num_tracklets = self.total_tracklets
      
    # Also, to avoid a double call to self.track_result inside generate_features, pass it here.
    # generate_features will subset the total tracklets to unly be until num_tracks internally
    data_dict = self.data_preprocessor.generate_features(self.input_data_path, self.output_processed_data_path, num_tracks=num_tracklets, tracks=self.tracklets)
    
    # Data dict is a set of tracklets. Each tracklet has many many images.
    # We need to loop over the entire set of tracklets.
    # And we must use CUDA wherever possible, if the user has it installed.
    all_tracklets = list(data_dict.keys())
    
    # Instantiate a data augmentor using the data_dict we just created
    self.data_augmentor = DataAugmentation(data_dict)
    
    num_images = 0
    
    # Loop over every tracklet and feed every single image to the SingleImagePipeline
    for tracklet in all_tracklets:
      # Get the images for this tracklet
      images = data_dict[tracklet]
      
      if num_images_per_tracklet is not None:
        images = images[:num_images_per_tracklet]
        
      # NOTE: Check if output already exists and skip if it does
      output_file = os.path.join(output_folder, f"{tracklet}_features.npy")
      if os.path.exists(output_file):
          logger.info(f"Skipping {tracklet} - output already exists")
          processed_count += 1
          continue
      
      # Loop over every image in this tracklet
      for image in images:        
        # Instantiate a single image pipeline.
        # As part of instantiation, that image is pre-processed internally within SingleImagePipeline.
        
        display_transformed_image = num_images <= self.DISP_IMAGE_CAP
        num_images += 1
        
        single_image_pipeline = SingleImagePipeline(
          image,
          model=ModelUniverse.DUMMY.value,
          silence_logs=True,
          display_transformed_image=display_transformed_image
        )
        
        # This run_model call is where we actually do things.
        # It is the full end-to-end soccer net pipeline from main.py, but heavily modularized.
        # This is also where we would adjust the pipeline with our own improvements.
        # Here, we have complete control on the image we are feeding.
        # This is the perfect entry point to call the retrieval module.
        # I.e. for RAC we would be asking: for this image, is it from an underrepresented class?
        # If so, we would like to augment this class with more images during training,
        # but during testing, fetch the retrieval module.
        single_image_pipeline.run_model_chain()