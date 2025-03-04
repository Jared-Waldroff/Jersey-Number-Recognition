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
from ModelDevelopment.ImageBatchPipeline import ImageBatchPipeline, DataLabelsUniverse
from DataProcessing.Logger import CustomLogger

class CentralPipeline:
  def __init__(self, input_data_path: DataPaths, output_processed_data_path: DataPaths, single_image_pipeline: bool=True):
    self.input_data_path = input_data_path
    self.output_processed_data_path = output_processed_data_path
    self.single_image_pipeline = single_image_pipeline
    
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
        
        # For each tracklet, choose to process the entire batch or each image individually
        if self.single_image_pipeline:
            # Process each image separately
            for image in images:
                display_flag = num_images <= 1
                num_images += 1
                pipeline = ImageBatchPipeline(image, output_file=os.path.join(output_folder, f"{tracklet}_features.npy"),
                                              model=ModelUniverse.DUMMY.value, silence_logs=True,
                                              display_transformed_image=display_flag)
                pipeline.run_model_chain()
        else:
            # Process the entire batch of images for the tracklet
            pipeline = ImageBatchPipeline(images, output_file=os.path.join(output_folder, f"{tracklet}_features.npy"),
                                          model=ModelUniverse.DUMMY.value, silence_logs=True,
                                          display_transformed_image=display_transformed_image)
            pipeline.run_model_chain()