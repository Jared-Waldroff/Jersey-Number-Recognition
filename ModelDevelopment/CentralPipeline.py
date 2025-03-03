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

from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from ModelDevelopment.SingleImagePipeline import SingleImagePipeline, DataLabelsUniverse

class CentralPipeline:
  def __init__(self):
    self.data_preprocessor = DataPreProcessing()
    self.data_augmentor = DataAugmentation()
    self.image_enhancer = ImageEnhancement()
    self.LEGAL_TRANSFORMATIONS = list(LegalTransformations.__members__.keys())
    
    # Determine if the user has pytorch cuda
    self.use_cuda = True if torch.cuda.is_available() else False
    self.device = torch.device('cuda' if self.use_cuda else 'cpu')
    logging.info(f"Using device: {self.device}")
    
  def run_soccernet_pipeline(self):
    # Obtain the tracklets
    # Iterate over the tracklets
    # And feed each image to the SingleImagePipeline
    data_dict = data_pre.generate_features(DataPaths.TRAIN_DATA_DIR.value, DataPaths.TEMP_EXPERIMENT_DIR.value, num_tracks=NUM_TRACKLETS)
    
    # Data dict is a set of tracklets. Each tracklet has many many images.
    # We need to loop over the entire set of tracklets.
    # And we must use CUDA wherever possible, if the user has it installed.