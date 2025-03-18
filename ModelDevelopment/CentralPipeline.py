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
import subprocess

from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse, CommonConstants
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from ModelDevelopment.ImageBatchPipeline import ImageBatchPipeline, DataLabelsUniverse
from DataProcessing.Logger import CustomLogger
import helpers

class CentralPipeline:
    def __init__(self,
                input_data_path: DataPaths,
                gt_data_path: DataPaths,
                output_processed_data_path: DataPaths,
                common_processed_data_dir: DataPaths,
                single_image_pipeline: bool=True,
                display_transformed_image_sample: bool=False,
                num_image_samples: int=1,
                use_cache: bool=True,
                suppress_logging: bool=False
                ):
        self.input_data_path = input_data_path
        self.gt_data_path = gt_data_path
        self.output_processed_data_path = output_processed_data_path
        self.common_processed_data_dir = common_processed_data_dir
        self.single_image_pipeline = single_image_pipeline
        self.display_transformed_image_sample = display_transformed_image_sample
        self.num_image_samples = num_image_samples
        self.use_cache = use_cache
        self.suppress_logging = suppress_logging
        self.loaded_legible_results = None
        
        self.data_preprocessor = DataPreProcessing(
        display_transformed_image_sample=self.display_transformed_image_sample,
        num_image_samples=self.num_image_samples,
        suppress_logging=self.suppress_logging
        )
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
                
    def set_legible_results_data(self):
        legible_results_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['legible_result'])
        
        if os.path.exists(legible_results_path):
            with open(legible_results_path, 'r') as openfile:
                self.loaded_legible_results = json.load(openfile)
                
    def init_json_for_pose_estimator(self):
        output_json = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['pose_input_json'])
        
        if self.use_cache and os.path.exists(output_json):
            self.logger.info("Pose input json already exists. Skipping generation.")
            return
        
        self.logger.info("Generating json for pose")
        self.set_legible_results_data()
        self.logger.info("Done generating json for pose")
        
        all_files = []
        if not self.loaded_legible_results is None:
            for key in self.loaded_legible_results.keys():
                for entry in self.loaded_legible_results[key]:
                    all_files.append(os.path.join(os.getcwd(), entry))
        else:
            for tr in self.tracks_to_process: # Only run this for the subset of the tracklet universe
                track_dir = os.path.join(self.input_data_path, tr)
                imgs = os.listdir(track_dir)
                for img in imgs:
                    all_files.append(os.path.join(track_dir, img))

        helpers.generate_json(all_files, output_json)
                
    def run_pose_estimation(self):
        input_json = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['pose_input_json'])
        output_json = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['pose_output_json'])

        self.logger.info("Detecting pose")
        command = [
            "conda", "run", "-n", config.pose_env, "python", f"{os.path.join(Path.cwd().parent.parent, 'pose.py')}",
            f"{config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py", # pose config
            f"{config.pose_home}/checkpoints/vitpose-h.pth", # pose checkpoint
            "--img-root", "/",
            "--json-file", input_json,
            "--out-json", output_json
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.logger.info(result.stdout)  # Log standard output
            self.logger.error(result.stderr)  # Log errors (if any)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running pose estimation: {e}")
            self.logger.info(e.stdout)  # Log stdout even in failure
            self.logger.error(e.stderr)  # Log stderr for debugging

        self.logger.info("Done detecting pose")
        
    def run_crops(self):
        output_json = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['pose_output_json'])
        
        self.logger.info("Generate crops")
        crops_destination_dir = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['crops_folder'], 'imgs')
        Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
        self.set_legible_results_data()
        helpers.generate_crops(output_json, crops_destination_dir, self.loaded_legible_results)
        self.logger.info("Done generating crops")
    
    def is_track_legible(self, track, illegible_list, legible_tracklets):
        THRESHOLD_FOR_TACK_LEGIBILITY = 0
        if track in illegible_list:
            return False
        try:
            if len(legible_tracklets[track]) <= THRESHOLD_FOR_TACK_LEGIBILITY:
                return False
        except KeyError:
            return False
        return True
    
    def evaluate_legibility_results(self, load_soccer_ball_list=False):
        self.logger.info(f"Evaluating legibility results on {len(self.tracklets_to_process)} tracklets")
        illegible_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['illegible_result'])
        legible_tracklets_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['legible_result'])
        
        with open(legible_tracklets_path, 'r') as legible_tracklets_path:
            legible_tracklets = json.load(legible_tracklets_path)
        
        with open(self.gt_data_path, 'r') as gf:
            gt_dict = json.load(gf)
        with open(illegible_path, 'r') as gf:
            illegible_list = json.load(gf)
            illegible_list = illegible_list['illegible']

        balls_list = []
        if load_soccer_ball_list is True:
            with open(load_soccer_ball_list, 'r') as sf:
                balls_json = json.load(sf)
            balls_list = balls_json['ball_tracks']

        correct = 0
        total = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        num_per_tracklet_FP = []
        num_per_tracklet_TP = []
        
        # Key adjustment: Run the classification only on the tracklet subset we care about
        for track in self.tracklets_to_process:
            # don't consider soccer balls
            if track in balls_list:
                continue

            true_value = str(gt_dict[track])
            predicted_legible = self.is_track_legible(track, illegible_list, legible_tracklets)
            if true_value == '-1' and not predicted_legible:
                #self.logger.info(f"1){track}")
                correct += 1
                TN += 1
            elif true_value != '-1' and predicted_legible:
                #self.logger.info(f"2){track}")
                correct += 1
                TP += 1
                # if legible_tracklets is not None:
                #     num_per_tracklet_TP.append(len(legible_tracklets[track]))
            elif true_value == '-1' and predicted_legible:
                FP += 1
                self.logger.info(f"FP:{track}")
                # if legible_tracklets is not None:
                #     num_per_tracklet_FP.append(len(legible_tracklets[track]))
            elif true_value != '-1' and not predicted_legible:
                FN += 1
                self.logger.info(f"FN:{track}")
            total += 1

        self.logger.info(f'Correct {correct} out of {total}. Accuracy {100*correct/total}%.')
        self.logger.info(f'TP={TP}, TN={TN}, FP={FP}, FN={FN}')
        Pr = TP / (TP + FP)
        Recall = TP / (TP + FN)
        self.logger.info(f"Precision={Pr}, Recall={Recall}")
        self.logger.info(f"F1={2 * Pr * Recall / (Pr + Recall)}")
        
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
        self.init_json_for_pose_estimator()
        
        num_images = 0
        for tracklet in tqdm(self.tracklets_to_process, desc="Phase 1: Data Pre-Processing Pipeline Progress"):
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
                                            run_pose=run_pose,
                                            run_crops=run_crops,
                                            run_str=run_str,
                                            run_combine=run_combine,
                                            run_eval=run_eval
                                            )
            pipeline.run_model_chain()
        
        # Phase 2: Running the Models on Pre-Processed + Filtered Data
        if run_legible_eval:
            self.evaluate_legibility_results()
            
        if run_pose:
            self.run_pose_estimation()
            
        if run_crops:
            self.run_crops()