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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

def process_tracklet_worker(args):
    """
    Worker function to process a single tracklet.
    
    Args:
        args (tuple): A tuple of parameters:
            - tracklet (str): Tracklet identifier.
            - images (list/np.array/tensor): The image batch for the tracklet.
            - output_processed_data_path (str): Base output path.
            - use_cache (bool): Flag for cache usage.
            - input_data_path (str): Input data directory.
            - tracklets_to_process (list): List of tracklets.
            - common_processed_data_dir (str): Common directory for processed data.
            - run_soccer_ball_filter (bool): Whether to run soccer ball filter.
            - generate_features (bool): Whether to generate features.
            - run_filter (bool): Whether to run filter.
            - run_legible (bool): Whether to run legibility stage.
            - display_transformed_image_sample (bool): Whether to display transformed samples.
            - suppress_logging (bool): Whether to suppress logging.
    
    Returns:
        str: The tracklet identifier after processing.
    """
    (tracklet, images, output_processed_data_path, use_cache,
     input_data_path, tracklets_to_process, common_processed_data_dir,
     run_soccer_ball_filter, generate_features, run_filter, run_legible,
     display_transformed_image_sample, suppress_logging, num_images_per_tracklet) = args

    # Reinitialize a logger inside the worker.
    logger = CustomLogger().get_logger()

    # Limit images if required.
    if num_images_per_tracklet is not None:
        images = images[:num_images_per_tracklet]

    # Remove cache file if caching is disabled.
    tracklet_data_file_stub = "features.npy"
    if not use_cache:
        tracklet_feature_file = os.path.join(output_processed_data_path, tracklet, tracklet_data_file_stub)
        if os.path.exists(tracklet_feature_file):
            os.remove(tracklet_feature_file)
        logger.info(f"Removed cached tracklet feature file (use_cache: False): {tracklet_feature_file}")

    # Instantiate and run the image batch pipeline for this tracklet.
    pipeline = ImageBatchPipeline(
        raw_tracklet_images_tensor=images,
        current_tracklet_number=tracklet,
        output_tracklet_processed_data_path=os.path.join(output_processed_data_path, tracklet),
        model=ModelUniverse.DUMMY.value,
        display_transformed_image_sample=display_transformed_image_sample,
        suppress_logging=suppress_logging,
        use_cache=use_cache,
        input_data_path=input_data_path,
        output_processed_data_path=output_processed_data_path,
        tracklets_to_process=tracklets_to_process,
        common_processed_data_dir=common_processed_data_dir,
        run_soccer_ball_filter=run_soccer_ball_filter,
        generate_features=generate_features,
        run_filter=run_filter,
        run_legible=run_legible
    )
    pipeline.run_model_chain()
    return tracklet

class CentralPipeline:
    def __init__(self,
                input_data_path: DataPaths,
                gt_data_path: DataPaths,
                output_processed_data_path: DataPaths,
                common_processed_data_dir: DataPaths,
                num_workers: int=8,
                single_image_pipeline: bool=True,
                display_transformed_image_sample: bool=False,
                num_image_samples: int=1,
                use_cache: bool=True,
                suppress_logging: bool=False,
                num_tracklets: int=None,
                num_images_per_tracklet: int=None
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
        self.loaded_illegible_results = None
        self.num_tracklets = num_tracklets
        self.num_images_per_tracklet = num_images_per_tracklet
        self.num_workers = num_workers
        
        self.loaded_ball_tracks = None
        self.analysis_results = None
        
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
        
        self.logger = CustomLogger().get_logger()
        
        # Determine if the user has pytorch cuda
        self.use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.track_result = self.data_preprocessor.get_tracks(self.input_data_path)
        self.tracklets = self.track_result[0]
        self.total_tracklets = self.track_result[1]
        
        self.image_dir = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['crops_folder'])
        self.str_result_file = os.path.join(self.common_processed_data_dir, "str_results.json")
        
        # Constants
        self.LEGAL_TRANSFORMATIONS = list(LegalTransformations.__members__.keys())
        self.DISP_IMAGE_CAP = 1
        self.PADDING = 5
        self.CONFIDENCE_THRESHOLD = 0.4
        
    def init_legibility_classifier_data_file(self):
        self.logger.info("Creating placeholder data files for Legibility Classifier.")
        # Legible data pattern: {tracklet_number: [list of image numbers]}
        for tracklet in self.tracklets_to_process:
            # Allow overwrite if user does not want to use cache
            legible_results_path = os.path.join(self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['legible_result'])
            illegible_results_path = os.path.join(self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['illegible_result'])
            
            if not os.path.exists(legible_results_path) or self.use_cache is False:
                with open(legible_results_path, "w") as outfile:
                    json.dump({tracklet: []}, outfile)
            
            # Illegible data pattern: {'illegible': [list of tracklet numbers]}
            if not os.path.exists(illegible_results_path) or self.use_cache is False:
                with open(illegible_results_path, "w") as outfile:
                    json.dump({'illegible': []}, outfile)
    
    def init_soccer_ball_filter_data_file(self):
        self.logger.info("Creating placeholder data files for Soccer Ball Filter.")
        for tracklet in self.tracklets_to_process:
            # Create one inside each tracklet folder
            soccer_ball_list_path = os.path.join(self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['soccer_ball_list'])
            
            # See if the user specified use_cache=False
            if self.use_cache is False and os.path.exists(soccer_ball_list_path):
                os.remove(soccer_ball_list_path)
            
            # Create only a single file to contain all tracklets
            # Check if the file  exists first
            if not os.path.exists(soccer_ball_list_path):
                # Create the file
                with open(soccer_ball_list_path, "w") as outfile:
                    json.dump({'ball_tracks': []}, outfile)
                
    def set_legibility_results_data(self):
        global_legible_results_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['legible_result'])
        global_illegible_results_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['illegible_result'])

        # 1) If use_cache is True and both global files exist, skip the loop entirely and load from cache.
        if self.use_cache and os.path.exists(global_legible_results_path) and os.path.exists(global_illegible_results_path):
            self.logger.info("Reading legible & illegible results from cache (both global files exist).")
            with open(global_legible_results_path, 'r') as f_leg:
                self.loaded_legible_results = json.load(f_leg)
            with open(global_illegible_results_path, 'r') as f_ill:
                self.loaded_illegible_results = json.load(f_ill)
            return  # Skip re-aggregation altogether

        # 2) Otherwise, we need to re-aggregate from individual tracklet files.
        self.logger.info("Aggregating legible & illegible results (cache not used or only one file is missing).")
        for tracklet in self.tracklets_to_process:
            legible_results_path = os.path.join(self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['legible_result'])
            illegible_results_path = os.path.join(self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['illegible_result'])

            # Aggregate legible results
            with open(legible_results_path, 'r') as openfile:
                legible_results = json.load(openfile)
                if self.loaded_legible_results is None:
                    self.loaded_legible_results = legible_results
                else:
                    for key, val in legible_results.items():
                        if key in self.loaded_legible_results:
                            self.loaded_legible_results[key].extend(val)
                        else:
                            self.loaded_legible_results[key] = val

            # Aggregate illegible results
            # The structure is: {'illegible': [list of tracklet numbers]}
            with open(illegible_results_path, 'r') as openfile:
                illegible_results = json.load(openfile)
                if self.loaded_illegible_results is None:
                    self.loaded_illegible_results = illegible_results
                else:
                    self.loaded_illegible_results['illegible'].extend(illegible_results['illegible'])

        # 3) Write aggregated results back to the global files.
        with open(global_legible_results_path, 'w') as outfile:
            json.dump(self.loaded_legible_results, outfile)

        with open(global_illegible_results_path, 'w') as outfile:
            json.dump(self.loaded_illegible_results, outfile)

        self.logger.info(f"Saved global legible results to: {global_legible_results_path}")
        self.logger.info(f"Saved global illegible results to: {global_illegible_results_path}")
        
    def set_ball_tracks(self):
        global_ball_tracks_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['soccer_ball_list'])

        # 1) If use_cache is True and the global file exists, load and skip aggregation.
        if self.use_cache and os.path.exists(global_ball_tracks_path):
            self.logger.info("Reading ball tracks from cache.")
            with open(global_ball_tracks_path, 'r') as f:
                data = json.load(f)  # data is expected to be {"ball_tracks": [...]}
                self.loaded_ball_tracks = data['ball_tracks']  # Extract the array
            return  # Skip re-aggregation

        # 2) Otherwise, aggregate ball tracks from individual tracklet files.
        self.logger.info("Aggregating ball tracks (cache not used or file missing).")
        aggregated_ball_tracks = []
        for tracklet in self.tracklets_to_process:
            local_ball_tracks_path = os.path.join(self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['soccer_ball_list'])
            with open(local_ball_tracks_path, 'r') as f:
                data = json.load(f)  # data is expected to be {"ball_tracks": [...]}
            # If "ball_tracks" is an array, possibly empty or with a single track name
            aggregated_ball_tracks.extend(data['ball_tracks'])

        self.loaded_ball_tracks = aggregated_ball_tracks

        # 3) Write the aggregated results back to the global file, preserving the same structure.
        with open(global_ball_tracks_path, 'w') as outfile:
            json.dump({"ball_tracks": self.loaded_ball_tracks}, outfile)

        self.logger.info(f"Saved global ball tracks to: {global_ball_tracks_path}")
                
    def init_json_for_pose_estimator(self):
        output_json = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['pose_input_json'])
        
        # IMPORTANT: Always generate the pose input json
        # REASON: If we have a cache from running on 50 tracklets and we want to do the remainder, we should use that!
        # input json needs to be populated with again so we know what to run pose on
        
        self.logger.info("Generating json for pose")
        self.set_legibility_results_data()
        self.logger.info("Done generating json for pose")
        
        #print(f"DEBUG: self.loaded_legible_results: {self.loaded_legible_results}")
        #print(f"DEBUG: self.tracklets_to_process: {self.tracklets_to_process}")
        
        all_files = []
        #print(f"DEBUG: not self.loaded_legible_results is None: {not self.loaded_legible_results is None}")
        if not self.loaded_legible_results is None:
            #print(f"DEBUG: self.loaded_legible_results.keys(): {self.loaded_legible_results.keys()}")
            for key in self.loaded_legible_results.keys():
                #print(f"DEBUG: self.loaded_legible_results[key]: {self.loaded_legible_results[key]}")
                for entry in self.loaded_legible_results[key]:
                    all_files.append(os.path.join(os.getcwd(), entry))
        else:
            for tr in self.tracklets_to_process: # Only run this for the subset of the tracklet universe
                track_dir = os.path.join(self.input_data_path, tr)
                imgs = os.listdir(track_dir)
                
                # Subset the images to only be up to
                imgs = imgs[:self.num_image_samples]
                for img in imgs:
                    all_files.append(os.path.join(track_dir, img))

        #print(f"DEBUG: all_files: {all_files}")
        #print(f"DEBUG: output_json: {output_json}")
        helpers.generate_json(all_files, output_json)
                
    def run_pose_estimation_model(self):
        input_json = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['pose_input_json'])
        output_json = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['pose_output_json'])

        self.logger.info("Detecting pose")
        command = [
            "conda", "run", "-n", config.pose_env, "python",
            f"{os.path.join(Path.cwd().parent.parent, 'StreamlinedPipelineScripts', 'pose.py')}",
            f"{config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py",
            f"{config.pose_home}/checkpoints/vitpose-h.pth",
            "--img-root", "/",
            "--json-file", input_json,
            "--out-json", output_json
        ]

        if self.use_cache:
            command.append("--use_cache")

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.logger.info(result.stdout)  # Log standard output
            self.logger.error(result.stderr)  # Log errors (if any)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running pose estimation: {e}")
            self.logger.info(e.stdout)  # Log stdout even in failure
            self.logger.error(e.stderr)  # Log stderr for debugging

        self.logger.info("Done detecting pose")
        
    # get confidence-filtered points from pose results
    def get_points(self, pose):
        points = pose["keypoints"]
        if len(points) < 12:
            #print("not enough points")
            return []
        relevant = [points[6], points[5], points[11], points[12]]
        result = []
        for r in relevant:
            if r[2] < self.CONFIDENCE_THRESHOLD:
                #print(f"confidence {r[2]}")
                return []
            result.append(r[:2])
        return result
        
    def process_crop(self, entry, all_legible, crops_destination_dir):
        """
        Process a single pose result entry: if the image is in the keep list (all_legible),
        compute a crop based on the pose keypoints and save the cropped image.
        
        Returns a dictionary with keys:
        - "skipped": a dict (possibly empty) with counts per track (derived from image name)
        - "saved": a list of image names that were successfully cropped and saved
        - "miss": count (0 or 1) for this entry if it was skipped due to unreliable points or wrong shape.
        """
        filtered_points = self.get_points(entry)
        img_name = entry["img_name"]
        base_name = os.path.basename(img_name)

        # Skip this entry if the image isn’t in the legible list.
        if base_name not in all_legible:
            return None

        # If no valid keypoints, count as a miss.
        if len(filtered_points) == 0:
            print(f"skipping {img_name}, unreliable points")
            tr = base_name.split('_')[0]
            return {"skipped": {tr: 1}, "saved": [], "miss": 1}

        img = cv2.imread(img_name)
        if img is None:
            print(f"can't find {img_name}")
            return None

        height, width, _ = img.shape
        x_min = min(p[0] for p in filtered_points) - self.PADDING
        x_max = max(p[0] for p in filtered_points) + self.PADDING
        y_min = min(p[1] for p in filtered_points) - self.PADDING
        y_max = max(p[1] for p in filtered_points)
        x1 = int(0 if x_min < 0 else x_min)
        y1 = int(0 if y_min < 0 else y_min)
        x2 = int(width - 1 if x_max > width else x_max)
        y2 = int(height - 1 if y_max > height else y_max)

        crop = img[y1:y2, x1:x2, :]
        h, w, _ = crop.shape
        if h == 0 or w == 0:
            print(f"skipping {img_name}, shape is wrong")
            tr = base_name.split('_')[0]
            return {"skipped": {tr: 1}, "saved": [], "miss": 1}

        out_path = os.path.join(crops_destination_dir, base_name)
        cv2.imwrite(out_path, crop)
        return {"skipped": {}, "saved": [img_name], "miss": 0}

    def generate_crops(self, json_file, crops_destination_dir, all_legible=None):
        """
        Parallelized cropping function.
        
        Arguments:
        - json_file: Path to the JSON file containing pose results.
        - crops_destination_dir: Directory where cropped images will be saved.
        - all_legible: Optionally, a precomputed list of image basenames that are legible.
        
        Returns:
        - skipped: Aggregated dictionary of skipped counts per track.
        - saved: Aggregated list of image names that were successfully processed.
        """
        # Compute all_legible if not provided.
        if all_legible is None:
            all_legible = []
            for key in self.loaded_legible_results.keys():
                for entry in self.loaded_legible_results[key]:
                    all_legible.append(os.path.basename(entry))

        with open(json_file, 'r') as f:
            data = json.load(f)
            all_poses = data["pose_results"]

        # Prepare containers for aggregated results.
        aggregated_skipped = {}
        aggregated_saved = []
        total_misses = 0

        # Use ThreadPoolExecutor for parallel processing.
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.process_crop, entry, all_legible, crops_destination_dir)
                for entry in all_poses
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating crops"):
                result = future.result()
                if result is None:
                    continue
                # Aggregate skipped counts.
                for tr, count in result["skipped"].items():
                    aggregated_skipped[tr] = aggregated_skipped.get(tr, 0) + count
                # Aggregate saved images.
                aggregated_saved.extend(result["saved"])
                total_misses += result["miss"]

        print(f"skipped {total_misses} out of {len(all_poses)}")
        return aggregated_skipped, aggregated_saved
        
    def run_crops_model(self):
        output_json = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['pose_output_json'])
        
        self.logger.info("Generate crops")
        crops_destination_dir = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['crops_folder'], 'imgs')
        Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Before setting legible results data: {self.loaded_legible_results}")
        self.set_legibility_results_data()
        self.logger.info(f"After setting legible results data: {self.loaded_legible_results}")
        self.generate_crops(output_json, crops_destination_dir)
        self.logger.info("Done generating crops")
        
    def run_str_model(self):
        self.logger.info("Predicting numbers")
        os.chdir(str(Path.cwd().parent.parent))  # ensure correct working directory
        print("Current working directory: ", os.getcwd())
        command = [
            "conda", "run", "-n", config.str_env, "python",
            os.path.join("StreamlinedPipelineScripts", "str.py"),
            DataPaths.STR_MODEL.value,
            f"--data_root={self.image_dir}",
            "--batch_size=1",
            "--inference",
            "--result_file", self.str_result_file
        ]
        
        # Cache flag:
        # if self.use_cache:
        #     command.append("--use_cache")

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            # Log standard output and errors
            self.logger.info(result.stdout)
            self.logger.error(result.stderr)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running STR model: {e}")
            self.logger.info(e.stdout)    # Log stdout even in failure
            self.logger.error(e.stderr)   # Log stderr for debugging

        self.logger.info("Done predicting numbers")
    
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

        # Initialize accumulators
        total_correct = 0
        total_tracks = 0
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_TN = 0

        # Read the ground truth file once (shared across workers)
        with open(self.gt_data_path, 'r') as gf:
            gt_dict = json.load(gf)

        # If a soccer ball list is to be used, load it once and pass to workers.
        balls_list = []
        if load_soccer_ball_list:
            # TODO: Change from load_soccer_ball_list to actual path
            with open(load_soccer_ball_list, 'r') as sf:
                balls_json = json.load(sf)
            balls_list = balls_json.get('ball_tracks', [])

        # Define a worker that processes a single tracklet.
        def worker(track):
            tracklet_processed_output_dir = os.path.join(self.output_processed_data_path, track)
            illegible_path = os.path.join(tracklet_processed_output_dir, config.dataset['SoccerNet']['illegible_result'])
            legible_tracklets_path = os.path.join(tracklet_processed_output_dir, config.dataset['SoccerNet']['legible_result'])

            # Read legibility results for this tracklet.
            with open(legible_tracklets_path, 'r') as f:
                legible_tracklets = json.load(f)
            with open(illegible_path, 'r') as f:
                illegible_list = json.load(f).get('illegible', [])

            # Skip processing if this tracklet is in the soccer ball list.
            if track in balls_list:
                return {"correct": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": 0}

            # Get the ground truth value and determine predicted legibility.
            true_value = str(gt_dict[track])
            predicted_legible = self.is_track_legible(track, illegible_list, legible_tracklets)

            # Initialize per-track statistics.
            stats = {"correct": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": 1}

            if true_value == '-1' and not predicted_legible:
                stats["correct"] = 1
                stats["TN"] = 1
            elif true_value != '-1' and predicted_legible:
                stats["correct"] = 1
                stats["TP"] = 1
            elif true_value == '-1' and predicted_legible:
                stats["FP"] = 1
                self.logger.info(f"FP: {track}")
            elif true_value != '-1' and not predicted_legible:
                stats["FN"] = 1
                self.logger.info(f"FN: {track}")

            return stats

        # Use ThreadPoolExecutor to run the worker for each track in parallel.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for track in self.tracklets_to_process:
                futures.append(executor.submit(worker, track))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating legibility"):
                try:
                    result = future.result()
                    total_correct += result["correct"]
                    total_TP += result["TP"]
                    total_TN += result["TN"]
                    total_FP += result["FP"]
                    total_FN += result["FN"]
                    total_tracks += result["total"]
                except Exception as e:
                    self.logger.error(f"Error processing a tracklet: {e}")

        # Compute metrics. Avoid division by zero.
        accuracy = (100 * total_correct / total_tracks) if total_tracks > 0 else 0
        precision = (total_TP / (total_TP + total_FP)) if (total_TP + total_FP) > 0 else 0
        recall = (total_TP / (total_TP + total_FN)) if (total_TP + total_FN) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        self.logger.info(f'Correct {total_correct} out of {total_tracks}. Accuracy {accuracy}%.')
        self.logger.info(f'TP={total_TP}, TN={total_TN}, FP={total_FP}, FN={total_FN}')
        self.logger.info(f"Precision={precision}, Recall={recall}")
        self.logger.info(f"F1={f1}")
        
    def consolidated_results(self, image_dir, dict, illegible_path, soccer_ball_list=None):
        if not soccer_ball_list is None or soccer_ball_list == []:
            self.logger.info("Consolidating results: Using soccer ball list")
            global_ball_tracks_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['soccer_ball_list'])
            with open(global_ball_tracks_path, 'r') as sf:
                balls_json = json.load(sf)
            balls_list = balls_json['ball_tracks']
            for entry in balls_list:
                dict[str(entry)] = 1

        with open(illegible_path, 'r') as f:
            illegile_dict = json.load(f)
        all_illegible = illegile_dict['illegible']
        for entry in all_illegible:
            if not str(entry) in dict.keys():
                dict[str(entry)] = -1

        all_tracks = os.listdir(image_dir)
        for t in all_tracks:
            if not t in dict.keys():
                dict[t] = -1
            else:
                dict[t] = int(dict[t])
        return dict
        
    def combine_results(self):
        # 8. combine tracklet results
        # Read predicted results, stack unique predictions, sum confidence scores for each, choose argmax
        results_dict, self.analysis_results = helpers.process_jersey_id_predictions(self.str_result_file, useBias=True)
        # You may also consider your alternative processing methods below.
        illegible_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['illegible_result'])
        
        #self.logger.info(f"DEBUG STR Results: {self.analysis_results}")
        #self.logger.info(f"DEBUG Results Dict: {results_dict}")
        #self.logger.info(f"DEBUG Illegible Path: {illegible_path}")
        #self.logger.info("DEBUG Soccer Ball List: ", self.loaded_ball_tracks)
        
        # add illegible tracklet predictions (if any)
        self.set_ball_tracks()
        self.consolidated_dict = self.consolidated_results(self.image_dir, results_dict, illegible_path, soccer_ball_list=self.loaded_ball_tracks)

        # Save results as JSON.
        final_results_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['final_result'])
        with open(final_results_path, 'w') as f:
            json.dump(self.consolidated_dict, f)
            
    def evaluate_end_results(self):
        # 9. evaluate accuracy
        final_results_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['final_result'])
        if self.consolidated_dict is None:
            with open(final_results_path, 'r') as f:
                self.consolidated_dict = json.load(f)
        with open(self.gt_data_path, 'r') as gf:
            gt_dict = json.load(gf)
        
        # Instead of evaluating against all keys in gt_dict, evaluate only over the common keys.
        common_keys = set(self.consolidated_dict.keys()).intersection(set(gt_dict.keys()))
        print(f"Evaluating on {len(common_keys)} tracklets (out of {len(gt_dict)} in GT).")
        
        self.evaluate_results(self.consolidated_dict, gt_dict, full_results=self.analysis_results, keys=common_keys)
        
    def evaluate_results(self, consolidated_dict, gt_dict, full_results=None, keys=None):
        """
        Evaluates the consolidated results against the ground truth.
        Only keys in the provided 'keys' set (or the intersection of consolidated_dict and gt_dict, if not provided)
        are evaluated.
        """
        SKIP_ILLEGIBLE = False
        if keys is None:
            keys = set(consolidated_dict.keys()).intersection(set(gt_dict.keys()))
            
        correct = 0
        total = 0
        mistakes = []
        count_of_correct_in_full_results = 0
        
        for tid in keys:
            predicted = consolidated_dict[tid]
            # If a key is missing in gt_dict, skip (or assume a default value of -1)
            true_value = gt_dict.get(tid, -1)
            
            if SKIP_ILLEGIBLE and (str(true_value) == "-1" or str(predicted) == "-1"):
                continue
            if str(true_value) == str(predicted):
                correct += 1
            else:
                mistakes.append(tid)
            total += 1
            
        if total > 0:
            accuracy = 100.0 * correct / total
        else:
            accuracy = 0.0
            
        print(f"Total evaluated tracklets: {total}, correct: {correct}, accuracy: {accuracy}%")
        
        # Additional evaluation details.
        illegible_mistake_count = 0
        illegible_gt_count = 0
        for tid in mistakes:
            if str(consolidated_dict[tid]) == "-1":
                illegible_mistake_count += 1
            elif str(gt_dict[tid]) == "-1":
                illegible_gt_count += 1
            elif full_results is not None and tid in full_results:
                if gt_dict[tid] in full_results[tid]['unique']:
                    count_of_correct_in_full_results += 1
                    
        print(f"Mistakes (illegible predicted): {illegible_mistake_count}")
        print(f"Mistakes (legible but GT illegible): {illegible_gt_count}")
        print(f"Correct in full results but not picked: {count_of_correct_in_full_results}")
        
    def run_soccernet(self,
                      run_soccer_ball_filter=True,
                      generate_features=True,
                      run_filter=True,
                      run_legible=True,
                      run_legible_eval=True,
                      run_pose=True,
                      run_crops=True,
                      run_str=True,
                      run_combine=True,
                      run_eval=True):
        self.logger.info("Running the SoccerNet pipeline.")

        if self.num_tracklets is None:
            self.num_tracklets = self.total_tracklets

        # Phase 0: Generate feature data for all tracklets.
        data_dict = self.data_preprocessor.generate_features(
            self.input_data_path,
            self.output_processed_data_path,
            num_tracks=self.num_tracklets,
            tracks=self.tracklets
        )
        
        # Get length of data dict
        num_images_per_tracklet_local = len(data_dict[list(data_dict.keys())[0]])
        self.logger.info(f"DEBUG Number of images per tracklet (should be < max (1400+)): {num_images_per_tracklet_local}")
        
        # This is our working subset.
        self.tracklets_to_process = list(data_dict.keys())
        
        # These init methods rely on self.tracklets_to_process.
        self.init_soccer_ball_filter_data_file()
        self.init_legibility_classifier_data_file()
        
        # Phase 1: Process each tracklet in parallel.
        tasks = []
        for tracklet in self.tracklets_to_process:
            images = data_dict[tracklet]
            args = (
                tracklet,
                images,
                self.output_processed_data_path,
                self.use_cache,
                self.input_data_path,
                self.tracklets_to_process,
                self.common_processed_data_dir,
                run_soccer_ball_filter,
                generate_features,
                run_filter,
                run_legible,
                self.display_transformed_image_sample,
                self.suppress_logging,
                self.num_images_per_tracklet  # <-- Add this
            )
            tasks.append(args)
            
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_tracklet_worker, task): task[0] for task in tasks}
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Phase 1: Data Pre-Processing Pipeline Progress"):
                try:
                    result = future.result()
                    self.logger.info(f"Processed tracklet: {result}")
                except Exception as e:
                    self.logger.error(f"Error processing tracklet {futures[future]}: {e}")

        # Phase 2: Running the Models on Pre-Processed + Filtered Data sequentially.
        if run_legible_eval:
            self.evaluate_legibility_results()
            
        if run_pose:
            # CRITICAL: Pose processing should occur after legibility results are computed.
            self.init_json_for_pose_estimator()
            self.run_pose_estimation_model()
            
        if run_crops:
            self.run_crops_model()
            
        if run_str:
            self.run_str_model()
        
        if run_combine:
            self.combine_results()
            
        if run_eval:
            self.evaluate_end_results()