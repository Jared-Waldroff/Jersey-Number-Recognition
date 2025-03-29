from enum import Enum
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
from tqdm.notebook import tqdm
import os
import re
import cv2
from PIL import Image
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import logging
import configuration as config
import json
import subprocess
import sys
import StreamlinedPipelineScripts.clip4str as clip4str_module

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
    logger = CustomLogger().get_logger()
    
    try:
        (tracklet, images, output_processed_data_path, use_cache,
         input_data_path, tracklets_to_process, common_processed_data_dir,
         run_soccer_ball_filter, generate_features, run_filter, run_legible,
         display_transformed_image_sample, suppress_logging, num_images_per_tracklet, image_batch_size) = args

        # Limit images if required.
        if num_images_per_tracklet is not None:
            images = images[:num_images_per_tracklet]

        # Log entry using the local logger

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
            run_legible=run_legible,
            image_batch_size=image_batch_size
        )
        pipeline.run_model_chain()
        return tracklet

    except Exception as e:
        # Log the exception with traceback and re-raise it.
        logger.error("Exception in process_tracklet_worker", exc_info=True)
        raise


class CentralPipeline:
    def __init__(self,
                 input_data_path: DataPaths,
                 gt_data_path: DataPaths,
                 output_processed_data_path: DataPaths,
                 common_processed_data_dir: DataPaths,
                 num_workers: int = 8,
                 single_image_pipeline: bool = True,
                 display_transformed_image_sample: bool = False,
                 num_image_samples: int = 1,
                 use_cache: bool = True,
                 suppress_logging: bool = False,
                 num_tracklets: int = None,
                 num_images_per_tracklet: int = None,
                 tracklet_batch_size=32,
                 image_batch_size: int = 200,
                 num_threads_multiplier: int = 3,
                 tracklets_to_process_override: list = None,
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
        self.tracklet_batch_size = tracklet_batch_size
        self.image_batch_size = image_batch_size
        self.num_threads_multiplier = num_threads_multiplier
        self.tracklets_to_process_override = tracklets_to_process_override

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

        self.image_dir = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['crops_folder'])
        self.str_result_file = os.path.join(self.common_processed_data_dir, "str_results.json")

        # Constants
        self.LEGAL_TRANSFORMATIONS = list(LegalTransformations.__members__.keys())
        self.DISP_IMAGE_CAP = 1
        self.PADDING = 5
        self.CONFIDENCE_THRESHOLD = 0.4

        tracks, max_track = self.data_preprocessor.get_tracks(self.input_data_path)
        self.tracklets_to_process = tracks[:self.num_tracklets]

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

    def aggregate_legibility_results_data(self):
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

    def aggregate_pose(self):
        pass

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
        """
        Generate pose input JSON files for each tracklet. This method aggregates
        legibility results and then, for each tracklet, writes a JSON file (stored
        in that tracklet's processed directory) that lists the full paths of the images.
        """
        self.logger.info("Generating json for pose")
        self.aggregate_legibility_results_data()

        # Need to perform a set difference because loaded_legible_results.keys() is just all tracklets, but array will be empty
        # loaded_illegible_results.keys() is only the illegible ones
        #print(self.loaded_legible_results.keys())
        #print(self.loaded_illegible_results["illegible"])
        self.legible_tracklets_list = self.loaded_legible_results.keys() - self.loaded_illegible_results["illegible"]

        # Sort the self.legible_tracklets_list
        self.legible_tracklets_list = sorted(self.legible_tracklets_list)

        #print(f"Legible tracklets: {self.legible_tracklets_list}")

        num_messages = 0

        def worker_from_loaded(key, entries):
            nonlocal num_messages
            # Compute output path for this tracklet
            output_json = os.path.join(self.output_processed_data_path, key, config.dataset['SoccerNet']['pose_input_json'])

            if self.use_cache and os.path.exists(output_json):
                num_messages += 1
                if num_messages == 1:
                    self.logger.info("Used cached data for pose JSON")
                return

            # Build full paths for each image entry
            images = [os.path.join(os.getcwd(), entry) for entry in entries]
            # Generate the JSON for this tracklet
            helpers.generate_json(images, output_json)

        def worker_from_dir(tracklet):
            nonlocal num_messages
            # For tracklets not in loaded_legible_results, we read images from the input directory.
            output_json = os.path.join(self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['pose_input_json'])

            if self.use_cache and os.path.exists(output_json):
                num_messages += 1
                if num_messages == 1:
                    self.logger.info("Used cached data for pose JSON")
                return

            track_dir = os.path.join(self.input_data_path, tracklet)
            imgs = os.listdir(track_dir)
            imgs = imgs[:self.num_image_samples]  # Subset to desired number of images
            # Build full paths; assuming you want the absolute path of each image
            images = [os.path.join(os.getcwd(), track_dir, img) for img in imgs]
            helpers.generate_json(images, output_json)

        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers * self.num_threads_multiplier) as executor:
            if self.loaded_legible_results is not None:
                # Process each tracklet from the loaded legible results
                for key, entries in self.loaded_legible_results.items():
                    futures.append(executor.submit(worker_from_loaded, key, entries))
            else:
                # Process each tracklet from tracklets_to_process by reading its directory.
                for tr in self.tracklets_to_process:
                    futures.append(executor.submit(worker_from_dir, tr))

            # Use tqdm to display progress over all futures
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating pose JSON"):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error processing tracklet for pose json: {e}")

        self.logger.info("Completed generating JSON for pose")

    def run_pose_estimation_model(self, series=False, pyscrippt=False):
        self.logger.info("Detecting pose")

        def worker(tracklet):
            # Build absolute paths for the input and output JSON files for this tracklet
            input_json = os.path.abspath(os.path.join(
                self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['pose_input_json']))
            output_json = os.path.abspath(os.path.join(
                self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['pose_output_json']))

            if not os.path.exists(input_json):
                self.logger.warning(f"[{tracklet}] Input JSON not found: {input_json}")

            # Build absolute path to the pose.py script
            pose_script_path = os.path.abspath(os.path.join(
                Path.cwd().parent.parent, "StreamlinedPipelineScripts", "pose.py"))

            # Config and checkpoint
            pose_config_path = os.path.abspath(os.path.join(
                Path.cwd().parent.parent, config.pose_home, "configs", "body", "2d_kpt_sview_rgb_img",
                "topdown_heatmap", "coco", "ViTPose_huge_coco_256x192.py"))
            pose_checkpoint_path = os.path.abspath(os.path.join(
                Path.cwd().parent.parent, config.pose_home, "checkpoints", "vitpose-h.pth"))

            if pyscrippt:
                # Get direct path to Python in vitpose environment
                vitpose_python = os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "vitpose", "python.exe")

                command = [
                    vitpose_python,  # Use direct path to Python executable instead of conda run
                    f"{os.path.join(Path.cwd().parent.parent, 'StreamlinedPipelineScripts', 'pose.py')}",
                    "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py",
                    "checkpoints/vitpose-h.pth",
                    "--img-root", "/",
                    "--json-file", input_json,
                    "--out-json", output_json
                ]

            else:
                command = [
                    "conda", "run", "-n", config.pose_env, "python", "-u",
                    pose_script_path,
                    "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py",
                    "checkpoints/vitpose-h.pth",
                    "--img-root", "/",
                    "--json-file", input_json,
                    "--out-json", output_json,
                    "--image-batch-size", str(self.image_batch_size)
                ]

            if self.use_cache and os.path.exists(output_json):
                return

            #self.logger.info(f"[{tracklet}] Running command: {' '.join(command)}")

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                for line in iter(process.stdout.readline, ''):
                    if line:
                        self.logger.info(f"[{tracklet}] {line.strip()}")
                process.stdout.close()
                return_code = process.wait()
                if return_code != 0:
                    self.logger.error(f"[{tracklet}] Process returned non-zero exit code: {return_code}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"[{tracklet}] Pose estimation failed: {e}")
                self.logger.info(e.stdout)
                self.logger.error(e.stderr)

        if series:
            # Run in series (safe for GPU memory)
            self.logger.info("Running pose estimation in series")
            for tracklet in tqdm(self.legible_tracklets_list, desc="Running pose estimation (series)", leave=True):
                worker(tracklet)
        else:
            # Run in parallel
            futures = []
            # Heavy duty process so do not multiply workers with the thread multiplier
            self.logger.info(f"Running pose estimation with multithreading with {self.num_workers} threads")
            if self.num_workers > 2:
                self.logger.warning("Using a high number of workers for pose estimation may cause GPU memory issues")
                self.logger.warning("Consider reducing the number of workers or number of images per batch for safety")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for tracklet in self.legible_tracklets_list:
                    futures.append(executor.submit(worker, tracklet))

                for _ in tqdm(as_completed(futures), total=len(futures), desc="Running pose estimation", position=0, leave=True):
                    pass  # tqdm progress

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

        # Parallel processing.
        with ThreadPoolExecutor(max_workers=self.num_workers * self.num_threads_multiplier) as executor:
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
        crops_destination_dir = os.path.join(self.common_processed_data_dir,
                                             config.dataset['SoccerNet']['crops_folder'], 'imgs')
        Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
        # self.logger.info(f"Before setting legible results data: {self.loaded_legible_results}")
        self.aggregate_legibility_results_data()
        # self.logger.info(f"After setting legible results data: {self.loaded_legible_results}")
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

    def run_clip4str_model(self):
        """
        Run the CLIP4STR model for scene text recognition.
        Uses the clip4str.py module to handle the processing.
        """
        self.logger.info("Predicting numbers using CLIP4STR model")
        # Use the built-in string conversion function explicitly
        base_dir = __builtins__['str'](Path.cwd().parent.parent)  # Get base project directory
        os.chdir(base_dir)  # ensure correct working directory
        print("Current working directory: ", os.getcwd())

        # Import the clip4str module with a different name to avoid conflict
        sys.path.append(os.path.join(base_dir, "StreamlinedPipelineScripts"))

        # Set paths
        clip4str_dir = os.path.join(base_dir, "str", "CLIP4STR")
        model_path = os.path.join(clip4str_dir, "pretrained", "clip", "clip4str_huge_3e942729b1.pt")
        clip_pretrained = os.path.join(clip4str_dir, "pretrained", "clip", "appleDFN5B-CLIP-ViT-H-14.bin")
        read_script_path = os.path.join(clip4str_dir, "read.py")

        # Path to Python executable
        python_exe = os.path.join(os.path.expanduser("~"), "miniconda3", "envs", config.clip4str_env, "python.exe")

        # Set environment variables
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Path to images directory and result file
        crops_dir = os.path.join(self.image_dir, 'imgs')
        result_file = self.str_result_file

        # Run CLIP4STR inference using the module
        success = clip4str_module.run_clip4str_inference(
            python_path=python_exe,
            read_script_path=read_script_path,
            model_path=model_path,
            clip_pretrained_path=clip_pretrained,
            images_dir=crops_dir,
            result_file=result_file,
            logger=self.logger,
            env=env
        )

        self.logger.info("Done predicting numbers with CLIP4STR")

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
                #self.logger.info(f"FP: {track}")
            elif true_value != '-1' and not predicted_legible:
                stats["FN"] = 1
                #self.logger.info(f"FN: {track}")

            return stats

        # Use ThreadPoolExecutor to run the worker for each track in parallel.
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers * self.num_threads_multiplier) as executor:
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

    def skip_preprocessing(self, current_tracklet_dir) -> bool:
        if self.use_cache:
            # Files to check if they exist
            files_to_check = [
                "features.npy",
                "illegible_results.json",
                "legible_results.json",
                "main_subject_gauss_th=3.5_r=1.json",
                "main_subject_gauss_th=3.5_r=2.json",
                "main_subject_gauss_th=3.5_r=3.json",
                "soccer_ball.json"
            ]
            # If any one of them is missing, return False (i.e., do NOT skip)
            for file in files_to_check:
                if not os.path.exists(os.path.join(current_tracklet_dir, file)):
                    #self.logger.info("Cache file missing. Preprocessing cannot be skipped")
                    return False

            #self.logger.info(f"Skipping preprocessing for tracklet: {current_tracklet_dir}")
            return True

        return False  # If no use_cache, we never skip
        
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
                      run_eval=True,
                      use_clip4str=True,
                      pyscrippt=True):
        self.logger.info("Running the SoccerNet pipeline.")
        
        if generate_features or run_filter or run_legible:
            # Determine which tracklets to process
            if self.tracklets_to_process_override is None:
                self.logger.info("No tracklets provided. Retrieving from input folder.")
                tracks, max_track = self.data_preprocessor.get_tracks(self.input_data_path)
                tracks = tracks[:self.num_tracklets]
            else:
                self.logger.info(f"Tracklet override applied. Using provided tracklets: {', '.join(self.tracklets_to_process_override)}")
                tracks = self.tracklets_to_process_override
                self.tracklets_to_process = tracks

            final_processed_data = {}

            # Loop over batches of tracklets
            pbar = tqdm(range(0, len(tracks), self.tracklet_batch_size), leave=True, position=0)
            
            self.logger.info(f"Tracklet batch size: {self.tracklet_batch_size}")
            self.logger.info(f"Image batch size: {self.image_batch_size}")
            self.logger.info(f"Number of workers: {self.num_workers}")
            self.logger.info(f"Number of threads created: {self.num_workers * self.num_threads_multiplier}")
            for batch_start in pbar:
                batch_end = min(batch_start + self.tracklet_batch_size, len(tracks))
                batch_tracklets = tracks[batch_start:batch_end]

                # Update tqdm description dynamically with batch range
                pbar.set_description(f"Processing Batch Tracklets ({batch_start}-{batch_end})")

                # Find the index of the first tracklet that isn't cached
                first_uncached_index = None
                for i, tracklet in enumerate(batch_tracklets):
                    tracklet_dir = os.path.join(self.output_processed_data_path, tracklet)
                    if not self.skip_preprocessing(tracklet_dir):
                        # Found a tracklet that needs processing
                        first_uncached_index = i
                        break

                if first_uncached_index is None:
                    # All tracklets in this batch are cached
                    self.logger.info(f"All tracklets in {batch_start}-{batch_end} are cached. Skipping feature generation.")
                    continue
                else:
                    # Subset from the first uncached tracklet to the end of the batch
                    if first_uncached_index > 0:
                        self.logger.info(f"Skipping first {first_uncached_index} cached tracklets in "
                                        f"batch {batch_start}-{batch_end}. Processing the rest.")
                    batch_tracklets = batch_tracklets[first_uncached_index:]

                # Now we only generate features for what we need in this batch
                # Phase 0: Generate feature data for the current batch
                data_dict = self.data_preprocessor.generate_features(
                    self.input_data_path,
                    self.output_processed_data_path,
                    num_tracks=len(batch_tracklets),
                    tracks=batch_tracklets
                )

                # Set working subset for this batch
                self.tracklets_to_process = list(data_dict.keys())

                # Initialize files that rely on self.tracklets_to_process
                self.init_soccer_ball_filter_data_file()
                #self.init_legibility_classifier_data_file()

                # Phase 1: Process each tracklet in parallel for this batch
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
                        self.num_images_per_tracklet,
                        self.image_batch_size
                    )
                    tasks.append(args)

                # If no tasks remain after skipping, move on
                if not tasks:
                    self.logger.info("Should not be here. No tasks to process.")
                    continue

                # RECOMMENDED MULTIPLIER: 3-5x number of workers.
                # e.g. if you have a 14 core CPU and you find 6 cores stable for ProcessPool,
                # you would do 6*3 or 6*5 as input to this ThreadPool
                with ThreadPoolExecutor(max_workers=self.num_workers * self.num_threads_multiplier) as executor:
                    futures = {executor.submit(process_tracklet_worker, task): task[0] for task in tasks}

                    pbar = tqdm(total=len(futures), 
                                desc=f"Processing Batch Tracklets ({batch_start}-{batch_start + self.tracklet_batch_size})", 
                                leave=True, position=0)  # Fixes flickering

                    for future in as_completed(futures):
                        tracklet_name = futures[future]  # Get tracklet name associated with this future
                        if future.exception() is not None:
                            # Log the worker exception BEFORE calling .result()
                            self.logger.error(f"Worker crashed for tracklet {tracklet_name}: {future.exception()}", exc_info=True)
                            continue

                        try:
                            result = future.result()  # Retrieve the successful result
                            self.logger.info(f"Processed tracklet: {result}")
                            # Merge into our overall dictionary
                            final_processed_data[result] = result  # Adjust as needed
                        except Exception as e:
                            # Log unexpected exceptions
                            self.logger.error(f"Unexpected error processing tracklet {tracklet_name}: {e}", exc_info=True)
                        
                        pbar.update(1)  # Ensure tqdm updates properly

                    pbar.close()

        # Phase 2: Running the Models on Pre-Processed + Filtered Data sequentially
        if run_legible_eval:
            self.evaluate_legibility_results()
        if run_pose:
            # CRITICAL: Pose processing should occur after legibility results are computed
            self.init_json_for_pose_estimator()
            self.run_pose_estimation_model(pyscrippt=pyscrippt)
        if run_crops:
            self.run_crops_model()
        if run_str:
            if use_clip4str:
                self.logger.info("Using CLIP4STR model for scene text recognition")
                self.run_clip4str_model()  # Use our new method
            else:
                self.logger.info("Using original model for scene text recognition")
                self.run_str_model()  # Use the original method
        if run_combine:
            self.combine_results()
        if run_eval:
            self.evaluate_end_results()