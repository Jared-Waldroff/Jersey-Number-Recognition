import threading
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
import threading
import sys
import StreamlinedPipelineScripts.clip4str as clip4str_module
from multiprocessing import Semaphore

from DataProcessing.DataPreProcessing import DataPreProcessing, DataPaths, ModelUniverse, CommonConstants
from DataProcessing.DataAugmentation import DataAugmentation, LegalTransformations, ImageEnhancement
from ModelDevelopment.ImageBatchPipeline import ImageBatchPipeline, DataLabelsUniverse
from DataProcessing.Logger import CustomLogger
import helpers

PROCESS_POOL_GPU_SEMAPHORE = Semaphore(6) # ProcessPoolExecutor semaphore

def pose_worker(tracklet, output_processed_data_path, image_batch_size,
                  pose_env, pose_home, use_cache, logging_config, pyscript):
    """
    Process one tracklet: build paths and run the pose.py command.
    """
    # Setup logging in the worker process
    logger = logging.getLogger(f"pose_worker_{tracklet}")
    logging.basicConfig(**logging_config)
    
    logger.info(f"Worker called for tracklet: {tracklet}")
    
    # Use absolute paths exclusively, avoid Path.cwd() which may differ in the worker process
    cwd = os.path.abspath(os.getcwd())
    parent_dir = os.path.dirname(os.path.dirname(cwd))
    
    # Build absolute paths for input/output JSON for this tracklet.
    input_json = os.path.abspath(os.path.join(
        output_processed_data_path, tracklet, config.dataset['SoccerNet']['pose_input_json']))
    output_json = os.path.abspath(os.path.join(
        output_processed_data_path, tracklet, config.dataset['SoccerNet']['pose_output_json']))

    if not os.path.exists(input_json):
        logger.warning(f"[{tracklet}] Input JSON not found: {input_json}")
        return  # Return early if input file doesn't exist

    # Build absolute path to the pose.py script using the explicit parent directory
    pose_script_path = os.path.abspath(os.path.join(
        parent_dir, "StreamlinedPipelineScripts", "pose.py"))

    # Build absolute paths for config and checkpoint files
    pose_config_path = os.path.abspath(os.path.join(
        parent_dir, pose_home, "configs", "body", "2d_kpt_sview_rgb_img",
        "topdown_heatmap", "coco", "ViTPose_huge_coco_256x192.py"))
    pose_checkpoint_path = os.path.abspath(os.path.join(
        parent_dir, pose_home, "checkpoints", "vitpose-h.pth"))

    if pyscript:
        # Use direct path to python.exe in the specified conda environment.
        vitpose_python = os.path.join(os.path.expanduser("~"), "miniconda3", "envs", pose_env, "python.exe")
        env_vars = os.environ.copy()
        env_vars["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        command = [
            vitpose_python,
            os.path.abspath(os.path.join(Path.cwd().parent.parent, "StreamlinedPipelineScripts", "pose.py")),
            os.path.join(pose_home, "configs", "body", "2d_kpt_sview_rgb_img", "topdown_heatmap", "coco", "ViTPose_huge_coco_256x192.py"),
            os.path.join(pose_home, "checkpoints", "vitpose-h.pth"),
            "--img-root", "/",
            "--json-file", input_json,
            "--out-json", output_json
        ]
    else:
        logger.info("Using conda run for pose estimation")
        command = [
            "conda", "run", "-n", pose_env, "python", "-u",
            pose_script_path,
            pose_config_path,
            pose_checkpoint_path,
            "--img-root", "/",
            "--json-file", input_json,
            "--out-json", output_json,
            "--image-batch-size", str(image_batch_size)
        ]

    if use_cache and os.path.exists(output_json):
        logger.info(f"[{tracklet}] Output JSON exists, skipping: {output_json}")
        return

    logger.info(f"[{tracklet}] Running command: {' '.join(command)}")
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
                logger.info(f"[{tracklet}] {line.strip()}")
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            logger.error(f"[{tracklet}] Process returned non-zero exit code: {return_code}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{tracklet}] Pose estimation failed: {e}")
        logger.info(e.stdout)
        logger.error(e.stderr)

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
            image_batch_size=image_batch_size,
            GPU_SEMAPHORE=PROCESS_POOL_GPU_SEMAPHORE
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
                 use_cache: bool = True,
                 suppress_logging: bool = False,
                 num_tracklets: int = None,
                 num_images_per_tracklet: int = None,
                 tracklet_batch_size=32,
                 image_batch_size: int = 200,
                 num_threads_multiplier: int = 3,
                 tracklets_to_process_override: list = None,
                 use_image_enhancement: bool = False
                 ):
        self.input_data_path = input_data_path
        self.gt_data_path = gt_data_path
        self.output_processed_data_path = output_processed_data_path
        self.common_processed_data_dir = common_processed_data_dir
        self.display_transformed_image_sample = display_transformed_image_sample
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
        self.use_image_enhancement = use_image_enhancement
        
        # If tracklets_to_process_override is a series of ints, convert them to strings
        if self.tracklets_to_process_override is not None and isinstance(self.tracklets_to_process_override, list):
            self.tracklets_to_process_override = [str(x) for x in self.tracklets_to_process_override]
        
        self.loaded_ball_tracks = None
        self.analysis_results = None

        self.data_preprocessor = DataPreProcessing(
            display_transformed_image_sample=self.display_transformed_image_sample,
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

        # TODO: Deprecate this
        self.image_dir = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['crops_folder'])

        # Constants
        self.LEGAL_TRANSFORMATIONS = list(LegalTransformations.__members__.keys())
        self.DISP_IMAGE_CAP = 1
        self.PADDING = 5
        self.CONFIDENCE_THRESHOLD = 0.4
        
        if self.tracklets_to_process_override is not None:
            self.tracklets_to_process = self.tracklets_to_process_override
        else:
            tracks, max_track = self.data_preprocessor.get_tracks(self.input_data_path)
            self.tracklets_to_process = tracks[:self.num_tracklets]
        
        # Limit concurrent GPU calls (example).
        # CRUCIAL to prevent too many parallel shipments to our GPU to prevent CUDA-out-of-memory issues
        # This will become a bottleneck as we enter series code here, but necessary to avoid exploding GPUs.
        self.GPU_SEMAPHORE = threading.Semaphore(value=6) # ThreadPoolExecutor semaphore
        
        self.image_enhancement = ImageEnhancement()
        
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
        """Aggregates pose results from individual tracklets into a global file."""
        self.logger.info("Aggregating pose results")

        # Path for the global pose results
        global_pose_results_path = os.path.join(self.common_processed_data_dir,
                                                config.dataset['SoccerNet']['pose_output_json'])

        # If cache is enabled and global file exists, we can skip aggregation
        if self.use_cache and os.path.exists(global_pose_results_path):
            self.logger.info("Using cached global pose results")
            return

        # Initialize empty list to hold all pose results
        all_pose_results = []

        # Process each legible tracklet
        for tracklet in self.legible_tracklets_list:
            tracklet_pose_path = os.path.join(self.output_processed_data_path,
                                              tracklet,
                                              config.dataset['SoccerNet']['pose_output_json'])

            # Skip if the tracklet doesn't have pose results
            if not os.path.exists(tracklet_pose_path):
                continue

            # Read the tracklet's pose results
            with open(tracklet_pose_path, 'r') as f:
                tracklet_pose_data = json.load(f)

            # Append to our global list
            if 'pose_results' in tracklet_pose_data:
                all_pose_results.extend(tracklet_pose_data['pose_results'])

        # Write the aggregated results to the global file
        with open(global_pose_results_path, 'w') as f:
            json.dump({"pose_results": all_pose_results}, f)

        self.logger.info(f"Saved aggregated pose results to: {global_pose_results_path}")
    
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
        
    def set_legibility_arrays(self):
        all_tracklets = {int(k) for k in self.loaded_legible_results.keys()}
        illegible = {int(x) for x in self.loaded_illegible_results["illegible"]}
        legible_set = all_tracklets - illegible
        self.legible_tracklets_list = sorted(legible_set)
        self.legible_tracklets_list = [str(x) for x in self.legible_tracklets_list]
        self.logger.info(f"Legible tracklets list: {', '.join([x for x in self.legible_tracklets_list])}")
                
    def init_json_for_pose_estimator(self):
        """
        Generate pose input JSON files for each tracklet. This method aggregates
        legibility results and then, for each tracklet, writes a JSON file (stored
        in that tracklet's processed directory) that lists the full paths of the images.
        """
        self.logger.info("Generating json for pose")
        self.aggregate_legibility_results_data()
        self.set_legibility_arrays()

        num_messages = 0

        def worker_from_loaded(key, entries):
            """
            Worker that processes tracklets we already have in self.loaded_legible_results.
            Returns a list of log messages so we can print them in the main thread.
            """
            nonlocal num_messages
            log_messages = []
            output_json = os.path.join(self.output_processed_data_path, key, config.dataset['SoccerNet']['pose_input_json'])

            if self.use_cache and os.path.exists(output_json):
                num_messages += 1
                if num_messages == 1:
                    # Instead of self.logger.info, store the message and return it.
                    log_messages.append("Used cached data for pose JSON")
                return log_messages

            # Build full paths for each image entry
            images = [os.path.join(os.getcwd(), entry) for entry in entries]
            helpers.generate_json(images, output_json)

            log_messages.append(f"Generated pose JSON for tracklet {key} with {len(images)} images.")
            return log_messages

        def worker_from_dir(tracklet):
            """
            Worker that processes tracklets NOT in loaded_legible_results. We read images from a folder.
            Returns a list of log messages so we can print them in the main thread.
            """
            nonlocal num_messages
            log_messages = []
            output_json = os.path.join(self.output_processed_data_path, tracklet, config.dataset['SoccerNet']['pose_input_json'])

            if self.use_cache and os.path.exists(output_json):
                num_messages += 1
                if num_messages == 1:
                    log_messages.append("Used cached data for pose JSON")
                return log_messages

            track_dir = os.path.join(self.input_data_path, tracklet)
            imgs = os.listdir(track_dir)
            
            self.logger.info(f"Images before any subsetting: {len(imgs)}")

            # Subset to desired number of images
            if self.num_images_per_tracklet is not None:
                log_messages.append(f"Subsetting images to {self.num_images_per_tracklet} for tracklet {tracklet}")
                imgs = imgs[:self.num_images_per_tracklet]
                
            self.logger.info(f"Images after subsetting: {len(imgs)}")

            images = [os.path.join(os.getcwd(), track_dir, img) for img in imgs]
            log_messages.append(f"Constructed images for tracklet {tracklet}: {','.join(images)}")
            helpers.generate_json(images, output_json)
            log_messages.append(f"Generated pose JSON for tracklet {tracklet} with {len(images)} images.")

            return log_messages

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
                    worker_log_messages = future.result()  # list of strings
                    # Now we do the actual logging on the main thread
                    for msg in worker_log_messages:
                        self.logger.info(msg)
                except Exception as e:
                    self.logger.error(f"Error processing tracklet for pose json: {e}")

        self.logger.info("Completed generating JSON for pose")
                
    def run_pose_estimation_model(self, series=False, pyscript=False):
        self.logger.info("Detecting pose")

        # If running in series, call the worker function for each tracklet sequentially.
        if series:
            self.logger.info("Running pose estimation in series")
            for tracklet in tqdm(self.legible_tracklets_list, desc="Running pose estimation (series)", leave=True):
                pose_worker(tracklet,
                                self.output_processed_data_path,
                                self.image_batch_size,
                                config.pose_env,
                                config.pose_home,
                                self.use_cache,
                                self.logger,
                                pyscript)
        else:
            self.logger.info(f"Legible tracklets list: {', '.join(self.legible_tracklets_list)}")
            futures = []
            self.logger.info(f"Running pose estimation with multiprocessing using {self.num_workers} workers")
            if self.num_workers > 2:
                self.logger.warning("High number of workers may cause GPU memory issues; consider reducing workers or images per batch.")
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            # Configure logging for processes
            logging_config = {
                'level': "INFO",
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                # Add other logger configuration if needed
            }
            
            # NOTE: ProcessPool is 25% faster but shoes no logs. ThreadPool shows us logs so might be better to just stick to ThreadPool
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for tracklet in self.legible_tracklets_list:
                    futures.append(executor.submit(pose_worker,
                                                tracklet,
                                                self.output_processed_data_path,
                                                self.image_batch_size,
                                                config.pose_env,
                                                config.pose_home,
                                                self.use_cache,
                                                logging_config,  # Pass config instead of logger object
                                                pyscript))
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Running pose estimation", position=0, leave=True):
                    pass  # Simply update progress

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
        compute a crop based on the pose keypoints and optionally enhance it.
        
        Returns a dictionary with:
        - "skipped": a dict (possibly empty) with counts per track
        - "saved": list of image names successfully processed
        - "miss": count of skipped entries due to invalid conditions
        """
        filtered_points = self.get_points(entry)
        img_name = entry["img_name"]
        base_name = os.path.basename(img_name)

        if base_name not in all_legible:
            return None

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

        # ========== Optional Enhancement ==========
        if getattr(self, "use_image_enhancement", False):
            # ------------------------------------------------
            # 1) Convert OpenCV crop (BGR) -> RGB -> Torch Tensor
            # ------------------------------------------------
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = transforms.ToTensor()(crop_rgb)  # (C, H, W)

            # ------------------------------------------------
            # 2) Enhance using the custom image enhancement pipeline
            # ------------------------------------------------
            enhanced_tensor = self.image_enhancement.enhance_image(crop_tensor)
            
            # ------------------------------------------------
            # 3) Convert back to NumPy (BGR) for cv2.imwrite
            # ------------------------------------------------
            # clamp to [0, 1] so no unexpected float rounding errors
            enhanced_clamped = enhanced_tensor.clamp(0, 1)
            enhanced_pil = transforms.ToPILImage()(enhanced_clamped)
            enhanced_rgb = np.array(enhanced_pil)
            crop = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)  # back to OpenCV format

        out_path = os.path.join(crops_destination_dir, base_name)
        cv2.imwrite(out_path, crop)

        return {"skipped": {}, "saved": [img_name], "miss": 0}


    def generate_crops(self, json_file, crops_destination_dir, all_legible=None):
        """
        Serial cropping function (no internal parallelization).
        
        Arguments:
            - json_file: Path to the JSON file containing pose results.
            - crops_destination_dir: Directory where cropped images will be saved.
            - all_legible: Optionally, a precomputed list of image basenames that are legible.
            
        Returns:
            - (aggregated_skipped, aggregated_saved)
        """
        # If not provided, gather up all 'legible' entries from self.loaded_legible_results
        if all_legible is None:
            all_legible = []
            for key in self.loaded_legible_results.keys():
                for entry in self.loaded_legible_results[key]:
                    all_legible.append(os.path.basename(entry))

        # Read the pose_results
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_poses = data["pose_results"]

        aggregated_skipped = {}
        aggregated_saved = []
        total_misses = 0

        # Simple serial loop over each pose entry
        for entry in tqdm(all_poses, desc="Generating crops"):
            result = self.process_crop(entry, all_legible, crops_destination_dir)
            if result is None:
                continue

            # Accumulate "skipped" dictionary
            for tr, count in result["skipped"].items():
                aggregated_skipped[tr] = aggregated_skipped.get(tr, 0) + count

            # Accumulate "saved" list
            aggregated_saved.extend(result["saved"])
            total_misses += result["miss"]

        print(f"Skipped {total_misses} out of {len(all_poses)} for {json_file}")
        return aggregated_skipped, aggregated_saved

    def run_crops_model(self):
        """
        Runs the crops model for all tracklets (sequentially).
        """
        # Ensure we have up-to-date legibility results
        self.aggregate_legibility_results_data()
        self.set_legibility_arrays()

        # If no tracklets to process, nothing to do.
        if not self.legible_tracklets_list:
            self.logger.warning("No tracklets found; nothing to do.")
            return

        # Prepare aggregated counters for all tracklets
        aggregated_skipped = {}
        aggregated_saved = []

        # Process each tracklet in a simple for-loop
        for tracklet in tqdm(self.legible_tracklets_list, desc="Generating crops for tracklets", leave=True):
            tracklet_processed_output_dir = os.path.join(self.output_processed_data_path, tracklet)
            output_json = os.path.join(tracklet_processed_output_dir, config.dataset['SoccerNet']['pose_output_json'])
            crops_destination_dir = os.path.join(tracklet_processed_output_dir, config.dataset['SoccerNet']['crops_folder'])
            
            # CACHING:
            if self.use_cache and os.path.exists(crops_destination_dir):
                self.logger.info(f"Skipping tracklet {tracklet} (cache found at {crops_destination_dir}).")
                continue

            # Ensure the crops folder exists
            Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
            
            # Build the all_legible for JUST this tracklet
            tracklet_legible_paths = self.loaded_legible_results.get(tracklet, [])
            all_legible_for_this_tracklet = [os.path.basename(p) for p in tracklet_legible_paths]

            try:
                # Call generate_crops for this single tracklet
                skipped, saved = self.generate_crops(
                    json_file=output_json,
                    crops_destination_dir=crops_destination_dir,
                    all_legible=all_legible_for_this_tracklet
                )

                # Log partial results
                self.logger.info(f"Done generating crops for tracklet {tracklet}.")
                self.logger.info(f"Skipped dictionary for {tracklet}: {skipped}")
                self.logger.info(f"Total saved images for {tracklet}: {len(saved)}")

                # Merge the per-tracklet results into aggregated counters
                for tr, count in skipped.items():
                    aggregated_skipped[tr] = aggregated_skipped.get(tr, 0) + count
                aggregated_saved.extend(saved)

            except Exception as e:
                self.logger.error(f"Error running crop generation for tracklet {tracklet}: {e}")

        # After processing all tracklets, log final aggregated results
        self.logger.info("Done generating crops (sequential) for all tracklets.")
        self.logger.info(f"Aggregated skipped: {aggregated_skipped}")
        self.logger.info(f"Total saved across all tracklets: {len(aggregated_saved)}")
        
    def run_str_model(self):
        self.logger.info("Predicting numbers")
        self.aggregate_legibility_results_data()
        self.set_legibility_arrays()
        
        # Ensure correct working directory.        
        os.chdir(str(Path.cwd().parent.parent))
        print("Current working directory: ", os.getcwd())
        
        def run_str_for_tracklet(tracklet):
            # Build the processed data path for the current tracklet.
            processed_data_path = os.path.join(self.output_processed_data_path, tracklet)
            processed_data_path_crops = os.path.join(processed_data_path, config.dataset['SoccerNet']['crops_folder'])
            
            # Construct the path to the result file.
            result_file = os.path.join(processed_data_path, config.dataset['SoccerNet']['str_results_file'])
            
            # If caching is enabled and the result file already exists, skip running the command.
            if self.use_cache and os.path.exists(result_file):
                self.logger.info(f"Skipping tracklet {tracklet} (cache found at {result_file}).")
                return tracklet, f"Cached: {result_file}"
            
            # Build the command; here we pass the tracklet's processed data path as the --data_root.
            command = [
                "conda", "run", "-n", config.str_env, "python",
                os.path.join("StreamlinedPipelineScripts", "str.py"),
                DataPaths.STR_MODEL.value,
                f"--data_root={processed_data_path_crops}",
                "--batch_size=1",
                "--inference",
                "--result_file", result_file,
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                self.logger.info(f"Tracklet {tracklet} stdout: {result.stdout}")
                self.logger.error(f"Tracklet {tracklet} stderr: {result.stderr}")
                return tracklet, result.stdout
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error processing tracklet {tracklet}: {e}")
                self.logger.info(e.stdout)
                self.logger.error(e.stderr)
                return tracklet, None

        # Use a ThreadPoolExecutor to run the STR inference in parallel for each tracklet.
        futures = {}
        with ThreadPoolExecutor(max_workers=self.num_workers * self.num_threads_multiplier) as executor:
            for tracklet in tqdm(self.legible_tracklets_list, desc="Dispatching STR on tracklets", leave=True):
                futures[executor.submit(run_str_for_tracklet, tracklet)] = tracklet

            # Process results as they complete.
            for future in tqdm(as_completed(futures), total=len(futures), desc="Aggregating STR results", leave=True):
                tracklet = futures[future]
                try:
                    tracklet, output = future.result()
                    # Optionally, aggregate or store the output per tracklet.
                except Exception as e:
                    self.logger.error(f"Error processing tracklet {tracklet}: {e}")

        self.logger.info("Done predicting numbers")
        
    def aggregate_str_results(self):
        """
        Aggregates per-tracklet STR results into one global file (str_results.json)
        located under self.common_processed_data_dir.
        Uses multithreading to speed up file IO.
        """
        self.aggregate_legibility_results_data()
        self.set_legibility_arrays()
        # If use_cache is True and the global STR results file already exists, skip
        if self.use_cache and os.path.exists(self.str_global_result_file):
            self.logger.info(f"Reading STR results from cache: {self.str_global_result_file}")
            with open(self.str_global_result_file, 'r') as f_global:
                self.loaded_str_results = json.load(f_global)
            return  # Skip re-aggregation altogether

        self.logger.info("Aggregating STR results (cache not used or global file missing).")
        # Initialize an empty dictionary to hold merged data
        aggregated_results = {}

        # 2) Load & merge results in parallel
        def load_str_file_for_tracklet(tracklet):
            """
            Loads the per-tracklet STR results JSON and returns a dict.
            Each file is expected to have structure like:
                {
                    "1064_226.jpg": {"label": "29", "confidence": [...], "raw": [...], "logits": [...]},
                    ...
                }
            """
            processed_tracklet_dir = os.path.join(self.output_processed_data_path, tracklet)
            str_result_path = os.path.join(processed_tracklet_dir, self.results_file_name)  
            # or whatever your STR file is named

            # If the file doesn't exist or is empty, return an empty dict
            if not os.path.exists(str_result_path):
                self.logger.warning(f"No STR results file for tracklet {tracklet} at {str_result_path}")
                return {}

            try:
                with open(str_result_path, 'r') as f_local:
                    data = json.load(f_local)
                return data
            except Exception as e:
                self.logger.error(f"Error reading {str_result_path}: {e}")
                return {}

        # Use a ThreadPoolExecutor to read files in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=self.num_workers * self.num_threads_multiplier) as executor:
            for tracklet in tqdm(self.legible_tracklets_list, desc="Dispatching STR file loads", leave=True):
                futures[executor.submit(load_str_file_for_tracklet, tracklet)] = tracklet

            # Merge each tracklet’s data as it completes
            for future in tqdm(as_completed(futures), total=len(futures), desc="Merging STR results", leave=True):
                tracklet = futures[future]
                try:
                    loaded_data = future.result()  # dict from that tracklet
                    self.logger.info(f"Raw loaded data for tracklet {tracklet}: {loaded_data}")
                    # Merge the per-tracklet dictionary into the global one.
                    for image_filename, image_data in loaded_data.items():
                        aggregated_results[image_filename] = image_data
                except Exception as e:
                    self.logger.error(f"Error merging data for tracklet {tracklet}: {e}")

        # 3) Write the aggregated STR results to the global file
        with open(self.str_global_result_file, 'w') as f_global:
            json.dump(aggregated_results, f_global, indent=4)

        self.logger.info(f"Saved global STR results to: {self.str_global_result_file}")
        # Optionally store them in self.loaded_str_results
        self.loaded_str_results = aggregated_results

    def run_clip4str_model(self):
        """
        Runs the CLIP4STR model for scene text recognition in parallel, per tracklet.
        Each tracklet will have its own crops directory and result file.
        """
        self.logger.info("Predicting numbers using parallel CLIP4STR model")
        self.aggregate_legibility_results_data()
        self.set_legibility_arrays()

        # Ensure correct working directory
        os.chdir(str(Path.cwd().parent.parent))
        print("Current working directory:", os.getcwd())

        # Build references to the CLIP4STR code and environment
        clip4str_dir = os.path.join(os.getcwd(), "str", "CLIP4STR")
        model_path = DataPaths.ENHANCED_STR_MAIN.value
        clip_pretrained = DataPaths.ENHANCED_STR_OPEN_CLIP.value
        read_script_path = os.path.join(clip4str_dir, "read.py")

        # Path to Python executable (in your clip4str_env)
        #python_exe = os.path.join(os.path.expanduser("~"), "miniconda3", "envs", config.clip4str_env, "python.exe")
        python_exe = os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "clip4str_py39", "python.exe")

        # Environment variables for the subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Configure how many parallel workers and batch size
        num_parallel_workers = self.num_workers
        batch_size = self.image_batch_size

        # Optional semaphore to limit GPU usage
        self.gpu_semaphore = threading.Semaphore(value=2)

        def run_clip4str_for_tracklet(tracklet):
            """
            Run CLIP4STR inference on a single tracklet's crops directory.
            Skips if a cached result file is found (if self.use_cache is True).
            """
            processed_data_path = os.path.join(self.output_processed_data_path, tracklet)
            crops_dir = os.path.join(processed_data_path, config.dataset['SoccerNet']['crops_folder'])
            #result_file = os.path.join(processed_data_path, config.dataset['SoccerNet']['str_results_file'])
            result_file = os.path.join(processed_data_path, self.results_file_name)

            # If caching is enabled and the result file already exists, skip running the command.
            if self.use_cache and os.path.exists(result_file):
                self.logger.info(f"Skipping tracklet {tracklet} (cache found at {result_file}).")
                return tracklet, f"Cached: {result_file}"

            # Run CLIP4STR inference for this tracklet's crops directory
            try:
                success = clip4str_module.run_clip4str_inference(
                    python_path=python_exe,
                    read_script_path=read_script_path,
                    model_path=model_path,
                    clip_pretrained_path=clip_pretrained,
                    images_dir=crops_dir,
                    result_file=result_file,
                    logger=self.logger,
                    #num_workers=num_parallel_workers,
                    #batch_size=batch_size,
                    env=env,
                    #gpu_semaphore=self.gpu_semaphore
                )
                return tracklet, success
            except Exception as e:
                self.logger.error(f"Error processing tracklet {tracklet}: {e}")
                return tracklet, None

        # Use a ThreadPoolExecutor to process each tracklet in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for tracklet in tqdm(self.legible_tracklets_list, desc="Dispatching CLIP4STR on tracklets", leave=True):
                futures[executor.submit(run_clip4str_for_tracklet, tracklet)] = tracklet

            # Aggregate the results as they complete
            for future in tqdm(as_completed(futures), total=len(futures),
                            desc="Aggregating CLIP4STR results", leave=True):
                tracklet = futures[future]
                try:
                    tracklet, output = future.result()
                    # Optionally store or log the output
                    if output is None:
                        self.logger.error(f"Tracklet {tracklet} failed or returned no output.")
                    else:
                        self.logger.info(f"Tracklet {tracklet} completed: {output}")
                except Exception as e:
                    self.logger.error(f"Error processing tracklet {tracklet}: {e}")

        self.logger.info("Done predicting numbers with parallel CLIP4STR")

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

        # Initialize accumulators and lists for failed cases.
        total_correct = 0
        total_tracks = 0
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_TN = 0
        all_false_positive = []  # Tracklets that are illegible in ground truth but predicted as legible.
        all_false_negative = []  # Tracklets that are legible in ground truth but predicted as illegible.

        # Read the ground truth file once (shared across workers)
        with open(self.gt_data_path, 'r') as gf:
            gt_dict = json.load(gf)

        # If a soccer ball list is to be used, load it once.
        balls_list = []
        if load_soccer_ball_list:
            # TODO: Change from load_soccer_ball_list to actual path.
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
                return {
                    "correct": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": 0,
                    "FP_tracks": [], "FN_tracks": []
                }

            # Get the ground truth value and determine predicted legibility.
            true_value = str(gt_dict[track])
            predicted_legible = self.is_track_legible(track, illegible_list, legible_tracklets)

            # Initialize per-track statistics and lists for failed cases.
            stats = {
                "correct": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": 1,
                "FP_tracks": [], "FN_tracks": []
            }

            # Evaluate and record misclassifications:
            # Ground truth == '-1' means track should be illegible.
            if true_value == '-1' and not predicted_legible:
                stats["correct"] = 1
                stats["TN"] = 1
            # Ground truth != '-1' means track should be legible.
            elif true_value != '-1' and predicted_legible:
                stats["correct"] = 1
                stats["TP"] = 1
            # Misclassification: track is illegible in GT but predicted as legible.
            elif true_value == '-1' and predicted_legible:
                stats["FP"] = 1
                stats["FP_tracks"].append(track)
            # Misclassification: track is legible in GT but predicted as illegible.
            elif true_value != '-1' and not predicted_legible:
                stats["FN"] = 1
                stats["FN_tracks"].append(track)

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
                    all_false_positive.extend(result.get("FP_tracks", []))
                    all_false_negative.extend(result.get("FN_tracks", []))
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

        # Save the failed cases (misclassified tracklets) as a JSON file.
        failed_cases = {
            "legible_but_marked_illegible": all_false_negative,  # Should be legible but predicted illegible.
            "illegible_but_marked_legible": all_false_positive   # Should be illegible but predicted legible.
        }
        failed_cases_file = os.path.join(self.common_processed_data_dir, 'failed_legibility_cases.json')
        with open(failed_cases_file, 'w') as f:
            json.dump(failed_cases, f)
        self.logger.info(f"Saved failed legibility cases to: {failed_cases_file}")
        
    def consolidated_results(self, results_dict, illegible_path, soccer_ball_list=None):
        # If a soccer ball list is provided, update predictions for those tracks.
        if soccer_ball_list is not None and soccer_ball_list != []:
            self.logger.info("Consolidating results: Using soccer ball list")
            global_ball_tracks_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['soccer_ball_list'])
            with open(global_ball_tracks_path, 'r') as sf:
                balls_json = json.load(sf)
            balls_list = balls_json['ball_tracks']
            for entry in balls_list:
                results_dict[str(entry)] = 1

        # Update predictions for illegible tracks.
        with open(illegible_path, 'r') as f:
            illegible_dict = json.load(f)
        all_illegible = illegible_dict['illegible']
        for entry in all_illegible:
            if str(entry) not in results_dict:
                results_dict[str(entry)] = -1

        # Mark illegible tracklets with a -1
        # For every tracklet in self.tracklets_to_process (this includes legible and non-legible: whole tracklet universe),
        # if the current tracklet is not in the results_dict, set it to -1
        self.logger.info(f"consolidated_results results_dict: {results_dict}")
        for tracklet in self.tracklets_to_process:
            if tracklet not in results_dict:
                results_dict[tracklet] = -1
            else:
                # If already present, force integer conversion (if needed).
                results_dict[tracklet] = int(results_dict[tracklet])
        return results_dict
        
    def combine_results(self):
        self.aggregate_legibility_results_data()
        self.set_legibility_arrays()
        
        # 8. Combine tracklet results
        # Process global STR predictions (results_dict) and get analysis results.
        results_dict, self.analysis_results = helpers.process_jersey_id_predictions(self.str_global_result_file, useBias=True)
        illegible_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['illegible_result'])
        
        self.logger.info(f"Results dict: {results_dict}")
        self.logger.info(f"Analysis results: {self.analysis_results}")
        
        # Set the soccer ball tracks if applicable.
        self.set_ball_tracks()
        self.consolidated_dict = self.consolidated_results(results_dict, illegible_path, soccer_ball_list=self.loaded_ball_tracks)
        
        # Save final results as JSON.
        final_results_path = os.path.join(self.common_processed_data_dir, config.dataset['SoccerNet']['final_result'])
        with open(final_results_path, 'w') as f:
            json.dump(self.consolidated_dict, f, indent=4)
            
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
                #"main_subject_gauss_th=0.97_r=1.json",
                #"soccer_ball.json"
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
                      pyscript=True):
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
                batch_tracklets_to_process = list(data_dict.keys())

                # Initialize files that rely on self.tracklets_to_process
                self.init_soccer_ball_filter_data_file()
                #self.init_legibility_classifier_data_file()

                # Phase 1: Process each tracklet in parallel for this batch
                tasks = []
                #self.logger.info(f"DEBUG data dict: {batch_tracklets_to_process}")
                for tracklet in batch_tracklets_to_process:
                    #self.logger.info(f"DEBUG tracklet to int: {int(tracklet)}")
                    images = data_dict[tracklet]
                    args = (
                        tracklet,
                        images,
                        self.output_processed_data_path,
                        self.use_cache,
                        self.input_data_path,
                        batch_tracklets_to_process,
                        self.common_processed_data_dir,
                        run_soccer_ball_filter,
                        generate_features,
                        run_filter,
                        run_legible,
                        self.display_transformed_image_sample,
                        self.suppress_logging,
                        self.num_images_per_tracklet,
                        self.image_batch_size,
                        #self.GPU_SEMAPHORE
                    )
                    tasks.append(args)

                # If no tasks remain after skipping, move on
                if not tasks:
                    self.logger.info("Should not be here. No tasks to process.")
                    continue

                # RECOMMENDED MULTIPLIER: 3-5x number of workers.
                # e.g. if you have a 14 core CPU and you find 6 cores stable for ProcessPool,
                # you would do 6*3 or 6*5 as input to this ThreadPool
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
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
            self.run_pose_estimation_model(pyscript=pyscript)
            self.aggregate_pose()
        if run_crops:
            self.run_crops_model()
            
        if use_clip4str:
            self.results_file_name = config.dataset['SoccerNet']['clpip4str_results_file']
        else:
            self.results_file_name = config.dataset['SoccerNet']['str_results_file']
            
        self.str_global_result_file = os.path.join(
            self.common_processed_data_dir,
            self.results_file_name  # e.g., "str_results.json"
        )
            
        if run_str:
            if use_clip4str:
                self.logger.info("Using CLIP4STR model for scene text recognition")
                self.run_clip4str_model()  # Use our new method
            else:
                self.logger.info("Using PARSEQ2 model for scene text recognition")
                self.run_str_model()  # Use the original method
            self.aggregate_str_results()
        if run_combine:
            self.combine_results()
        if run_eval:
            self.evaluate_end_results()