import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import sys

# Add the parent path to the sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Import the config after adding the parent path
from configuration import dataset as config
from DataProcessing.Logger import CustomLogger
from DataProcessing.DataPreProcessing import CommonConstants, DataPaths

def filter_outliers(
    current_tracklet_images_input_dir: DataPaths,
    current_tracklet_processed_data_dir: DataPaths,
    common_processed_data_dir: DataPaths,
    threshold: float=config['SoccerNet']['gauss_filtered']['th'],
    rounds: int=config['SoccerNet']['gauss_filtered']['r'],
    suppress_logging: bool=False,
    use_cache: bool=True
    ):
    # We output 3 results jsons
    # BEFORE: \global_feature_dir => We output 3 results jsons
        # \main_subject_gaus_th=3.5_r=1: {0: [], 1: [], ..., n: []}
        # \main_subject_gaus_th=3.5_r=2: {0: [], 1: [], ..., n: []}
        # \main_subject_gaus_th=3.5_r=3: {0: [], 1: [], ..., n: []}
    # AFTER: \tracklet_processed_data_path (i.e. \0, \1, \2, ..., \n) => We output 1 result JSON
        # \main_subject_gaus_th=3.5: {1: 1_or_0, 2: 1_or_0, 3: 1_or_0} corresponding to r=1, r=2, r=3
        # -> Then we access this main_subject_gaus_th=3.5 and index into which r we want.
        
    # This script is designed to be run on the features.npy data for a single tracklet only, controlled via tracklet_processed_data_path
    # We extract that features.npy data from the tracklet_processed_data_path.
    # This features.npy has data for every single image in that tracklet.
    # We do the outlier filtering by loading this features.npy into memory and the algo runs on all images for this single tracklet.
    # Then, we output a 1 if we should keep this tracklet, or 0 if we should exclude it after looking through all the images and doing 3 passes.
    # We have two choices:
    #  A. output a single main_subject_gaus_th=3.5 script and inside have 0..3: [array of image numbers indicating they should be included at this pass]
    #   e.g: \main_subject_gaus_th=3.5:
    #    1: {1, 5, 8, 9, 10}
    #    2: {1, 5, 8}
    #    3: {1, 5}
    #  B. output 3 main_subject_gaus_th=3.5_r={r+1} scripts and inside have just a single key corresponding to the r value, and value is array of imgs to keep.
    #   e.g.
    #   \main_subject_gaus_th=3.5_r=1 => 1: [1, 5, 8, 9, 10]
    #   \main_subject_gaus_th=3.5_r=2 => 2: [1, 5, 8]
    #   \main_subject_gaus_th=3.5_r=3 => 3: [1, 5]
    # Option A is cleaner and makes more sense.
    # Updates to be made to the original script to make it work for the new use-case where looping over tracks happens outside:
    # 1. Intake a current_tracklet_images_input_dir
    # 2. Intake a current_tracklet_processed_data_dir
    # 3. Delete tracks = [] line.
    # 4. Adjust loop: for r in range(rounds): results[r] = []
    # 5. Remove loop: for tr in tqdm(tracks)
    # 6. Inside that loop, do all_files = os.listdir(current_tracklet_images_input_dir)
    # 7. Continue with adjustments necessary to account for the fact taht tracklet for loop is happening outside this script.
    
    # Inside ImageBatchPipeline, retrieve the gaussian outliers JSON for that tracklet
    # Then parse the 'gauss_filtered' value from the configuration file, access the r to extract
    # Then index into that gauss_filtered json so that we do images = gauss_filtered[r]. Problem solved.
    
    results = {}
    
    for r in range(rounds):
        results[r] = [] # This this is the keep list for a single tracklet for the round in consideration

    # Get the files for the current tracklet
    all_files = os.listdir(current_tracklet_images_input_dir)
    
    # Filter out hidden files in the tracklet directory
    images = [img for img in all_files if not img.startswith('.')]
    
    # Get the features path for the current tracklet
    feature_file_path = os.path.join(current_tracklet_processed_data_dir, CommonConstants.FEATURE_DATA_FILE_NAME.value)

    with open(feature_file_path, 'rb') as f:
        features = np.load(f)
    if len(images) <= 2:
        # Too few images to do pruning on. Just include all of them.
        # This means indexing into every r of the results object and including all images of the current tracklet across all rounds/
        for r in range(rounds):
            results[r] = images
            
        # And do not run the pruning algo. Just return here, because the outer loop controls the next tracklet.
        return results

    # If we reached this point, we have enough data to run pruning on.
    cleaned_data = features
    for r in range(rounds):
        # Fit a Gaussian distribution to the data
        mu = np.mean(cleaned_data, axis=0)

        euclidean_distance = np.linalg.norm(features - mu, axis = 1)

        mean_euclidean_distance = np.mean(euclidean_distance)
        std = np.std(euclidean_distance)
        th = threshold * std

        # Remove outliers from the data
        cleaned_data = features[(euclidean_distance - mean_euclidean_distance) <= threshold]
        cleaned_data_indexes = np.where((euclidean_distance - mean_euclidean_distance)<= threshold)[0]

        for i in cleaned_data_indexes:
            # Access the value array for this round (implicitly for this tracklet as this file is for this tracklet and only this tracklet)
            results[r].append(images[i])

    # For every round, open the appropriate file, access the tracklet, append the keep list, and close the file.
    # Before appending, check if it is empty. If not and user has supplied use_cache=False, overwrite it. Otherwise, skip.
    for r in range(rounds):
        result_file_name = f"main_subject_gauss_th={threshold}_r={r + 1}.json"
        result_file_path = os.path.join(common_processed_data_dir, result_file_name)
        
        # Check if the file exists. If not, raise an error because it should have been created in CentralPipeline
        if not os.path.exists(result_file_path):
            raise FileNotFoundError(f"File not found: {result_file_path}. This should have been created in CentralPipeline.")
        
        with open(result_file_path, 'w') as f:
            data = json.load(f)
            data_for_this_tracklet = data[tracklet] #= results[r]
            
            # Do not write anything
            if use_cache and len(data_for_this_tracklet) > 0:
                continue
            else:
                # Write the data
                data[tracklet] = results[r]
                json.dump(data, f)
                
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_tracklet_images_input_dir', help="Path to the raw images for the current tracklet", required=True)
    parser.add_argument('--current_tracklet_processed_data_dir', help="Path to the processed output data dir for the current tracklet", required=True)
    parser.add_argument('--common_processed_data_dir', help="Path to the shared processed data output for the test/train/challenge data", required=True)
    parser.add_argument('--threshold', type=float, default=3.5, help="Offset threshold for (distance - mean_dist)")
    parser.add_argument('--rounds', type=int, default=3, help="Number of rounds for iterative outlier filtering")
    parser.add_argument('--suppress_logging', action='store_true', help="Suppress logging output")
    parser.add_argument('--use_cache', action='store_true', help="Flag to know if we should rebuild the cache or use it")
    args = parser.parse_args()
    
    filter_outliers(
        current_tracklet_images_input_dir=args.current_tracklet_images_input_dir,
        current_tracklet_processed_data_dir=args.current_tracklet_processed_data_dir,
        common_processed_data_dir=args.common_processed_data_dir,
        threshold=args.threshold,
        rounds=args.rounds,
        suppress_logging=args.suppress_logging,
        use_cache=args.use_cache)