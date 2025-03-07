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
from DataProcessing.DataPreProcessing import CommonConstants

def filter_outliers(tracklet_processed_data_path, threshold: float=3.5, rounds: int=3, suppress_logging: bool=False, use_cache: bool=True):
    """
    Process a single feature file (a .npy file) containing feature vectors for a tracklet.
    Iteratively filters out outliers based on the Euclidean distance from the mean of
    the 'cleaned_data' at each round, but uses the original features for computing distances.
    
    This replicates the original code's logic:
      cleaned_data = features
      for r in range(rounds):
          mu = mean(cleaned_data)
          dist = norm(features - mu)
          mean_dist = mean(dist)
          std = std(dist)
          # Filtering: (dist - mean_dist) <= threshold
          # (threshold * std is computed but not used)
          # Then update cleaned_data = features[inlier_mask]
          # and record the inlier indices.
    
    Args:
        tracklet_processed_data_path (str): Path to the .npy feature file (features of shape (N, d)).
        threshold (float): The raw offset threshold for (distance - mean).
        rounds (int): Number of iterations for outlier filtering.
        
    Returns:
        dict: A dictionary mapping round number (0-indexed) -> list of inlier indices (from the original feature matrix).
    """
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
    #   e.g:
    #    0: {1, 5, 8, 9, 10}
    #    1: {1, 5, 8}
    #    2: {1, 5}
    #  B. output 3 main_subject_gaus_th=3.5_r={r+1} scripts and inside have just a single key corresponding to the r value, and value is array of imgs to keep.
    
    logger = CustomLogger(suppress_logging=suppress_logging).get_logger()
    feature_file = os.path.join(tracklet_processed_data_path, CommonConstants.FEATURE_DATA_FILE_NAME.value)
    logger.info(f"Loading features from {feature_file}")
    features = np.load(feature_file, allow_pickle=True)
    N = features.shape[0]
    original_features = features.copy()  # Unmodified copy for distance calculations
    indices = np.arange(N)               # Original indices for the feature rows
    
    # Start with the full set as "cleaned_data"
    cleaned_data = features.copy()
    cleaned_indices = indices.copy()
    
    # NOTE: The file name is being hardcoded to be the one from the config.
    # The reason for this is that these params are dynamic, and should be updated in the config as well.
    # We need a deterministic way to know the file name as if it is purely dynamic, other scripts won't know what it is called.
    out_json = os.path.join(tracklet_processed_data_path, config['SoccerNet']['gauss_filtered'])

    if not use_cache:
        if os.path.exists(out_json):
            try:
                os.remove(out_json)
                logger.info(f"Removed cached json data file (use_cache: False): {out_json}")
            except Exception as e:
                logger.warning(f"Failed to remove cached json data file: {out_json}. Error: {e}")

    results = {}
    for r in range(rounds):
        mu = np.mean(cleaned_data, axis=0)
        euclidean_distance = np.linalg.norm(original_features - mu, axis=1)
        mean_euclidean_distance = np.mean(euclidean_distance)
        std = np.std(euclidean_distance)
        th_val = threshold * std  # Calculated but not used in filtering
        
        # Log current round details
        logger.info(f"Round {r+1}: mean_euclidean_distance = {mean_euclidean_distance:.4f}, std = {std:.4f}, threshold = {threshold}")
        
        inlier_mask = (euclidean_distance - mean_euclidean_distance) <= threshold
        inlier_indices = cleaned_indices[inlier_mask]
        results[r] = inlier_indices.tolist()
        
        logger.info(f"Round {r+1}: kept {len(inlier_indices)} out of {len(cleaned_indices)} features")
        
        # Update for next round
        cleaned_data = original_features[inlier_mask]
        cleaned_indices = cleaned_indices[inlier_mask]
    
    with open(out_json, "w") as f:
        json.dump(results, f)
    logger.info(f"Outlier filtering complete. Results saved to {out_json}")
    
    return results


    
    
# OPTION A: ONE FILE PER TRACKLET
def filter_outliers(current_tracklet_images_input_dir, current_tracklet_processed_data_dir, threshold: float=3.5, rounds: int=3, suppress_logging: bool=False, use_cache: bool=True):
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
        # 0: []
        # 1: []
        # 2: []
        results[r] = []

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

    for r in range(rounds):
        result_file_name = f"main_subject_gauss_th={threshold}_r={r + 1}.json"
        with open(os.path.join(feature_folder, result_file_name), "w") as outfile:
            json.dump(results[r], outfile)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_tracklet_images_input_dir', help="Path to the raw images for the current tracklet", required=True)
    parser.add_argument('--current_tracklet_processed_data_dir', help="Path to the processed output data dir for the current tracklet", required=True)
    parser.add_argument('--threshold', type=float, default=3.5, help="Offset threshold for (distance - mean_dist)")
    parser.add_argument('--rounds', type=int, default=3, help="Number of rounds for iterative outlier filtering")
    parser.add_argument('--suppress_logging', action='store_true', help="Suppress logging output")
    parser.add_argument('--use_cache', action='store_true', help="Flag to know if we should rebuild the cache or use it")
    args = parser.parse_args()
    
    filter_outliers(
        args.tracklet_processed_data_path,
        threshold=args.threshold,
        rounds=args.rounds,
        suppress_logging=args.suppress_logging,
        use_cache=args.use_cache)
    
    
    
# TODO: GO BACK TO THIS VERSION.
# OPTION B: MASTER FILE WITH MULTIPLE ROUNDS AND THEN ALL TRACKLETS, LIKE ORIGINAL.
# ONLY CHANGE: Just open the file, access the tracklet to which we wish to add the keep list to, then close.
def filter_outliers(current_tracklet_images_input_dir, current_tracklet_processed_data_dir, threshold: float=3.5, rounds: int=3, suppress_logging: bool=False, use_cache: bool=True):
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
        # 0: []
        # 1: []
        # 2: []
        results[r] = []

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

    for r in range(rounds):
        result_file_name = f"main_subject_gauss_th={threshold}_r={r + 1}.json"
        with open(os.path.join(feature_folder, result_file_name), "w") as outfile:
            json.dump(results[r], outfile)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_tracklet_images_input_dir', help="Path to the raw images for the current tracklet", required=True)
    parser.add_argument('--current_tracklet_processed_data_dir', help="Path to the processed output data dir for the current tracklet", required=True)
    parser.add_argument('--threshold', type=float, default=3.5, help="Offset threshold for (distance - mean_dist)")
    parser.add_argument('--rounds', type=int, default=3, help="Number of rounds for iterative outlier filtering")
    parser.add_argument('--suppress_logging', action='store_true', help="Suppress logging output")
    parser.add_argument('--use_cache', action='store_true', help="Flag to know if we should rebuild the cache or use it")
    args = parser.parse_args()
    
    filter_outliers(
        args.tracklet_processed_data_path,
        threshold=args.threshold,
        rounds=args.rounds,
        suppress_logging=args.suppress_logging,
        use_cache=args.use_cache)