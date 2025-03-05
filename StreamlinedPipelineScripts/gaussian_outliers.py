import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from DataProcessing.Logger import CustomLogger

def filter_outliers(feature_file, threshold=3.5, rounds=3):
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
        feature_file (str): Path to the .npy feature file (features of shape (N, d)).
        threshold (float): The raw offset threshold for (distance - mean).
        rounds (int): Number of iterations for outlier filtering.
        
    Returns:
        dict: A dictionary mapping round number (0-indexed) -> list of inlier indices (from the original feature matrix).
    """
    logger = CustomLogger().get_logger()
    logger.info(f"Loading features from {feature_file}")
    features = np.load(feature_file, allow_pickle=True)
    N = features.shape[0]
    original_features = features.copy()  # Unmodified copy for distance calculations
    indices = np.arange(N)               # Original indices for the feature rows
    
    # Start with the full set as "cleaned_data"
    cleaned_data = features.copy()
    cleaned_indices = indices.copy()
    
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
    
    out_json = os.path.splitext(feature_file)[0] + f"_gauss_th={threshold}_r={rounds}.json"
    with open(out_json, "w") as f:
        json.dump(results, f)
    logger.info(f"Outlier filtering complete. Results saved to {out_json}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_file', help="Path to the .npy feature file for a tracklet", required=True)
    parser.add_argument('--threshold', type=float, default=3.5, help="Offset threshold for (distance - mean_dist)")
    parser.add_argument('--rounds', type=int, default=3, help="Number of rounds for iterative outlier filtering")
    args = parser.parse_args()
    
    filter_outliers(args.feature_file, threshold=args.threshold, rounds=args.rounds)