import numpy as np
import json
import os
import argparse
from tqdm import tqdm

def filter_outliers(feature_file, threshold=3.5, rounds=3):
    """
    Process a single feature file (a .npy file) containing feature vectors for a tracklet.
    Iteratively filters out outliers based on the Euclidean distance from the mean of
    the 'cleaned_data' at each round, but uses the original features for computing distances.
    
    This faithfully replicates the original code's logic:
      cleaned_data = features
      for r in range(rounds):
          mu = mean(cleaned_data)
          dist = norm(features - mu)
          mean_dist = mean(dist)
          std = std(dist)
          # Condition used:
          #   (dist - mean_dist) <= threshold
          # 'threshold * std' is computed but never used
          # Then we filter cleaned_data = features[inlier_mask]
          # and record the inlier indices
    
    Args:
        feature_file (str): Path to the .npy feature file (features of shape (N, d)).
        threshold (float): The raw offset threshold for (distance - mean).
        rounds (int): Number of iterations for outlier filtering.
        
    Returns:
        dict: A dictionary mapping round number -> list of inlier indices (from original features).
              Round numbers are 0-based in this dict.
    """
    # Load all features (N, d)
    features = np.load(feature_file, allow_pickle=True)
    N = features.shape[0]
    original_features = features.copy()  # Keep an unmodified copy for distance calculations
    indices = np.arange(N)               # Original indices for the feature rows
    
    # Start with the entire feature set for 'cleaned_data'
    cleaned_data = features.copy()
    cleaned_indices = indices.copy()
    
    results = {}
    for r in range(rounds):
        # Fit a Gaussian distribution => basically compute mu
        mu = np.mean(cleaned_data, axis=0)
        
        # Compute Euclidean distance using the original features
        euclidean_distance = np.linalg.norm(original_features - mu, axis=1)
        
        # The original code calculates but never uses threshold * std for filtering
        mean_euclidean_distance = np.mean(euclidean_distance)
        std = np.std(euclidean_distance)
        # leftover: th_val = threshold * std  (Not actually used in the condition)
        
        # The actual filtering condition from the original script:
        inlier_mask = (euclidean_distance - mean_euclidean_distance) <= threshold
        
        # Update cleaned_data for the next round
        cleaned_data = original_features[inlier_mask]
        cleaned_indices = cleaned_indices[inlier_mask]
        
        # Store the inlier indices for this round
        results[r] = cleaned_indices.tolist()
    
    # Write the results to a JSON in the same directory as 'feature_file'
    out_json = os.path.splitext(feature_file)[0] + f"_gauss_th={threshold}_r={rounds}.json"
    with open(out_json, "w") as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_file', help="Path to the .npy feature file for a tracklet", required=True)
    parser.add_argument('--threshold', type=float, default=3.5, help="Offset threshold for (distance - mean_dist)")
    parser.add_argument('--rounds', type=int, default=3, help="Number of rounds for iterative outlier filtering")
    args = parser.parse_args()
    
    filter_outliers(args.feature_file, threshold=args.threshold, rounds=args.rounds)