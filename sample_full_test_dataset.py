#!/usr/bin/env python3
import os
import json
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Sample full test dataset for evaluation')
    parser.add_argument('--test_dir', type=str, default='Jersey-2023/test',
                        help='Path to test directory')
    parser.add_argument('--output_dir', type=str, default='sampled_evaluation_data',
                        help='Path to output directory for processed data')
    parser.add_argument('--gt_file', type=str, default='Jersey-2023/test/test_gt.json',
                        help='Path to ground truth file')
    parser.add_argument('--sample_size', type=int, default=2000,
                        help='Number of images to sample per tracklet (max)')
    parser.add_argument('--max_images_per_tracklet', type=int, default=10,
                        help='Maximum number of images to sample per tracklet')
    return parser.parse_args()

def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def sample_test_data(args):
    """Sample test data, collecting representative jersey images"""
    # Create output directory structure
    crops_dir = os.path.join(args.output_dir, 'crops')
    create_directory(args.output_dir)
    create_directory(crops_dir)
    
    # Load ground truth data
    with open(args.gt_file, 'r') as f:
        gt_data = json.load(f)
    
    # Initialize counters
    images_processed = 0
    illegible_count = 0
    legible_count = 0
    illegible_results = {}
    
    # Process each tracklet
    images_dir = os.path.join(args.test_dir, 'images')
    tracklet_dirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    print(f"Found {len(tracklet_dirs)} tracklet directories")
    
    # If sample_size is specified, randomly sample tracklet_dirs
    if args.sample_size < len(tracklet_dirs):
        random.seed(42)  # For reproducibility
        tracklet_dirs = random.sample(tracklet_dirs, args.sample_size)
        print(f"Randomly sampled {len(tracklet_dirs)} tracklets")
    
    for tracklet_id in tqdm(tracklet_dirs, desc="Processing tracklets"):
        # Get label information - jersey number is directly stored as value, -1 indicates illegible
        jersey_number = gt_data.get(tracklet_id, None)
        if jersey_number is None:
            print(f"Warning: Tracklet {tracklet_id} not found in ground truth")
            continue
            
        # Jersey number -1 indicates illegible
        is_illegible = (jersey_number == -1)
        
        # Mark in illegible results
        illegible_results[tracklet_id] = is_illegible
        
        # Count based on legibility
        if is_illegible:
            illegible_count += 1
        else:
            legible_count += 1
        
        # Process images in this tracklet
        tracklet_path = os.path.join(images_dir, tracklet_id)
        image_files = [f for f in os.listdir(tracklet_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample a subset of images from this tracklet
        if len(image_files) > args.max_images_per_tracklet:
            random.seed(int(tracklet_id) if tracklet_id.isdigit() else 42)  # For reproducibility
            image_files = random.sample(image_files, args.max_images_per_tracklet)
        
        # Copy images to output directory
        for img_file in image_files:
            source_path = os.path.join(tracklet_path, img_file)
            
            # Use tracklet_id and original filename to create a unique destination filename
            dest_filename = f"{tracklet_id}_{img_file}"
            dest_path = os.path.join(crops_dir, dest_filename)
            
            shutil.copy(source_path, dest_path)
            images_processed += 1
    
    # Save illegible results to json
    illegible_file = os.path.join(args.output_dir, 'illegible_results.json')
    with open(illegible_file, 'w') as f:
        json.dump(illegible_results, f, indent=4)
    
    # Print summary
    print(f"\nProcessed {images_processed} images from {len(tracklet_dirs)} tracklets")
    print(f"Legible jerseys: {legible_count}")
    print(f"Illegible jerseys: {illegible_count}")
    print(f"Data saved to: {args.output_dir}")
    
    return crops_dir, illegible_file

def main():
    args = parse_args()
    print("Sampling test dataset for comprehensive evaluation")
    crops_dir, illegible_file = sample_test_data(args)
    
    print("\nData sampling complete. Ready for evaluation.")
    print(f"To run evaluation, use: python comprehensive_evaluation.py --model_paths ResNetModels/baseline_ResNet_crops_model.pth ResNetModels/fpn_resnet34_epoch10_model.pth ResNetModels/squeeze_Resnet_epoch10_model.pth --model_names baseline fpn squeeze --image_dir {crops_dir} --illegible_file {illegible_file} --ground_truth_file {args.gt_file} --output_dir sampled_evaluation_results")

if __name__ == "__main__":
    main()
