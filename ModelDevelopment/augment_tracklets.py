import os
import sys
import json
import time
import random
import torch
import cv2
from PIL import Image
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now import project modules
from DataProcessing.DataAugmentation import DataAugmentation


# Function for augmenting a single image and saving to output directory
def augment_single_image(img_path, augmenter, output_dir):
    """Augment a single image and save results to output directory"""
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return False

        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Convert to tensor
        to_tensor = lambda x: torch.from_numpy(np.array(x).transpose(2, 0, 1) / 255.0).float()
        img_tensor = to_tensor(img_pil)

        # Apply augmentations
        aug_versions = augmenter.augment_image(img_tensor, augmenter.allowed_transformations)

        # Save augmented versions
        basename = os.path.basename(img_path)
        name, ext = os.path.splitext(basename)

        to_pil = lambda x: Image.fromarray((x.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        for aug_type, aug_tensor in aug_versions.items():
            if aug_type == 'original':
                continue

            # Save the augmented image
            aug_img = to_pil(aug_tensor)
            save_path = os.path.join(output_dir, f"{name}_{aug_type}{ext}")
            aug_img.save(save_path)

        return True
    except Exception as e:
        print(f"Error augmenting {img_path}: {e}")
        return False


# Process one tracklet directory in parallel
def process_tracklet(args):
    tracklet_dir, tracklet_id, jersey_num = args

    try:
        # Initialize augmenter with dummy data
        dummy_tensor = torch.zeros((1, 3, 256, 256))
        dummy_data = {'dummy': dummy_tensor}
        augmenter = DataAugmentation(dummy_data)

        # Get all images in this tracklet
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(tracklet_dir, ext)))

        if not image_paths:
            print(f"No images found in tracklet {tracklet_id}")
            return None, tracklet_id, jersey_num

        # Create augmented_data directory
        augmented_dir = os.path.join(tracklet_dir, 'augmented_data')
        os.makedirs(augmented_dir, exist_ok=True)

        # Select 50% of images to augment
        num_to_augment = max(1, int(len(image_paths) * 0.5))
        images_to_augment = random.sample(image_paths, num_to_augment)

        # Process each selected image
        successful = 0
        for img_path in images_to_augment:
            if augment_single_image(img_path, augmenter, augmented_dir):
                successful += 1

        if successful > 0:
            return augmented_dir, tracklet_id, jersey_num
        else:
            return None, tracklet_id, jersey_num
    except Exception as e:
        print(f"Error processing tracklet {tracklet_id}: {e}")
        return None, tracklet_id, jersey_num


def augment_tracklets(num_workers=5, output_mapping_path=None):
    """
    Augment tracklet images and create a mapping of directories to jersey numbers

    Args:
        num_workers: Number of parallel workers
        output_mapping_path: Custom path to save the mapping JSON (optional)

    Returns:
        augmented_mapping: Dictionary mapping augmented directories to jersey numbers
        mapping_path: Path where the mapping was saved
    """
    start_time = time.time()
    print("Starting tracklet augmentation...")

    # Get data paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data_dir = os.path.join(project_root, "data", "SoccerNet", "jersey-2023", "extracted", "train", "images")
    train_gt_json = os.path.join(project_root, "data", "SoccerNet", "jersey-2023", "extracted", "train",
                                 "train_gt.json")

    print(f"Train data directory: {train_data_dir}")
    print(f"Train GT JSON: {train_gt_json}")

    # Verify paths exist
    if not os.path.exists(train_data_dir):
        print(f"ERROR: Train data directory not found: {train_data_dir}")
        return {}, None

    if not os.path.exists(train_gt_json):
        print(f"ERROR: Train GT JSON not found: {train_gt_json}")
        return {}, None

    # Load jersey mapping
    with open(train_gt_json, 'r') as f:
        jersey_mapping = json.load(f)

    # Convert to integers, filtering out -1 values
    valid_mapping = {}
    for k, v in jersey_mapping.items():
        if v != -1:  # Skip illegible jerseys
            valid_mapping[int(k)] = int(v)

    print(f"Found {len(valid_mapping)} tracklets with valid jersey numbers")

    # Create list of tracklets to process
    tracklet_tasks = []
    for tracklet_id, jersey_num in valid_mapping.items():
        tracklet_dir = os.path.join(train_data_dir, str(tracklet_id))
        if os.path.isdir(tracklet_dir):
            tracklet_tasks.append((tracklet_dir, tracklet_id, jersey_num))

    print(f"Processing {len(tracklet_tasks)} tracklet directories")

    # Process tracklets in parallel
    print(f"Using {num_workers} CPU workers for parallel processing")

    # Create a mapping for training
    augmented_mapping = {}

    # Process tracklets
    if num_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_tracklet, tracklet_tasks),
                total=len(tracklet_tasks),
                desc="Augmenting tracklets"
            ))

            for aug_dir, tracklet_id, jersey_num in results:
                if aug_dir:
                    augmented_mapping[aug_dir] = jersey_num
    else:
        # Sequential processing (for troubleshooting)
        for task in tqdm(tracklet_tasks, desc="Augmenting tracklets"):
            aug_dir, tracklet_id, jersey_num = process_tracklet(task)
            if aug_dir:
                augmented_mapping[aug_dir] = jersey_num

    print(f"Successfully augmented {len(augmented_mapping)} tracklets")

    # Save the mapping for reference
    if output_mapping_path:
        mapping_path = output_mapping_path
    else:
        output_dir = os.path.join(project_root, "ModelDevelopment", "output")
        os.makedirs(output_dir, exist_ok=True)
        mapping_path = os.path.join(output_dir, "augmented_tracklets.json")

    with open(mapping_path, 'w') as f:
        # Convert paths to strings for JSON
        serializable = {str(k): v for k, v in augmented_mapping.items()}
        json.dump(serializable, f, indent=2)

    print(f"Saved augmented mapping to {mapping_path}")
    print(f"Augmentation time: {time.time() - start_time:.2f} seconds")

    return augmented_mapping, mapping_path


if __name__ == "__main__":
    # Run augmentation with default parameters
    augment_tracklets()