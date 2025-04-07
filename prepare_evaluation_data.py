import os
import json
import shutil
import random

# Paths
JERSEY_DATASET_PATH = '/home/mahalaa/Jersey-Number-Recognition/Jersey-2023/test/images'
GROUND_TRUTH_FILE = '/home/mahalaa/Jersey-Number-Recognition/Jersey-2023/test/test_gt.json'
ILLEGIBLE_FILE = '/home/mahalaa/Jersey-Number-Recognition/evaluation_data/illegible_results.json'
OUT_DIR = '/home/mahalaa/Jersey-Number-Recognition/evaluation_data/crops/imgs'

# Load ground truth data
with open(GROUND_TRUTH_FILE, 'r') as f:
    ground_truth = json.load(f)

# Load illegible data
with open(ILLEGIBLE_FILE, 'r') as f:
    illegible_data = json.load(f)
    illegible_ids = set(illegible_data.get('illegible', []))

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Dictionary to count number of samples per class
class_count = {}

# Maximum images to collect per tracklet
MAX_IMAGES_PER_TRACKLET = 5

# Maximum number of tracklets to process
MAX_TRACKLETS = 200

# Find all tracklets
tracklet_ids = list(ground_truth.keys())
tracklet_ids = random.sample(tracklet_ids, min(len(tracklet_ids), MAX_TRACKLETS))

# Create map of tracklet_id to label
tracklet_to_label = {}
for tracklet_id, label in ground_truth.items():
    if tracklet_id in tracklet_ids:
        tracklet_to_label[tracklet_id] = label

# Count number of samples processed
total_samples = 0

print("Collecting sample jersey images for evaluation...")

# Process each tracklet
for tracklet_id in tracklet_ids:
    # Skip if tracklet folder doesn't exist
    tracklet_dir = os.path.join(JERSEY_DATASET_PATH, tracklet_id)
    if not os.path.exists(tracklet_dir):
        continue
    
    # Get true label
    true_label = ground_truth.get(tracklet_id, -1)
    
    # Initialize class count if not already
    if true_label not in class_count:
        class_count[true_label] = 0
    
    # Find all images in the tracklet folder
    images = [f for f in os.listdir(tracklet_dir) if f.endswith('.jpg')]
    images = random.sample(images, min(len(images), MAX_IMAGES_PER_TRACKLET))
    
    # Copy each image to the output directory
    for img in images:
        src_path = os.path.join(tracklet_dir, img)
        dst_filename = f"{tracklet_id}_{img}"
        dst_path = os.path.join(OUT_DIR, dst_filename)
        
        # Copy the image
        shutil.copy(src_path, dst_path)
        
        # Count the sample
        class_count[true_label] = class_count.get(true_label, 0) + 1
        total_samples += 1

# Print summary
print(f"\nEvaluation data preparation complete!")
print(f"Total samples collected: {total_samples}")
print("\nDistribution by class:")
for label, count in sorted(class_count.items()):
    print(f"Class {label}: {count} samples")
