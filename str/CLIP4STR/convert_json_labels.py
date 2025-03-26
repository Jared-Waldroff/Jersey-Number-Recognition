import json
import os
import glob
import random
import shutil

# Load the JSON with jersey labels
with open('test_gt.json', 'r') as f:
    jersey_labels = json.load(f)

# Set up directories
image_dir = 'dataset/soccernet/train/images'
output_dir = 'dataset/soccernet'

# Ensure output directories exist
os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)

# Find all image files in your directory
all_images = []
for prefix in jersey_labels.keys():
    pattern = os.path.join(image_dir, f"{prefix}_*.jpg")
    matching_files = glob.glob(pattern)
    all_images.extend(matching_files)

print(f"Found {len(all_images)} total images")

# Create label entries (exclude -1 values which indicate no visible number)
valid_entries = []
for img_path in all_images:
    base_name = os.path.basename(img_path)
    prefix = base_name.split('_')[0]
    jersey_num = jersey_labels[prefix]

    # Skip images with -1 label (not legible)
    if jersey_num != -1:
        relative_path = os.path.basename(img_path)
        valid_entries.append((relative_path, img_path, jersey_num))

print(f"Found {len(valid_entries)} valid images with jersey numbers")

# Split into train (80%) and validation (20%)
random.shuffle(valid_entries)
split_idx = int(len(valid_entries) * 0.8)
train_entries = valid_entries[:split_idx]
val_entries = valid_entries[split_idx:]

print(f"Split into {len(train_entries)} training and {len(val_entries)} validation images")

# Write train labels and copy images
train_labels = []
for filename, src_path, jersey_num in train_entries:
    train_labels.append(f"{filename} {jersey_num}")
    dst_path = os.path.join(output_dir, 'train', 'images', filename)

    if os.path.exists(dst_path):
        # Skip if file already exists
        continue

    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"Error copying {src_path} to {dst_path}: {e}")

with open(os.path.join(output_dir, 'train', 'labels.txt'), 'w') as f:
    f.write('\n'.join(train_labels))

# Write val labels and copy images
val_labels = []
for filename, src_path, jersey_num in val_entries:
    val_labels.append(f"{filename} {jersey_num}")
    dst_path = os.path.join(output_dir, 'val', 'images', filename)

    if os.path.exists(dst_path):
        # Skip if file already exists
        continue

    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"Error copying {src_path} to {dst_path}: {e}")

with open(os.path.join(output_dir, 'val', 'labels.txt'), 'w') as f:
    f.write('\n'.join(val_labels))

print("Dataset preparation complete!")