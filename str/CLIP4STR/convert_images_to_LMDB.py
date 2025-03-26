import os
import lmdb
import json
import glob
import random
from PIL import Image
import io
import numpy as np
from tqdm import tqdm

# Make map size of 5GB
map_size = 5 * 1024 * 1024 * 1024


def create_lmdb_dataset(image_dir, json_file, output_dir, max_percent=0.1, map_size=map_size):
    """
    Convert a directory of images and a JSON label file to LMDB format.

    Args:
        image_dir: Directory containing jersey images
        json_file: Path to the JSON file with labels
        output_dir: Output directory for LMDB
        max_percent: Maximum percent of images to use (0.1 = 10%)
        map_size: Size of the LMDB map
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load labels
    with open(json_file, 'r') as f:
        label_dict = json.load(f)

    # Find all image files
    image_files = []
    valid_labels = []

    # Get all image files that have valid labels
    for prefix in label_dict.keys():
        jersey_num = label_dict[prefix]
        if jersey_num != -1:  # Skip images with no label
            # Find all images with this prefix
            pattern = os.path.join(image_dir, f"{prefix}_*.jpg")
            matching_files = glob.glob(pattern)

            for img_path in matching_files:
                image_files.append(img_path)
                valid_labels.append(str(jersey_num))  # Convert to string

    total_images = len(image_files)
    print(f"Found {total_images} images in {image_dir}")

    # Limit to max_percent of images
    max_images = int(total_images * max_percent)
    if total_images > max_images:
        print(f"Limiting to {max_images} images (10% of total)")
        # Get a random sample
        indices = random.sample(range(total_images), max_images)
        image_files = [image_files[i] for i in indices]
        valid_labels = [valid_labels[i] for i in indices]

    print(f"Processing {len(image_files)} images for LMDB")

    # Create LMDB
    env = lmdb.open(output_dir, map_size=map_size)

    with env.begin(write=True) as txn:
        # Write data
        for idx, (img_path, label) in enumerate(tqdm(zip(image_files, valid_labels), total=len(image_files))):
            # Index starts at 1
            index = idx + 1

            # Read and convert image
            img = Image.open(img_path).convert('RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            # Write image
            txn.put(f'image-{index:09d}'.encode(), img_bytes)
            # Write label
            txn.put(f'label-{index:09d}'.encode(), label.encode())

        # Write total count
        txn.put('num-samples'.encode(), str(len(image_files)).encode())

    env.close()
    print(f"Created LMDB dataset with {len(image_files)} images at {output_dir}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Paths to your already split image directories
    train_image_dir = "C:/Users/jared/PycharmProjects/CLIP4STR/dataset/soccernet/train/images"
    val_image_dir = "C:/Users/jared/PycharmProjects/CLIP4STR/dataset/soccernet/val/images"
    json_file = "C:/Users/jared/PycharmProjects/CLIP4STR/test_gt.json"

    # Output locations
    train_output = "C:/Users/jared/PycharmProjects/CLIP4STR/dataset/soccernet/train"
    val_output = "C:/Users/jared/PycharmProjects/CLIP4STR/dataset/soccernet/val"

    # Process each directory separately
    print("Creating TRAIN LMDB...")
    create_lmdb_dataset(train_image_dir, json_file, train_output, max_percent=0.1)

    print("Creating VAL LMDB...")
    create_lmdb_dataset(val_image_dir, json_file, val_output, max_percent=0.1)