import os
import glob
import json
import random
import shutil
from PIL import Image
import lmdb
import io
from tqdm import tqdm


def fix_validation_dataset(base_dir, val_pct=0.1, seed=42, replace_existing=True):
    """
    Fix the validation dataset by properly setting up labels and LMDB files

    Args:
        base_dir: Base directory containing train and val folders
        val_pct: Percentage of training images to use for validation (0-1)
        seed: Random seed for reproducibility
        replace_existing: Whether to replace existing validation images
    """
    # Set random seed
    random.seed(seed)

    # Setup directories
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    train_img_dir = os.path.join(train_dir, 'images')
    val_img_dir = os.path.join(val_dir, 'images')

    # Create val/images directory if it doesn't exist
    os.makedirs(val_img_dir, exist_ok=True)

    # Load jersey labels from JSON
    json_file = os.path.join(train_dir, 'ground_truth.json')
    with open(json_file, 'r') as f:
        jersey_labels = json.load(f)

    print(f"Loaded {len(jersey_labels)} jersey label entries from JSON")

    # If replacing existing validation, clean out the directory
    if replace_existing and os.path.exists(val_img_dir):
        existing_val_images = glob.glob(os.path.join(val_img_dir, '*.*'))
        if existing_val_images:
            print(f"Removing {len(existing_val_images)} existing validation images...")
            for img_path in existing_val_images:
                os.remove(img_path)

    # Get all image files from train/images
    train_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        train_images.extend(glob.glob(os.path.join(train_img_dir, ext)))

    print(f"Found {len(train_images)} images in training directory")

    # Build mapping from image path to label
    image_labels = {}
    for img_path in train_images:
        img_filename = os.path.basename(img_path)
        prefix = img_filename.split('_')[0]  # Assume format like "prefix_rest.jpg"

        if prefix in jersey_labels:
            jersey_num = jersey_labels[prefix]
            if jersey_num != -1:  # Skip images with no label (-1)
                image_labels[img_path] = jersey_num

    valid_images = list(image_labels.keys())
    print(f"Found {len(valid_images)} images with valid labels")

    # Calculate how many images to move
    num_val = int(len(valid_images) * val_pct)
    print(f"Moving {num_val} images to validation set ({val_pct * 100:.1f}%)")

    # Randomly select images for validation
    val_images = random.sample(valid_images, num_val)

    # Move images to validation folder
    train_label_entries = []
    val_label_entries = []

    print("Moving images to validation directory...")
    for img_path in tqdm(valid_images):
        img_filename = os.path.basename(img_path)
        label = image_labels[img_path]

        if img_path in val_images:
            # Move to validation
            dest_path = os.path.join(val_img_dir, img_filename)
            if replace_existing or not os.path.exists(dest_path):
                shutil.copy(img_path, dest_path)  # Use copy instead of move to keep training set intact
            val_label_entries.append(f"{img_filename} {label}")
        else:
            # Keep in training
            train_label_entries.append(f"{img_filename} {label}")

    # Write train labels file
    train_labels_file = os.path.join(train_dir, 'labels.txt')
    with open(train_labels_file, 'w') as f:
        f.write('\n'.join(train_label_entries))

    # Write validation labels file
    val_labels_file = os.path.join(val_dir, 'labels.txt')
    with open(val_labels_file, 'w') as f:
        f.write('\n'.join(val_label_entries))

    print(f"Updated train labels file with {len(train_label_entries)} entries")
    print(f"Created validation labels file with {len(val_label_entries)} entries")

    # Create LMDB files
    create_lmdb(train_dir)
    create_lmdb(val_dir)

    print("Validation split complete!")


def create_lmdb(data_dir, map_size=5 * 1024 * 1024 * 1024):
    """
    Create LMDB files from images and labels.txt file
    """
    images_dir = os.path.join(data_dir, 'images')
    labels_file = os.path.join(data_dir, 'labels.txt')

    if not os.path.exists(images_dir) or not os.path.exists(labels_file):
        print(f"Missing images or labels file in {data_dir}")
        return

    # Read labels
    labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    label = parts[1]
                    labels[image_name] = label

    # Prepare file paths
    image_files = []
    label_values = []

    for img_name in labels.keys():
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            image_files.append(img_path)
            label_values.append(labels[img_name])

    print(f"Creating LMDB in {data_dir} with {len(image_files)} images")

    # Create LMDB
    env = lmdb.open(data_dir, map_size=map_size)

    with env.begin(write=True) as txn:
        # Write data
        for idx, (img_path, label) in enumerate(tqdm(zip(image_files, label_values), total=len(image_files))):
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
    print(f"Created LMDB dataset with {len(image_files)} images in {data_dir}")


if __name__ == "__main__":
    base_dir = "C:/Users/jared/PycharmProjects/CLIP4STR/dataset/soccernet"
    fix_validation_dataset(base_dir, val_pct=0.1, replace_existing=True)