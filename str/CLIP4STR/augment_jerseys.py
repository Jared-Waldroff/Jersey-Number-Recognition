import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import glob
from tqdm import tqdm
import argparse

# Import from the data_augmentation.py file we created
from data_augmentation import DataAugmentation, LegalTransformations

# Add debug printing
print("Script initialized")


def create_augmentations(image_path, output_path=None, chosen_aug=None, device='cuda'):
    """
    Create one augmented version of a single image using the specified augmentation type
    """
    if output_path is None:
        output_path = os.path.dirname(image_path)

    # Use GPU if available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # Load image and convert to tensor
    try:
        img = Image.open(image_path).convert("RGB")
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img).unsqueeze(0).to(device)  # (1, C, H, W)

        # Create mean and std tensors on the same device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

        # Normalize on device
        img_tensor = (img_tensor - mean) / std

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Setup augmentation with custom parameters
    augmenter = DataAugmentation(
        {'image': img_tensor},
        max_rotation=30,
        max_blur=1.2,
        brightness_range=(0.7, 1.3),
        stretch_range=(0.8, 1.2),
        horizontal_flip=False,
        patch_mask_keep=0.9,
        patch_size=16
    )

    # Make sure mean and std are on the right device
    augmenter.mean = augmenter.mean.to(device)
    augmenter.std = augmenter.std.to(device)

    # If no specific augmentation chosen, randomly select one
    if not chosen_aug:
        aug_types = ["rotation", "blur", "brightness", "stretch", "patch_mask"]
        chosen_aug = random.choice(aug_types)

    # Apply the chosen augmentation
    aug_results = augmenter.augment_image(img_tensor[0], [chosen_aug])

    # Get base filename and extension
    base_name, ext = os.path.splitext(os.path.basename(image_path))

    # Skip if augmentation wasn't successful
    if chosen_aug not in aug_results:
        return None

    # Create and save the augmented image
    aug_img = aug_results[chosen_aug]
    aug_filename = f"{base_name}_aug1_{chosen_aug}{ext}"
    aug_path = os.path.join(output_path, aug_filename)

    # Convert tensor to PIL and save
    # First denormalize
    aug_img = aug_img * augmenter.std + augmenter.mean
    aug_img = aug_img.clamp(0, 1).cpu()
    aug_pil = transforms.ToPILImage()(aug_img)
    aug_pil.save(aug_path)

    return aug_path


def process_tracklet_folders(base_dir, start_tracklet=0, device='cuda'):
    """
    Process all tracklet folders and augment images inside them
    Only augments 2 out of every 10 images encountered
    """
    try:
        # Check for CUDA availability if requested
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        else:
            print(f"Using device: {device}")

        # Process folders in numerical order
        tracklet_dirs = []
        max_folder_id = 1500  # Set an upper limit

        print("Scanning for tracklet folders...")
        for i in range(max_folder_id):
            folder_name = str(i)
            folder_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(folder_path):
                tracklet_dirs.append((i, folder_path))

        print(f"Found {len(tracklet_dirs)} tracklet folders")

        # Sort by the numerical ID and filter by start_tracklet
        tracklet_dirs.sort(key=lambda x: x[0])
        tracklet_dirs = [t for t in tracklet_dirs if t[0] >= start_tracklet]
        print(f"Starting from tracklet {start_tracklet}, {len(tracklet_dirs)} folders to process")

        # Counters for overall tracking
        total_images_counter = 0
        total_augmented = 0

        # Track augmentation type distribution
        aug_types = ["rotation", "blur", "brightness", "stretch", "patch_mask"]
        aug_counts = {aug_type: 0 for aug_type in aug_types}

        # Process each tracklet folder
        for folder_idx, (folder_id, tracklet_dir) in enumerate(tracklet_dirs):
            print(f"\nProcessing folder {folder_id} ({folder_idx + 1}/{len(tracklet_dirs)})")

            # Find all images in the folder
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(tracklet_dir, ext)))

            # Filter out any existing augmented images
            image_files = [img for img in image_files if "aug" not in img]
            image_files.sort()

            print(f"  Found {len(image_files)} images in folder {folder_id}")

            # Process each image
            for idx, img_path in enumerate(image_files):
                total_images_counter += 1

                # Only augment the 3rd and 7th image in each group of 10
                if total_images_counter % 10 in [3, 7]:
                    try:
                        # Choose least used augmentation type
                        min_count = min(aug_counts.values())
                        least_used = [aug for aug, count in aug_counts.items() if count == min_count]
                        chosen_aug = random.choice(least_used)

                        print(f"  Augmenting image {idx + 1}: {os.path.basename(img_path)} with {chosen_aug}")

                        # Create one augmentation
                        new_path = create_augmentations(img_path, chosen_aug=chosen_aug, device=device)
                        if new_path:
                            aug_counts[chosen_aug] += 1
                            total_augmented += 1
                            print(f"  Created: {os.path.basename(new_path)}")
                        else:
                            print(f"  Failed to create augmentation")
                    except Exception as e:
                        print(f"  Error augmenting image: {e}")

                # Print progress every 50 images
                if idx > 0 and idx % 50 == 0:
                    print(f"  Processed {idx}/{len(image_files)} images")

            # Print status after each folder
            distribution = ", ".join([f"{k}:{v}" for k, v in aug_counts.items()])
            print(f"Current status: {total_augmented}/{total_images_counter} images. Distribution: {distribution}")

        return total_augmented, total_images_counter, aug_counts

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return 0, 0, {}


def main():
    parser = argparse.ArgumentParser(description="Augment soccer jersey images for training")
    parser.add_argument("--base_dir", type=str,
                        default="C:/Users/jared/PycharmProjects/CLIP4STR/data/SoccerNet/jersey-2023/train/images",
                        help="Base directory containing tracklet folders")
    parser.add_argument("--start_tracklet", type=int, default=0,
                        help="ID of the tracklet to start processing from")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda',
                        help="Device to use for computation")

    args = parser.parse_args()

    # Add diagnostics
    print(f"Current working directory: {os.getcwd()}")
    print(f"Base directory: {args.base_dir}")
    print(f"Directory exists: {os.path.exists(args.base_dir)}")

    # List contents if directory exists
    if os.path.exists(args.base_dir):
        print(f"Contents of {args.base_dir}:")
        for item in os.listdir(args.base_dir):
            print(f"  - {item}")

    process_tracklet_folders(args.base_dir, args.start_tracklet, args.device)
    print("Augmentation completed!")


if __name__ == "__main__":
    print("Script starting main function")
    main()
    print("Script completed")