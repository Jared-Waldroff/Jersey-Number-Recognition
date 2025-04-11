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


def create_all_augmentations(image_path, output_path=None):
    """
    Create all augmentation types for a single image

    Args:
        image_path: Path to the original image
        output_path: Path to save augmented images (default: same as original)
    """
    if output_path is None:
        output_path = os.path.dirname(image_path)

    # Load image and convert to tensor
    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(to_tensor(img)).unsqueeze(0)  # (1, C, H, W)

    # Setup augmentation with custom parameters
    augmenter = DataAugmentation(
        {'image': img_tensor},
        max_rotation=30,  # Limit rotation to 30 degrees
        max_blur=3,
        brightness_range=(0.7, 1.3),
        stretch_range=(0.8, 1.2),
        horizontal_flip=False,  # No flipping
        patch_mask_keep=0.9,
        patch_size=16
    )

    # Define all augmentation types
    aug_types = ["rotation", "blur", "brightness", "stretch", "patch_mask"]

    # Get base filename and extension
    base_name, ext = os.path.splitext(os.path.basename(image_path))

    # Apply augmentations and save images
    saved_paths = []
    for aug_type in aug_types:
        # Get augmented versions
        aug_results = augmenter.augment_image(img_tensor[0], [aug_type])

        # Skip the 'original' version
        if 'original' in aug_results:
            del aug_results['original']

        for aug_name, aug_img in aug_results.items():
            # Create new filename
            aug_filename = f"{base_name}_aug_{aug_type}{ext}"
            aug_path = os.path.join(output_path, aug_filename)

            # Denormalize and convert tensor to PIL
            aug_img = aug_img * augmenter.std + augmenter.mean
            aug_img = aug_img.clamp(0, 1)

            aug_pil = transforms.ToPILImage()(aug_img)
            aug_pil.save(aug_path)
            saved_paths.append(aug_path)
            print(f"Saved augmented image: {aug_filename}")

    return saved_paths


def main():
    # Hardcoded image path
    image_path = r"C:\Users\jared\PycharmProjects\CLIP4STR\data\SoccerNet\jersey-2023\train\images\2\1418_416.jpg"

    # Create augmentations
    augmented_images = create_all_augmentations(image_path)

    print("\nAugmentation complete!")
    print("Augmented images:")
    for path in augmented_images:
        print(f"  - {os.path.basename(path)}")


if __name__ == "__main__":
    main()