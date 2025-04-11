import random
import math
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from enum import Enum
from PIL import ImageEnhance
import numpy as np
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class LegalTransformations(Enum):
    rotation = 40
    blur = 3
    brightness = (0.7, 1.0)
    stretch = (0.7, 1.3)
    horizontal_flip = True
    patch_mask = (0.9, 16)  # (fraction, patch_size)


class DataAugmentation:
    def __init__(self, data,
                 max_rotation=LegalTransformations.rotation.value,
                 max_blur=LegalTransformations.blur.value,
                 brightness_range=LegalTransformations.brightness.value,
                 stretch_range=LegalTransformations.stretch.value,
                 horizontal_flip=LegalTransformations.horizontal_flip.value,
                 patch_mask_keep=LegalTransformations.patch_mask.value[0],
                 patch_size=LegalTransformations.patch_mask.value[1]
                 ):
        """
        Args:
            data: dict mapping track names to tensors of shape (N, C, H, W).
            max_rotation: maximum rotation angle in degrees.
            max_blur: maximum Gaussian blur radius.
            brightness_range: tuple (min, max) for brightness adjustment factors.
            stretch_range: tuple (min, max) for independent scaling of width and height.
            horizontal_flip: whether to randomly apply horizontal flip.
            patch_mask_keep: fraction (c) of patches to keep in patch masking (0..1).
            patch_size: size of each patch square (in pixels) for patch masking.
        """
        self.data = data
        self.max_rotation = max_rotation
        self.max_blur = max_blur
        self.brightness_range = brightness_range
        self.stretch_range = stretch_range
        self.horizontal_flip = horizontal_flip
        self.patch_mask_keep = patch_mask_keep
        self.patch_size = patch_size

        # Instantiate converters to and from PIL
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        # Allowed transformation keys as strings (from the enum)
        self.allowed_transformations = [t.name for t in LegalTransformations]

        # Define the normalization parameters used originally.
        # Check if CUDA is available and data is on CUDA
        device = 'cpu'
        for tensor in data.values():
            if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
                device = 'cuda'
                break

        # Move mean and std to the same device as data
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    def denormalize(self, tensor):
        """
        Revert the normalization so that the image pixel values are in [0, 1].
        """
        return tensor * self.std + self.mean

    def normalize(self, tensor):
        """
        Apply normalization to the tensor.
        """
        return (tensor - self.mean) / self.std

    def patch_masking(self, pil_img, keep_fraction, patch_size):
        """
        Masks out (1 - keep_fraction) of the patches in the image by setting them to white.

        Args:
            pil_img: a PIL Image in [0..255] range
            keep_fraction: fraction of patches to keep (0..1)
            patch_size: size of each patch in pixels
        Returns:
            A new PIL Image with some patches replaced by white.
        """
        # 1) Convert to a float32 array in [0..1]
        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        height, width, _ = img_array.shape

        # 2) Determine how many patches to keep
        patch_count_w = width // patch_size
        patch_count_h = height // patch_size
        total_patches = patch_count_w * patch_count_h
        keep_count = int(total_patches * keep_fraction)

        # 3) Randomly select which patches to keep
        all_indices = list(range(total_patches))
        random.shuffle(all_indices)
        keep_indices = set(all_indices[:keep_count])

        # 4) For each patch index *not* in keep_indices, set that patch to 1.0 (white in [0..1])
        idx = 0
        for py in range(patch_count_h):
            for px in range(patch_count_w):
                if idx not in keep_indices:
                    y_start = py * patch_size
                    x_start = px * patch_size
                    # White in [0..1] space
                    img_array[y_start:y_start + patch_size, x_start:x_start + patch_size, :] = 1.0
                idx += 1

        # 5) Convert back to a standard [0..255] PIL image
        masked_pil = Image.fromarray((img_array * 255).astype(np.uint8))

        return masked_pil

    def augment_image(self, image, transformations_array):
        """
        Augment a single image tensor (C, H, W) using only the specified transformations.
        Returns a dictionary of augmented versions. Always includes the original image.
        """
        valid_transforms = [t for t in transformations_array if t in self.allowed_transformations]
        device = self.mean.device  # Get the device of mean tensor for consistency

        # 1) Denormalize so we can safely do PIL ops
        image_denorm = self.denormalize(image)
        pil_img = self.to_pil(image_denorm)
        aug_versions = {}

        # Always include the original
        aug_versions['original'] = image

        # Horizontal flip
        if "horizontal_flip" in valid_transforms and self.horizontal_flip and random.random() < 0.5:
            flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            tensor = self.to_tensor(flipped).to(device)  # Move to correct device
            aug_versions['horizontal_flip'] = self.normalize(tensor)

        # Blur
        if "blur" in valid_transforms and self.max_blur > 0:
            blur_radius = random.uniform(0, self.max_blur)
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            tensor = self.to_tensor(blurred).to(device)  # Move to correct device
            aug_versions['blur'] = self.normalize(tensor)

        # Rotation
        if "rotation" in valid_transforms:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            rotated = pil_img.rotate(angle)
            tensor = self.to_tensor(rotated).to(device)  # Move to correct device
            aug_versions['rotation'] = self.normalize(tensor)

        # Brightness
        if "brightness" in valid_transforms:
            brightness_factor = random.uniform(*self.brightness_range)
            brightened = F.adjust_brightness(pil_img, brightness_factor)
            tensor = self.to_tensor(brightened).to(device)  # Move to correct device
            aug_versions['brightness'] = self.normalize(tensor)

        # # Contrast
        # if "contrast" in valid_transforms:
        #     try:
        #         # Use a wider range for contrast, similar to brightness
        #         contrast_factor = random.uniform(0.7, 1.3)  # Instead of using self.contrast_range
        #
        #         # Use PIL's ImageEnhance for contrast instead of functional F
        #         enhancer = ImageEnhance.Contrast(pil_img)
        #         contrasted = enhancer.enhance(contrast_factor)
        #
        #         # Ensure tensor is on the correct device
        #         tensor = self.to_tensor(contrasted).to(device)
        #         aug_versions['contrast'] = self.normalize(tensor)
        #
        #     except Exception as e:
        #         print(f"Contrast error: {e}")
        #         # Fallback to brightness if contrast fails
        #         if "brightness" not in aug_versions:
        #             brightness_factor = random.uniform(0.7, 1.3)
        #             brightened = F.adjust_brightness(pil_img, brightness_factor)
        #             tensor = self.to_tensor(brightened).to(device)
        #             aug_versions['contrast'] = self.normalize(tensor)  # Still label it as contrast

        # Stretch
        if "stretch" in valid_transforms:
            orig_width, orig_height = pil_img.size
            new_width = int(orig_width * random.uniform(*self.stretch_range))
            new_height = int(orig_height * random.uniform(*self.stretch_range))
            stretched = pil_img.resize((new_width, new_height), resample=Image.BILINEAR)
            # Center crop back to original
            stretched_cropped = F.center_crop(stretched, (orig_height, orig_width))
            tensor = self.to_tensor(stretched_cropped).to(device)  # Move to correct device
            aug_versions['stretch'] = self.normalize(tensor)

        # Patch masking
        if "patch_mask" in valid_transforms:
            # use the patch_mask_keep fraction to discard patches
            masked_pil = self.patch_masking(pil_img, self.patch_mask_keep, self.patch_size)
            tensor = self.to_tensor(masked_pil).to(device)  # Move to correct device
            aug_versions['patch_mask'] = self.normalize(tensor)

        return aug_versions
    def augment_data(self, transformations_array: list = None):
        """
        Applies the specified augmentations to each image in self.data.
        Returns a new dictionary mapping each track to a list (one per image) of dictionaries.
        Each inner dictionary maps augmentation type to the augmented image tensor.

        If transformations_array is None or empty, all allowed transformations are applied.
        """
        if not transformations_array:
            transformations_array = self.allowed_transformations

        augmented_data = {}
        for track, tensor in self.data.items():
            # tensor shape: (N, C, H, W)
            track_aug = []
            for i in range(tensor.shape[0]):
                img = tensor[i]
                aug_versions = self.augment_image(img, transformations_array)
                track_aug.append(aug_versions)
            augmented_data[track] = track_aug
        return augmented_data