import random
import math
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from enum import Enum

class Transformations(Enum):
    rotation = 20
    blur = 3
    brightness = (0.7, 1.0)
    contrast = (0.1, 0.2)
    stretch = (0.7, 1.3)
    horizontal_flip = True

class DataAugmentation:
    def __init__(self, data, 
                 max_rotation=Transformations.rotation.value, 
                 max_blur=Transformations.blur.value, 
                 brightness_range=Transformations.brightness.value, 
                 contrast_range=Transformations.contrast.value, 
                 stretch_range=Transformations.stretch.value,
                 horizontal_flip=Transformations.horizontal_flip.value):
        """
        Args:
            data: dict mapping track names to tensors of shape (N, C, H, W).
            max_rotation: maximum rotation angle in degrees.
            max_blur: maximum Gaussian blur radius.
            brightness_range: tuple (min, max) for brightness adjustment factors.
            contrast_range: tuple (min, max) for contrast adjustment factors.
            stretch_range: tuple (min, max) for independent scaling of width and height.
            horizontal_flip: whether to randomly apply horizontal flip.
        """
        self.data = data
        self.max_rotation = max_rotation
        self.max_blur = max_blur
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.stretch_range = stretch_range
        self.horizontal_flip = horizontal_flip
        
        # Instantiate converters to and from PIL
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        
        # Allowed transformation keys as strings (from the enum)
        self.allowed_transformations = [t.name for t in Transformations]
        
    def augment_image(self, image, transformations_array):
        """
        Augment a single image tensor (C, H, W) using only the specified transformations.
        Returns a dictionary of augmented versions. Always includes the original image.
        """
        # Validate transformations: only keep those that are allowed.
        valid_transforms = [t for t in transformations_array if t in self.allowed_transformations]
        
        # Convert tensor to PIL Image
        pil_img = self.to_pil(image)
        aug_versions = {}
        
        # Always include the original
        aug_versions['original'] = image
        
        # Horizontal flip: applied if requested and with 50% probability.
        if "horizontal_flip" in valid_transforms and self.horizontal_flip and random.random() < 0.5:
            flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            aug_versions['horizontal_flip'] = self.to_tensor(flipped)
        
        # Blur: apply Gaussian blur with random radius in [0, max_blur]
        if "blur" in valid_transforms and self.max_blur > 0:
            blur_radius = random.uniform(0, self.max_blur)
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            aug_versions['blur'] = self.to_tensor(blurred)
        
        # Rotation: rotate by a random angle in [-max_rotation, max_rotation]
        if "rotation" in valid_transforms:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            rotated = pil_img.rotate(angle)
            aug_versions['rotation'] = self.to_tensor(rotated)
        
        # Brightness: adjust brightness by a random factor in brightness_range.
        if "brightness" in valid_transforms:
            brightness_factor = random.uniform(*self.brightness_range)
            brightened = F.adjust_brightness(pil_img, brightness_factor)
            aug_versions['brightness'] = self.to_tensor(brightened)
        
        # Contrast: adjust contrast by a random factor in contrast_range.
        if "contrast" in valid_transforms:
            contrast_factor = random.uniform(*self.contrast_range)
            contrasted = F.adjust_contrast(pil_img, contrast_factor)
            aug_versions['contrast'] = self.to_tensor(contrasted)
        
        # Stretch: independently scale width and height, then center-crop back to original size.
        if "stretch" in valid_transforms:
            orig_width, orig_height = pil_img.size  # (width, height)
            new_width = int(orig_width * random.uniform(*self.stretch_range))
            new_height = int(orig_height * random.uniform(*self.stretch_range))
            stretched = pil_img.resize((new_width, new_height), resample=Image.BILINEAR)
            # Center crop to original dimensions
            stretched_cropped = F.center_crop(stretched, (orig_height, orig_width))
            aug_versions['stretch'] = self.to_tensor(stretched_cropped)
        
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
                img = tensor[i]  # shape: (C, H, W)
                aug_versions = self.augment_image(img, transformations_array)
                track_aug.append(aug_versions)
            augmented_data[track] = track_aug
        return augmented_data
