import random
import math
import torch
import torchvision.transforms.functional as F
from PIL import ImageEnhance, Image, ImageFilter
import torchvision.transforms as transforms
from enum import Enum
import numpy as np
import cv2

class LegalTransformations(Enum):
    rotation = 40
    blur = 3
    brightness = (0.7, 1.0)
    contrast = (0.7, 1.0)
    stretch = (0.7, 1.3)
    horizontal_flip = True
    patch_mask = (0.9, 16) # (fraction, patch_size)

class DataAugmentation:
    def __init__(self, data, 
                 max_rotation=LegalTransformations.rotation.value, 
                 max_blur=LegalTransformations.blur.value, 
                 brightness_range=LegalTransformations.brightness.value, 
                 contrast_range=LegalTransformations.contrast.value, 
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
            contrast_range: tuple (min, max) for contrast adjustment factors.
            stretch_range: tuple (min, max) for independent scaling of width and height.
            horizontal_flip: whether to randomly apply horizontal flip.
            patch_mask_keep: fraction (c) of patches to keep in patch masking (0..1).
            patch_size: size of each patch square (in pixels) for patch masking.
        """
        self.data = data
        self.max_rotation = max_rotation
        self.max_blur = max_blur
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
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
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        
    def denormalize(self, tensor):
        """
        Revert the normalization so that the image pixel values are in [0, 1].
        """
        return tensor * self.std + self.mean

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
                    img_array[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = 1.0
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
        
        # 1) Denormalize so we can safely do PIL ops
        image_denorm = self.denormalize(image)
        pil_img = self.to_pil(image_denorm)
        aug_versions = {}
        
        # Always include the original
        aug_versions['original'] = image
        
        # Horizontal flip
        if "horizontal_flip" in valid_transforms and self.horizontal_flip and random.random() < 0.5:
            flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            aug_versions['horizontal_flip'] = self.to_tensor(flipped)
        
        # Blur
        if "blur" in valid_transforms and self.max_blur > 0:
            blur_radius = random.uniform(0, self.max_blur)
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            aug_versions['blur'] = self.to_tensor(blurred)
        
        # Rotation
        if "rotation" in valid_transforms:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            rotated = pil_img.rotate(angle)
            aug_versions['rotation'] = self.to_tensor(rotated)
        
        # Brightness
        if "brightness" in valid_transforms:
            brightness_factor = random.uniform(*self.brightness_range)
            brightened = F.adjust_brightness(pil_img, brightness_factor)
            aug_versions['brightness'] = self.to_tensor(brightened)
        
        # Contrast
        if "contrast" in valid_transforms:
            contrast_factor = random.uniform(*self.contrast_range)
            contrasted = F.adjust_contrast(pil_img, contrast_factor)
            aug_versions['contrast'] = self.to_tensor(contrasted)
        
        # Stretch
        if "stretch" in valid_transforms:
            orig_width, orig_height = pil_img.size
            new_width = int(orig_width * random.uniform(*self.stretch_range))
            new_height = int(orig_height * random.uniform(*self.stretch_range))
            stretched = pil_img.resize((new_width, new_height), resample=Image.BILINEAR)
            # Center crop back to original
            stretched_cropped = F.center_crop(stretched, (orig_height, orig_width))
            aug_versions['stretch'] = self.to_tensor(stretched_cropped)
        
        # Patch masking
        if "patch_mask" in valid_transforms:
            # use the patch_mask_keep fraction to discard patches
            masked_pil = self.patch_masking(pil_img, self.patch_mask_keep, self.patch_size)
            aug_versions['patch_mask'] = self.to_tensor(masked_pil)
        
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

class CLAHEEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8),
                 mean=torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1),
                 std=torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)):
        """
        Args:
            clip_limit: Threshold for contrast limiting.
            tile_grid_size: Size of grid for histogram equalization.
            mean, std: Normalization parameters used for the model.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.mean = mean
        self.std = std

    def denormalize(self, tensor):
        return tensor * self.std + self.mean

    def normalize(self, tensor):
        return (tensor - self.mean) / self.std

    def enhance_with_clahe(self, image_tensor):
        """
        Applies CLAHE enhancement to a normalized image tensor (C, H, W).
        Returns a normalized tensor with enhanced contrast.
        """
        # Convert from torch.Tensor to NumPy image
        img = self.denormalize(image_tensor).clamp(0, 1).cpu().numpy()  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        img = (img * 255).astype(np.uint8)

        # Convert to LAB color space (better for CLAHE)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)

        # Merge the channels and convert back to RGB
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Back to tensor
        enhanced_img = torch.tensor(enhanced_img / 255., dtype=torch.float32).permute(2, 0, 1)
        return self.normalize(enhanced_img)

class ImageEnhancement:
    """
    Our algo is designed so that we train our images on tough scenarios.
    At runtime, we want to help our model out by enhancing the images.
    This means using properties from DataAugmentation to our advantage at runtime.
    Aspects include: increasing contrast, brightness, and sharpness,
    applying CLAHE, optionally an Unsharp Mask, and (optionally) deblurring
    using a Wiener filter to reduce motion blur.
    """

    def __init__(self,
                 brightness_factor=1,
                 contrast_factor=1,
                 sharpness_factor=1.25,
                 use_clahe=True,
                 clahe_clip_limit=1.75,
                 clahe_tile_grid_size=(1, 1),
                 use_unsharp=False,
                 unsharp_radius=0.5,
                 unsharp_percent=120,
                 unsharp_threshold=2,
                 use_deblur=False,
                 deblur_kernel_length=5,       # Small images may use 3 or 5
                 deblur_kernel_angle=0,        # Adjust if motion blur at an angle
                 wiener_K=0.01,                # Regularization constant for the Wiener filter
                 mean=torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1),
                 std=torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)):
        """
        Args:
            brightness_factor: Factor to enhance brightness.
            contrast_factor: Factor to enhance contrast.
            sharpness_factor: Factor to enhance sharpness.
            use_clahe: If True, uses CLAHE instead of PIL enhancements.
            clahe_clip_limit: CLAHE contrast limit.
            clahe_tile_grid_size: CLAHE tile grid size.
            use_unsharp: If True, applies an unsharp mask filter after the primary enhancement.
            unsharp_radius: Radius for the unsharp mask filter.
            unsharp_percent: Percentage for unsharp mask strength.
            unsharp_threshold: Threshold for unsharp mask filter.
            use_deblur: If True, applies a Wiener filter deblurring step to reduce motion blur.
            deblur_kernel_length: Kernel length for the motion blur kernel.
            deblur_kernel_angle: Kernel angle (in degrees) for the motion blur kernel.
            wiener_K: Regularization constant for the Wiener deblurring filter.
            mean, std: Normalization stats used by the model.
        """
        self.use_clahe = use_clahe
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.sharpness_factor = sharpness_factor
        self.use_unsharp = use_unsharp
        self.unsharp_radius = unsharp_radius
        self.unsharp_percent = unsharp_percent
        self.unsharp_threshold = unsharp_threshold
        self.use_deblur = use_deblur
        self.deblur_kernel_length = deblur_kernel_length
        self.deblur_kernel_angle = deblur_kernel_angle
        self.wiener_K = wiener_K
        self.mean = mean
        self.std = std

        # Converters for tensor <-> PIL Image
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        # Optional CLAHE enhancer
        if self.use_clahe:
            self.clahe_enhancer = CLAHEEnhancer(
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
                mean=mean,
                std=std
            )

    def denormalize(self, tensor):
        return tensor * self.std + self.mean

    def normalize(self, tensor):
        return (tensor - self.mean) / self.std

    def enhance_image(self, image):
        """
        Enhances a normalized image tensor (C, H, W).
        If use_clahe=True, applies CLAHE; otherwise uses PIL-based enhancements.
        Optionally applies an unsharp mask and a Wiener deblurring filter.
        Returns a normalized enhanced image tensor.
        """
        # Primary enhancement
        if self.use_clahe:
            enhanced_tensor = self.clahe_enhancer.enhance_with_clahe(image)
        else:
            denorm = self.denormalize(image).clamp(0, 1)
            pil_img = self.to_pil(denorm)
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(self.brightness_factor)
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(self.contrast_factor)
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(self.sharpness_factor)
            enhanced_tensor = self.to_tensor(pil_img)
            enhanced_tensor = self.normalize(enhanced_tensor)

        # Optionally apply an unsharp mask to further enhance edges
        if self.use_unsharp:
            pil_img = self.to_pil(self.denormalize(enhanced_tensor).clamp(0, 1))
            pil_img = pil_img.filter(ImageFilter.UnsharpMask(
                radius=self.unsharp_radius,
                percent=self.unsharp_percent,
                threshold=self.unsharp_threshold))
            enhanced_tensor = self.to_tensor(pil_img)
            enhanced_tensor = self.normalize(enhanced_tensor)

        # Optionally apply Wiener deblurring to reduce motion blur
        if self.use_deblur:
            # Convert tensor back to a PIL image (denormalized) and then to a NumPy array.
            pil_img = self.to_pil(self.denormalize(enhanced_tensor).clamp(0, 1))
            img_np = np.array(pil_img)  # shape: (H, W, C)
            # Create the motion blur kernel.
            kernel = self.motion_blur_kernel(self.deblur_kernel_length, self.deblur_kernel_angle)
            # Apply the Wiener filter deblurring to each channel independently.
            deblurred = np.zeros_like(img_np)
            for c in range(img_np.shape[2]):
                deblurred[..., c] = self.wiener_deblur(img_np[..., c], kernel, self.wiener_K)
            # Convert back to a PIL image.
            pil_img_deblurred = Image.fromarray(deblurred)
            # Convert to tensor and re-normalize.
            enhanced_tensor = self.to_tensor(pil_img_deblurred)
            enhanced_tensor = self.normalize(enhanced_tensor)

        return enhanced_tensor

    @staticmethod
    def motion_blur_kernel(length, angle):
        """
        Create a normalized motion blur kernel of a given length and angle.
        """
        # Create a horizontal line kernel.
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[int((length - 1) / 2), :] = np.ones(length, dtype=np.float32)
        # Rotate the kernel to the specified angle.
        center = (length / 2 - 0.5, length / 2 - 0.5)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
        kernel = kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel
        return kernel

    @staticmethod
    def wiener_deblur(image, kernel, K=0.01):
        """
        Apply Wiener filter deconvolution to remove blur.
        Args:
            image: Input blurred image as a 2D NumPy array.
            kernel: Blur kernel.
            K: Regularization constant.
        Returns:
            Deblurred image as a NumPy array.
        """
        image = np.float32(image) / 255.0
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel, s=image.shape)
        kernel_fft_conj = np.conj(kernel_fft)
        wiener_filter = kernel_fft_conj / (np.abs(kernel_fft)**2 + K)
        result_fft = image_fft * wiener_filter
        result = np.fft.ifft2(result_fft)
        result = np.abs(result)
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)
        return result