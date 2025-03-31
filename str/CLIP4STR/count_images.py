# Quick script to count actual images
import os
import glob

image_dir = "C:/Users/jared/PycharmProjects/CLIP4STR/dataset/soccernet/train/images"
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
print(f"Found {len(image_files)} total images")