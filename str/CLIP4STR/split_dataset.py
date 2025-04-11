import os
import random
import shutil
from pathlib import Path

# Define paths
source_dir = r"C:\Users\jared\PycharmProjects\CLIP4STR\data\SoccerNet\jersey-2023\train"
target_dir = r"C:\Users\jared\PycharmProjects\CLIP4STR\data\SoccerNet\jersey-2023\val"

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# Get all folders in the source directory
folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

# Calculate number of folders to move (20%)
num_to_move = int(len(folders) * 0.2)

# Randomly select folders to move
folders_to_move = random.sample(folders, num_to_move)

print(f"Moving {num_to_move} folders from {len(folders)} total folders")

# Move each selected folder
for folder in folders_to_move:
    source_path = os.path.join(source_dir, folder)
    target_path = os.path.join(target_dir, folder)

    print(f"Moving folder {folder} to validation set")

    # Move the folder
    shutil.move(source_path, target_path)

print(f"Successfully moved {num_to_move} folders to the validation set")