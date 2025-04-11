import os
import shutil

# Paths to the directories
val_images_dir = r'C:\Users\jared\PycharmProjects\CLIP4STR\dataset\soccernet\val\images'
train_images_dir = r'C:\Users\jared\PycharmProjects\CLIP4STR\dataset\soccernet\train\images'

# Get list of filenames in the val images directory
val_image_filenames = set(os.listdir(val_images_dir))

# Counter for deleted files
deleted_count = 0

# Iterate through files in the train images directory
for filename in os.listdir(train_images_dir):
    if filename in val_image_filenames:
        # Full path to the file to be deleted
        file_path = os.path.join(train_images_dir, filename)

        try:
            # Delete the file
            os.remove(file_path)
            print(f"Deleted: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

print(f"\nTotal files deleted: {deleted_count}")