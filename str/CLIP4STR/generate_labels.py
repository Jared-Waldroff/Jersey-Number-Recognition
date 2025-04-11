import os
import json
import glob

# Paths
json_path = r'C:\Users\jared\PycharmProjects\CLIP4STR\dataset\soccernet\train\train_gt.json'
image_dir = r'C:\Users\jared\PycharmProjects\CLIP4STR\dataset\soccernet\val\images'
output_path = r'C:\Users\jared\PycharmProjects\CLIP4STR\dataset\soccernet\val\labels.txt'

# Load jersey numbers from JSON
with open(json_path, 'r') as f:
    jersey_labels = json.load(f)

# Find all image files
image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

# Prepare labels
labels = []
for img_path in image_files:
    filename = os.path.basename(img_path)
    # Split filename into group and image number
    parts = filename.split('_')
    group = parts[0]

    # Check if this group exists in labels and is not -1
    if group in jersey_labels and jersey_labels[group] != -1:
        labels.append(f"{filename} {jersey_labels[group]}")

# Sort labels for consistency
labels.sort()

# Write labels to file
with open(output_path, 'w') as f:
    f.write('\n'.join(labels))

print(f"Created labels.txt with {len(labels)} entries")