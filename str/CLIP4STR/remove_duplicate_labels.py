import os

# Paths to the files
val_labels_path = r'C:\Users\jared\PycharmProjects\CLIP4STR\dataset\soccernet\val\labels.txt'
train_labels_path = r'C:\Users\jared\PycharmProjects\CLIP4STR\dataset\soccernet\train\labels.txt'

# Read the validation labels
with open(val_labels_path, 'r') as f:
    val_label_files = {line.split()[0] for line in f}

# Read the training labels
with open(train_labels_path, 'r') as f:
    train_label_files = {line.split()[0] for line in f}

# Find common files
common_files = val_label_files.intersection(train_label_files)

print("Detailed comparison:")
print(f"Total validation image files: {len(val_label_files)}")
print(f"Total training image files: {len(train_label_files)}")
print(f"Number of common files: {len(common_files)}")

# If there are common files, show the first few
if common_files:
    print("\nFirst 20 common files:")
    for file in list(common_files)[:20]:
        print(file)

# Optional: check for partial matches or patterns
val_prefixes = set(f.split('_')[0] for f in val_label_files)
train_prefixes = set(f.split('_')[0] for f in train_label_files)

print("\nPrefix analysis:")
print(f"Unique validation prefixes: {len(val_prefixes)}")
print(f"Unique training prefixes: {len(train_prefixes)}")
print(f"Common prefixes: {len(val_prefixes.intersection(train_prefixes))}")

# If there are common prefixes, show some
common_prefix_list = list(val_prefixes.intersection(train_prefixes))
print("\nFirst 20 common prefixes:")
print(common_prefix_list[:20])