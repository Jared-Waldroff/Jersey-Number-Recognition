import os
import re

# Compile the regex pattern to skip paths with SoccerNet/challenge, SoccerNet/test, or SoccerNet/train.
pattern = re.compile(r'data[\\/]SoccerNet[\\/](challenge|test|train)', re.IGNORECASE)

root_dir = '.'
output_file = 'project_structure.txt'
all_items = []

print("Scanning directories and files...")

# Walk through the directory structure.
for current_root, dirs, files in os.walk(root_dir):
    # Process the current directory (skip the root directory itself to avoid extra indent)
    if os.path.abspath(current_root) != os.path.abspath(root_dir):
        # Check if the directory should be skipped based on the regex.
        if pattern.search(current_root):
            print(f"Skipped directory due to pattern: {current_root}")
        else:
            all_items.append(current_root)
            print(f"Added directory: {current_root}")

    # Process files in the current directory.
    for file in files:
        full_path = os.path.join(current_root, file)
        # Skip .jpg files.
        if os.path.splitext(file)[1].lower() == '.jpg':
            print(f"Skipped .jpg file: {full_path}")
            continue
        # Skip files in excluded SoccerNet folders.
        if pattern.search(full_path):
            print(f"Skipped file due to pattern: {full_path}")
            continue
        all_items.append(full_path)
        print(f"Added file: {full_path}")

# Sort items by their full path.
all_items.sort()

print("\nWriting output to file...")
with open(output_file, 'w') as f:
    for item in all_items:
        # Calculate depth relative to the root directory for indentation.
        rel_path = os.path.relpath(item, root_dir)
        depth = rel_path.count(os.sep)
        indent = ' ' * depth
        line = f"{indent}{os.path.basename(item)}"
        f.write(line + '\n')
        print(f"Wrote: {line}")

print(f"\nDone. Project structure has been written to '{output_file}'.")
