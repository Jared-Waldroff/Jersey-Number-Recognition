import os
import re


def delete_augmented_images(root_dir):
    pattern = re.compile(r'.*aug.*\.jpg', re.IGNORECASE)
    deleted_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if pattern.match(filename):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    print(f"\nTotal augmented images deleted: {deleted_count}")


if __name__ == "__main__":
    root_directory = r"C:\Users\jared\PycharmProjects\CLIP4STR\data\SoccerNet\jersey-2023\train\images"

    confirmation = input(f"This will delete all augmented images under {root_directory}. Continue? (y/n): ")

    if confirmation.lower() == 'y':
        delete_augmented_images(root_directory)
    else:
        print("Operation cancelled.")