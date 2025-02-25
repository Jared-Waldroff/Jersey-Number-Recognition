import zipfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

def extract_folder_members(zip_path, members, output_dir):
    # Open the zip file once per folder extraction task
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in members:
            zf.extract(member, path=output_dir)

def parallel_extract_folders(zip_path, output_dir, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    # Open the zip file to get all member names
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()

    # Group members by their top-level directory (assumes paths like "1/img.jpg")
    groups = defaultdict(list)
    for member in members:
        # The top-level directory is the part before the first '/'
        top_folder = member.split('/')[0]
        groups[top_folder].append(member)

    folder_groups = list(groups.items())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Schedule extraction for each folder group in parallel
        futures = [executor.submit(extract_folder_members, zip_path, member_list, output_dir)
                   for folder, member_list in folder_groups]

        # Wrap the future iterator with tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting folders"):
            future.result()

if __name__ == '__main__':
    root = './data/SoccerNet/jersey-2023/'
    zip_file = f'{root}/challenge.zip'
    extract_dir = f'{root}/extracted'
    parallel_extract_folders(zip_file, extract_dir, max_workers=4)
