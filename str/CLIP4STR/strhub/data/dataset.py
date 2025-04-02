# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import unicodedata
from pathlib import Path
from typing import Union, Callable, Optional
from torch.utils.data import Dataset
from PIL import Image

class TreeImageDataset(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 charset: str,
                 max_label_len: int,
                 label_path: Optional[str] = None,
                 remove_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 min_image_dim: int = 0,
                 transform: Optional[Callable] = None):

        self.root = Path(root)
        self.charset = charset
        self.max_label_len = max_label_len
        self.transform = transform
        self.labels = []
        self.image_paths = []

        # Load labels from JSON if provided
        label_dict = {}
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Convert all labels to strings
                    label_dict = {k: str(v) for k, v in data.items()}
                elif isinstance(data, list):
                    for item in data:
                        label_dict[item['image']] = str(item['label'])
                else:
                    raise ValueError("Unsupported label file format in " + label_path)

        # List items in the root directory
        items = os.listdir(self.root)
        has_subdirs = any((self.root / x).is_dir() for x in items)

        if has_subdirs:
            # Iterate over each subdirectory
            for sub in items:
                # Skip any directory named "augmented_data"
                if sub.lower() == "augmented_data":
                    continue

                sub_path = self.root / sub
                if sub_path.is_dir():
                    # Use the folder name to look up its label
                    folder_label = label_dict.get(sub, None)
                    # Skip if no label is found or if the label indicates illegibility ("-1")
                    if folder_label is None or folder_label == "-1":
                        continue

                    folder_label = folder_label.strip()
                    if remove_whitespace:
                        folder_label = ''.join(folder_label.split())
                    if normalize_unicode:
                        folder_label = unicodedata.normalize('NFKD', folder_label).encode('ascii', 'ignore').decode()
                    # Validate the label (only allow characters in the charset and within length limit)
                    if not (len(folder_label) <= max_label_len and all(c in charset for c in folder_label)):
                        continue

                    # Iterate over image files in this subdirectory
                    for img_file in os.listdir(sub_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            full_path = sub_path / img_file
                            if min_image_dim > 0:
                                img = Image.open(full_path)
                                if img.width < min_image_dim or img.height < min_image_dim:
                                    continue
                            self.labels.append(folder_label)
                            self.image_paths.append(full_path)
        else:
            # No subdirectories: iterate directly over files in self.root
            for img_file in items:
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = self.root / img_file
                    label = label_dict.get(img_file, img_file)
                    label = str(label)
                    if label == "-1":
                        continue  # Skip illegible images
                    if remove_whitespace:
                        label = ''.join(label.split())
                    if normalize_unicode:
                        label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                    if len(label) <= max_label_len and all(c in charset for c in label):
                        if min_image_dim > 0:
                            img = Image.open(full_path)
                            if img.width < min_image_dim or img.height < min_image_dim:
                                continue
                        self.labels.append(label)
                        self.image_paths.append(full_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_tree_dataset(root: Union[Path, str], charset: str, max_label_len: int,
                       min_image_dim: int = 0, remove_whitespace: bool = True,
                       normalize_unicode: bool = True, transform: Optional[Callable] = None,
                       label_path: Optional[str] = None):
    # Use provided label_path, or default to parent's train_gt.json
    if label_path is None:
        label_path = Path(root).parent / 'train_gt.json'
    else:
        label_path = Path(label_path)
    return TreeImageDataset(
        root=root,
        charset=charset,
        max_label_len=max_label_len,
        label_path=str(label_path),
        min_image_dim=min_image_dim,
        remove_whitespace=remove_whitespace,
        normalize_unicode=normalize_unicode,
        transform=transform
    )

