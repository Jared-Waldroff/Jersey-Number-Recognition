import json
import cv2
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import shutil
from pathlib import Path
from scipy.special import softmax as softmax

HEIGHT_MIN = 35
WIDTH_MIN = 30

def identify_soccer_balls(image_dir, soccer_ball_list):
    # check 10 random images for each track, mark as soccer ball if the size matches typical soccer ball size
    ball_list = []
    # Filter out hidden files and ensure we only process directories
    tracklets = [t for t in os.listdir(image_dir) if
                 not t.startswith('.') and os.path.isdir(os.path.join(image_dir, t))]

    for track in tqdm(tracklets):
        track_path = os.path.join(image_dir, track)
        # Skip if not a directory (extra safety check)
        if not os.path.isdir(track_path):
            continue

        # Filter out hidden files when listing images
        image_names = [img for img in os.listdir(track_path) if not img.startswith('.')]

        if not image_names:  # Skip if no images found
            continue

        sample = len(image_names) if len(image_names) < 10 else 10
        imgs = np.random.choice(image_names, size=sample, replace=False)
        width_list = []
        height_list = []
        for img_name in imgs:
            img_path = os.path.join(track_path, img_name)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            width_list.append(w)
            height_list.append(h)
        mean_w, mean_h = np.mean(width_list), np.mean(height_list)
        if mean_h <= HEIGHT_MIN and mean_w <= WIDTH_MIN:
            # this must be a soccer ball
            ball_list.append(track)
    print(f"Found {len(ball_list)} balls, Ball list: {ball_list}")
    with open(soccer_ball_list, 'w') as fp:
        json.dump({'ball_tracks': ball_list}, fp)
    return True