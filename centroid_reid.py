from pathlib import Path
import sys
import os
import argparse
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

ROOT = './reid/centroids-reid/'
sys.path.append(str(ROOT))  # add ROOT to PATH

import numpy as np
import torch
from tqdm.notebook import tqdm
import cv2
from PIL import Image

from reid.CentroidsReidRepo.config.defaults import _C as cfg
from train_ctl_model import CTLModel

from datasets.transforms import ReidTransforms



# Based on this repo: https://github.com/mikwieczorek/centroids-reid
# Trained model from here: https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK
CONFIG_FILE = str(ROOT+'/configs/256_resnet50.yml')
MODEL_FILE = str(ROOT+'/models/resnet50-19c8e357.pth')

# dict used to get model config and weights using model version
ver_to_specs = {}
ver_to_specs["res50_market"] = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/market1501_resnet50_256_128_epoch_120.ckpt')
ver_to_specs["res50_duke"]   = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt')


def get_specs_from_version(model_version):
    conf, weights = ver_to_specs[model_version]
    conf, weights = str(conf), str(weights)
    return conf, weights

def generate_features(input_folder, output_folder, model_version='res50_market'):
    # load model
    CONFIG_FILE, MODEL_FILE = get_specs_from_version(model_version)
    cfg.merge_from_file(CONFIG_FILE)
    opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
    cfg.merge_from_list(opts)
    
    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False
    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)

    # print("Loading from " + MODEL_FILE)
    if use_cuda:
        model.to('cuda')
        print("using GPU")
    model.eval()

    # Skip hidden files like .DS_Store
    tracks = [t for t in os.listdir(input_folder) if not t.startswith('.')]
    transforms_base = ReidTransforms(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False)

    # Count how many we've processed
    processed_count = 0
    start_time = time.time()

    # Check if we should use tqdm
    use_tqdm = True
    try:
        track_iterator = tqdm(tracks)
    except:
        use_tqdm = False
        track_iterator = tracks
        logger.info("tqdm not available, using basic logging")

    for track in track_iterator:
        # Normalize the path to handle Windows path issues
        track_path = os.path.normpath(os.path.join(input_folder, track))

        # Skip if not a directory
        if not os.path.isdir(track_path):
            logger.info(f"Skipping {track_path} - not a directory")
            continue

        # Check if output already exists and skip if it does
        output_file = os.path.join(output_folder, f"{track}_features.npy")
        if os.path.exists(output_file):
            logger.info(f"Skipping {track} - output already exists")
            processed_count += 1
            continue

        if not use_tqdm and processed_count % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Processed {processed_count}/{len(tracks)} tracks ({processed_count / len(tracks) * 100:.1f}%) in {elapsed:.1f}s")

        features = []
        # Skip hidden files
        images = [img for img in os.listdir(track_path) if not img.startswith('.')]
        logger.info(f"Processing {track} with {len(images)} images")

        # Process each image
        for img_path in images:
            img_full_path = os.path.normpath(os.path.join(track_path, img_path))
            try:
                img = cv2.imread(img_full_path)
                if img is None:
                    logger.warning(f"Could not read image {img_full_path}")
                    continue

                input_img = Image.fromarray(img)
                input_img = torch.stack([val_transforms(input_img)])
                with torch.no_grad():
                    _, global_feat = model.backbone(input_img.cuda() if use_cuda else input_img)
                    global_feat = model.bn(global_feat)
                features.append(global_feat.cpu().numpy().reshape(-1, ))
            except Exception as e:
                logger.error(f"Error processing {img_full_path}: {e}")
                continue

        if features:
            np_feat = np.array(features)
            with open(output_file, 'wb') as f:
                np.save(f, np_feat)
            logger.info(f"Saved features for {track} with shape {np_feat.shape}")
        else:
            logger.warning(f"No features generated for {track_path}")

        processed_count += 1

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    args = parser.parse_args()

    #create if does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    # Convert Windows backslashes to forward slashes for consistency
    input_folder = args.tracklets_folder.replace('\\', '/')
    output_folder = args.output_folder.replace('\\', '/')

    generate_features(input_folder, output_folder)