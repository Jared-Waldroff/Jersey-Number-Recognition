import os
import warnings
import json
import sys
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import threading

# Limit concurrent GPU calls (example).
# CRUCIAL to prevent too many parallel shipments to our GPU to prevent CUDA-out-of-memory issues
# This will become a bottleneck as we enter series code here, but necessary to avoid exploding GPUs.
GPU_SEMAPHORE = threading.Semaphore(value=1)

os.chdir(str(Path.cwd().parent.parent))
#print(f"(pre-logger) Current working directory: {os.getcwd()}", flush=True)
sys.path.append(os.getcwd())
from DataProcessing.Logger import CustomLogger

# Now CD into pose
os.chdir('./pose/ViTPose/')
#print(f"(prextcoco) Current working directory: {os.getcwd()}", flush=True)

# Append ROOT to PATH
# ROOT = './pose/ViTPose/'
# sys.path.append(str(ROOT))

# Add the current working directory to the path
sys.path.append(os.getcwd())

from xtcocotools.coco import COCO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

# Now the inputs are cleared, cd back to the original directory
#os.chdir(str(Path.cwd().parent.parent))
#print(f"(premain) Current working directory: {os.getcwd()}", flush=True)

def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    logger = CustomLogger().get_logger()
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--json-file', type=str, default='', help='Json file containing image info.')
    parser.add_argument('--out-json', type=str, default='', help='Json file containing results.')
    parser.add_argument('--show', action='store_true', default=False, help='whether to show img')
    parser.add_argument('--out-img-root', type=str, default='',
                        help='Root of the output img file. '
                             'Default not saving the visualization images.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=4, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')

    args = parser.parse_args()
    
    #print(args.show, args.out_img_root)
    # assert args.show or (args.out_img_root != '')

    #print(f"Intaking json file: {args.json_file}")
    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    #print(args.pose_config)
    #print(args.pose_checkpoint)
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # This script is loading some files from prior processes to know which tracklets/images to pull, so handled already
    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    results = []

    # process each image
    for i in range(len(img_keys)):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        # test a single image, with a list of bboxes
        with GPU_SEMAPHORE: # CRITICAL TO NOT EXPOLDE GPU!!!
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                image_name,
                person_results,
                bbox_thr=None,
                format='xywh',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

        # print(pose_results)
        results.append(
            {"img_name": image['file_name'], "id": image_id, "keypoints": pose_results[0]['keypoints'].tolist()})

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)

    if args.out_json != '':
        with open(args.out_json, 'w') as fp:
            json.dump({"pose_results": results}, fp)
            
    logger.info(f"Wrote pose results to file: {args.out_json}")

if __name__ == '__main__':
    main()