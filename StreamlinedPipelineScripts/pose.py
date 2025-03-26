import os
import warnings
import json
import sys
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import threading
import math

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
    logger = CustomLogger().get_logger()

    try:
        parser = ArgumentParser()
        parser.add_argument('pose_config', help='Config file for detection')
        parser.add_argument('pose_checkpoint', help='Checkpoint file')
        parser.add_argument('--img-root', type=str, default='', help='Image root')
        parser.add_argument('--json-file', type=str, default='', help='Json file containing image info.')
        parser.add_argument('--out-json', type=str, default='', help='Json file containing results.')
        parser.add_argument('--show', action='store_true', default=False, help='whether to show img')
        parser.add_argument('--out-img-root', type=str, default='',
                            help='Root of the output img file.')
        parser.add_argument('--device', default='cuda:0', help='Device used for inference')
        parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
        parser.add_argument('--radius', type=int, default=4, help='Keypoint radius for visualization')
        parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
        
        # image_batch_size param:
        parser.add_argument('--image-batch-size', type=int, default=200, help='Number of images to process in a batch.')

        args = parser.parse_args()
    except Exception as e:
        logger.error(f"Argument parsing failed: {e}")
        sys.exit(1)

    try:
        coco = COCO(args.json_file)
    except Exception as e:
        logger.error(f"Failed to load COCO annotations from {args.json_file}: {e}")
        sys.exit(1)

    try:
        pose_model = init_pose_model(
            args.pose_config, args.pose_checkpoint, device=args.device.lower())
    except Exception as e:
        logger.error(f"Failed to initialize pose model: {e}")
        sys.exit(1)

    try:
        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.',
                DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)

        img_keys = list(coco.imgs.keys())
        return_heatmap = False
        output_layer_names = None
        results = []
        
        # Determine total number of images and compute number of batches.
        num_images_to_process = len(img_keys)
        num_batches = math.ceil(num_images_to_process / args.image_batch_size)

        results = []

        for b in tqdm(range(num_batches), desc="Processing batches", leave=True):
            batch_keys = img_keys[b * batch_size : (b + 1) * batch_size]
            # Process each image in this batch.
            for i, image_id in enumerate(batch_keys):
                try:
                    image = coco.loadImgs(image_id)[0]
                    image_name = os.path.join(args.img_root, image['file_name'])
                    ann_ids = coco.getAnnIds(image_id)

                    # Build bounding boxes for the image.
                    person_results = []
                    for ann_id in ann_ids:
                        ann = coco.anns[ann_id]
                        person_results.append({'bbox': ann['bbox']})

                    # Use the GPU semaphore to protect GPU memory.
                    with GPU_SEMAPHORE:
                        pose_results, returned_outputs = inference_top_down_pose_model(
                            pose_model,
                            image_name,
                            person_results,
                            bbox_thr=None,
                            format='xywh',
                            dataset=dataset,
                            dataset_info=dataset_info,
                            return_heatmap=return_heatmap,
                            outputs=output_layer_names
                        )

                    # Append the result for this image.
                    results.append({
                        "img_name": image['file_name'],
                        "id": image_id,
                        "keypoints": pose_results[0]['keypoints'].tolist()
                    })

                    # Visualization: if an output image root is specified, write the vis result.
                    if args.out_img_root:
                        os.makedirs(args.out_img_root, exist_ok=True)
                        # Use the global image index computed as b * batch_size + i.
                        out_file = os.path.join(args.out_img_root, f'vis_{b * batch_size + i}.jpg')
                    else:
                        out_file = None

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
                        out_file=out_file
                    )

                except Exception as img_error:
                    logger.error(f"Failed to process image {b * batch_size + i} ({image['file_name']}): {img_error}")

        if args.out_json:
            try:
                with open(args.out_json, 'w') as fp:
                    json.dump({"pose_results": results}, fp)
                logger.info(f"Wrote pose results to file: {args.out_json}")
            except Exception as write_error:
                logger.error(f"Failed to write output JSON to {args.out_json}: {write_error}")

    except Exception as e:
        logger.error(f"Unexpected error during pose estimation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()