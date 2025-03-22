import os
import warnings
import json
import sys
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Append ROOT to PATH and change working directory.
ROOT = './pose/ViTPose/'
sys.path.append(str(ROOT))
os.chdir(str(Path.cwd().parent.parent))
print("Current working directory: ", os.getcwd())

from xtcocotools.coco import COCO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

# Global variables for workers (will be initialized once per worker)
global_pose_model = None
global_dataset = None
global_dataset_info = None
global_args = None
global_coco = None

def worker_init(args):
    """Initializer for each worker process.
    Loads the pose model and dataset info so each worker can run inference on GPU.
    """
    global global_pose_model, global_dataset, global_dataset_info, global_args, global_coco
    global_args = args
    # Initialize COCO object (each worker will have its own copy)
    global_coco = COCO(args.json_file)
    # Load the pose model on the specified device.
    global_pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device.lower())
    dataset = global_pose_model.cfg.data['test']['type']
    dataset_info = global_pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config. See https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    global_dataset = dataset
    global_dataset_info = dataset_info

def process_image(image_id):
    """Worker function to process a single image.
    Loads the image, runs pose estimation and visualization, and returns a result dict.
    """
    # Load image info from the worker's COCO instance.
    image = global_coco.loadImgs(image_id)[0]
    image_name = os.path.join(global_args.img_root, image['file_name'])
    ann_ids = global_coco.getAnnIds(image_id)
    person_results = []
    for ann_id in ann_ids:
        ann = global_coco.anns[ann_id]
        person_results.append({'bbox': ann['bbox']})
    
    # Run pose estimation on the image.
    pose_results, returned_outputs = inference_top_down_pose_model(
        global_pose_model,
        image_name,
        person_results,
        bbox_thr=None,
        format='xywh',
        dataset=global_dataset,
        dataset_info=global_dataset_info,
        return_heatmap=global_args.return_heatmap,
        outputs=global_args.output_layer_names
    )
    
    # Prepare output filename for visualization.
    if global_args.out_img_root != '':
        os.makedirs(global_args.out_img_root, exist_ok=True)
        # Use the image ID to generate a unique output filename.
        out_file = os.path.join(global_args.out_img_root, f'vis_{image_id}.jpg')
    else:
        out_file = None

    # Visualize the pose results.
    vis_pose_result(
        global_pose_model,
        image_name,
        pose_results,
        dataset=global_dataset,
        dataset_info=global_dataset_info,
        kpt_score_thr=global_args.kpt_thr,
        radius=global_args.radius,
        thickness=global_args.thickness,
        show=global_args.show,
        out_file=out_file
    )
    
    # If any pose results are available, return a dict.
    if pose_results:
        return {
            "img_name": image['file_name'],
            "id": image_id,
            "keypoints": pose_results[0]['keypoints'].tolist()
        }
    else:
        return None

def main():
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--json-file', type=str, default='', help='Json file containing image info.')
    parser.add_argument('--out-json', type=str, default='', help='Json file containing results.')
    parser.add_argument('--show', action='store_true', default=False, help='whether to show img')
    parser.add_argument('--out-img-root', type=str, default='',
                        help='Root of the output img file. Default not saving the visualization images.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=4, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--use_cache', action='store_true', default=False, help='Use cached data to speed up the process')
    
    args = parser.parse_args()
    print("Show:", args.show, "Out img root:", args.out_img_root)
    
    # Initialize COCO using the json_file
    coco = COCO(args.json_file)
    img_keys = list(coco.imgs.keys())
    
    # Set additional parameters needed for inference.
    args.return_heatmap = False
    args.output_layer_names = None
    args.coco = coco  # also pass the COCO instance (for main process caching)
    
    # -------------------------
    # Step 1: Load from cache (if exists and if requested)
    # -------------------------
    loaded_results = []
    if args.use_cache and os.path.exists(args.out_json):
        print(f"Using cached results from: {args.out_json}")
        with open(args.out_json, 'r') as fp:
            data = json.load(fp)
            loaded_results = data.get('pose_results', [])
    
    processed_ids = set(r["id"] for r in loaded_results)
    remaining_img_keys = [k for k in img_keys if k not in processed_ids]
    print(f"Total images: {len(img_keys)}, Already processed: {len(processed_ids)}, Remaining: {len(remaining_img_keys)}")
    
    # -------------------------
    # Step 2: Process remaining images in parallel
    # -------------------------
    new_results = []
    num_workers = 4  # Set number of workers as appropriate (or dynamically, e.g. os.cpu_count())
    with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init, initargs=(args,)) as executor:
        futures = {executor.submit(process_image, image_id): image_id for image_id in remaining_img_keys}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pose estimation"):
            try:
                result = future.result()
                if result is not None:
                    new_results.append(result)
            except Exception as e:
                print(f"Error processing image {futures[future]}: {e}")
    
    # -------------------------
    # Step 3: Combine loaded results + new results and save
    # -------------------------
    final_results = loaded_results + new_results
    if args.out_json:
        with open(args.out_json, 'w') as fp:
            json.dump({"pose_results": final_results}, fp)
        print(f"Saved combined results to {args.out_json}")

if __name__ == '__main__':
    main()
