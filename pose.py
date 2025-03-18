import os
import warnings
import json
import sys
from pathlib import Path

ROOT = './pose/ViTPose/'
sys.path.append(str(ROOT))  # add ROOT to PATH

os.chdir(str(Path.cwd().parent.parent))  # ensure correct working directory
print("Current working directory: ", os.getcwd())

from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
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
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='Use cached data to speed up the process')

    args = parser.parse_args()

    print(args.show, args.out_img_root)

    coco = COCO(args.json_file)

    # build the pose model from a config file and a checkpoint file
    print(args.pose_config)
    print(args.pose_checkpoint)
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

    # all images in the COCO-style annotation
    img_keys = list(coco.imgs.keys())

    # If not returning heatmaps or specific layers, just set these to None
    return_heatmap = False
    output_layer_names = None

    # We will accumulate final pose results here
    final_results = []

    # -------------------------
    # Step 1: Load from cache (if exists and if requested)
    # -------------------------
    loaded_results = []
    if args.use_cache and os.path.exists(args.out_json):
        print(f"Using cached results from: {args.out_json}")
        with open(args.out_json, 'r') as fp:
            data = json.load(fp)
            loaded_results = data.get('pose_results', [])

    # Build a set of processed image IDs from loaded_results
    processed_ids = set(r["id"] for r in loaded_results)

    # Filter out the images we have already processed
    remaining_img_keys = [k for k in img_keys if k not in processed_ids]

    print(f"Total images: {len(img_keys)}, Already processed: {len(processed_ids)}, "
          f"Remaining: {len(remaining_img_keys)}")

    # -------------------------
    # Step 2: Process remaining images
    # -------------------------
    new_results = []
    for i, image_id in enumerate(remaining_img_keys):
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person_results.append({'bbox': ann['bbox']})

        # inference on a single image with bounding boxes
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

        # store the first person's keypoints (if there's at least one)
        if pose_results:
            new_results.append({
                "img_name": image['file_name'],
                "id": image_id,
                "keypoints": pose_results[0]['keypoints'].tolist()
            })

        # visualization
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
            out_file=out_file
        )

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
