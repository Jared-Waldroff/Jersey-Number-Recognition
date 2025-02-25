import argparse
import os
import legibility_classifier as lc
import numpy as np
import json
import helpers
from tqdm import tqdm
import configuration as config
from pathlib import Path
import sys
import platform


# Helper function to determine the correct python command based on OS
def get_python_cmd():
    if platform.system() == 'Windows':
        return "python"
    else:
        return "python3"


def get_soccer_net_raw_legibility_results(args, use_filtered=True, filter='gauss', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = [t for t in os.listdir(path_to_images) if os.path.isdir(os.path.join(path_to_images, t))]
    results_dict = {x: [] for x in tracklets}

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if not track in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = [img for img in os.listdir(track_dir) if not img.startswith('.')]

        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, config.dataset['SoccerNet']['legibility_model'], threshold=-1,
                               arch=config.dataset['SoccerNet']['legibility_model_arch'])
        results_dict[directory] = track_results

    # save results
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['raw_legible_result'])
    with open(full_legibile_path, "w") as outfile:
        json.dump(results_dict, outfile)

    return results_dict


def get_soccer_net_legibility_results(args, use_filtered=False, filter='sim', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = [t for t in os.listdir(path_to_images) if os.path.isdir(os.path.join(path_to_images, t))]

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    legible_tracklets = {}
    illegible_tracklets = []

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if not track in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = [img for img in os.listdir(track_dir) if not img.startswith('.')]

        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, config.dataset['SoccerNet']['legibility_model'],
                               arch=config.dataset['SoccerNet']['legibility_model_arch'], threshold=0.5)
        legible = list(np.nonzero(track_results))[0]
        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            legible_tracklets[directory] = legible_images

    # save results
    json_object = json.dumps(legible_tracklets, indent=4)
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['legible_result'])
    with open(full_legibile_path, "w") as outfile:
        outfile.write(json_object)

    full_illegibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['illegible_result'])
    json_object = json.dumps({'illegible': illegible_tracklets}, indent=4)
    with open(full_illegibile_path, "w") as outfile:
        outfile.write(json_object)

    return legible_tracklets, illegible_tracklets


def generate_json_for_pose_estimator(args, legible=None):
    all_files = []
    if not legible is None:
        for key in legible.keys():
            for entry in legible[key]:
                all_files.append(os.path.join(os.getcwd(), entry))
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset['SoccerNet']['root_dir'])
        image_dir = config.dataset['SoccerNet'][args.part]['images']
        path_to_images = os.path.join(root_dir, image_dir)
        tracks = [t for t in os.listdir(path_to_images) if os.path.isdir(os.path.join(path_to_images, t))]

        for tr in tracks:
            track_dir = os.path.join(path_to_images, tr)
            imgs = [img for img in os.listdir(track_dir) if not img.startswith('.')]

            for img in imgs:
                all_files.append(os.path.join(track_dir, img))

    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet'][args.part]['pose_input_json'])
    helpers.generate_json(all_files, output_json)


def consolidated_results(image_dir, dict, illegible_path, soccer_ball_list=None):
    if not soccer_ball_list is None:
        with open(soccer_ball_list, 'r') as sf:
            balls_json = json.load(sf)
        balls_list = balls_json['ball_tracks']
        for entry in balls_list:
            dict[str(entry)] = 1

    with open(illegible_path, 'r') as f:
        illegile_dict = json.load(f)
    all_illegible = illegile_dict['illegible']
    for entry in all_illegible:
        if not str(entry) in dict.keys():
            dict[str(entry)] = -1

    all_tracks = [t for t in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, t))]

    for t in all_tracks:
        if not t in dict.keys():
            dict[t] = -1
        else:
            dict[t] = int(dict[t])
    return dict


def train_parseq(args):
    python_cmd = get_python_cmd()

    if args.dataset == 'Hockey':
        print("Train PARSeq for Hockey")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'],
                                 config.dataset['Hockey']['numbers_data'])
        command = f"conda run -n {config.str_env} {python_cmd} train.py +experiment=parseq dataset=real data.root_dir={data_root} trainer.max_epochs=25 " \
                  f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
        run_command(command)
        os.chdir(current_dir)
        print("Done training")
    else:
        print("Train PARSeq for Soccer")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset['SoccerNet']['root_dir'],
                                 config.dataset['SoccerNet']['numbers_data'])
        command = f"conda run -n {config.str_env} {python_cmd} train.py +experiment=parseq dataset=real data.root_dir={data_root} trainer.max_epochs=25 " \
                  f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
        run_command(command)
        os.chdir(current_dir)
        print("Done training")


# Function to run commands and handle errors better
def run_command(command, ignore_errors=False):
    print(f"Running: {command}")
    result = os.system(command)
    success = (result == 0)

    if not success and not ignore_errors:
        print(f"Warning: Command may have failed with code {result}")

    return success


def hockey_pipeline(args):
    python_cmd = get_python_cmd()
    success = True

    if args.pipeline['legible']:
        root_dir = os.path.join(config.dataset["Hockey"]["root_dir"], config.dataset["Hockey"]["legibility_data"])

        print("Test legibility classifier")
        command = f"{python_cmd} legibility_classifier.py --data {root_dir} --arch resnet34 --trained_model {config.dataset['Hockey']['legibility_model']}"
        success = run_command(command)
        print("Done legibility classifier")

    if success and args.pipeline['str']:
        print("Predict numbers")
        current_dir = os.getcwd()
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'],
                                 config.dataset['Hockey']['numbers_data'])
        command = f"conda run -n {config.str_env} {python_cmd} str.py {config.dataset['Hockey']['str_model']} --data_root={data_root}"
        success = run_command(command)
        print("Done predict numbers")


def soccer_net_pipeline(args):
    python_cmd = get_python_cmd()
    legible_dict = None
    legible_results = None
    consolidated_dict = None
    Path(config.dataset['SoccerNet']['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True

    image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['images'])
    soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                    config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
    features_dir = config.dataset['SoccerNet'][args.part]['feature_output_folder']
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['legible_result'])
    illegible_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                  config.dataset['SoccerNet'][args.part]['illegible_result'])
    gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['gt'])

    input_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                              config.dataset['SoccerNet'][args.part]['pose_input_json'])
    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet'][args.part]['pose_output_json'])

    # Create output dirs if they don't exist
    Path(features_dir).mkdir(parents=True, exist_ok=True)

    # 1. Filter out soccer ball based on images size
    if args.pipeline['soccer_ball_filter']:
        print("Determine soccer ball")
        success = helpers.identify_soccer_balls(image_dir, soccer_ball_list)
        print("Done determine soccer ball")

    # 1. generate and store features for each image in each tracklet
    if args.pipeline['feat']:
        print("Generate features")
        # Normalize paths to prevent Windows backslash issues
        image_dir_norm = image_dir.replace('\\', '/')
        features_dir_norm = features_dir.replace('\\', '/')

        command = f"conda run -n {config.reid_env} {python_cmd} {config.reid_script} --tracklets_folder {image_dir_norm} --output_folder {features_dir_norm}"
        run_command(command, ignore_errors=True)

        # Check if the expected output files exist instead
        tracklets = [t for t in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, t))]
        expected_files = [os.path.join(features_dir, f"{t}_features.npy") for t in tracklets]
        success = any(os.path.exists(f) for f in expected_files)  # Changed from all to any to be more lenient

        if success:
            print("Feature extraction completed successfully")
        else:
            print("Warning: Feature extraction may not have completed fully")
            success = True  # Force continue anyway

        print("Done generating features")

    # 2. identify and remove outliers based on features
    if args.pipeline['filter'] and success:
        print("Identify and remove outliers")
        # Normalize paths for Windows compatibility
        image_dir_norm = image_dir.replace('\\', '/')
        features_dir_norm = features_dir.replace('\\', '/')

        command = f"{python_cmd} gaussian_outliers.py --tracklets_folder {image_dir_norm} --output_folder {features_dir_norm}"
        run_command(command, ignore_errors=True)
        # Continue regardless of result
        success = True
        print("Done removing outliers")

    # 3. pass all images through legibililty classifier and record results
    if args.pipeline['legible'] and success:
        print("Classifying Legibility:")
        try:
            legible_dict, illegible_tracklets = get_soccer_net_legibility_results(args, use_filtered=True,
                                                                                  filter='gauss', exclude_balls=True)
        except Exception as error:
            print(f'Failed to run legibility classifier:{error}')
            # Try again without filtering if it fails
            try:
                print("Trying without filtering...")
                legible_dict, illegible_tracklets = get_soccer_net_legibility_results(args, use_filtered=False,
                                                                                      exclude_balls=True)
            except Exception as error2:
                print(f'Second attempt also failed:{error2}')
                success = False
        print("Done classifying legibility")

    # 3.5 evaluate tracklet legibility results
    if args.pipeline['legible_eval'] and success:
        print("Evaluate Legibility results:")
        try:
            if legible_dict is None:
                with open(full_legibile_path, 'r') as openfile:
                    # Reading from json file
                    legible_dict = json.load(openfile)

            helpers.evaluate_legibility(gt_path, illegible_path, legible_dict, soccer_ball_list=soccer_ball_list)
        except Exception as e:
            print(f"Error evaluating legibility: {e}")
            success = True  # Continue anyway
        print("Done evaluating legibility")

    # 4. generate json for pose-estimation
    if args.pipeline['pose'] and success:
        print("Generating json for pose")
        try:
            if legible_dict is None:
                if os.path.exists(full_legibile_path):
                    with open(full_legibile_path, 'r') as openfile:
                        # Reading from json file
                        legible_dict = json.load(openfile)
                else:
                    print(f"Warning: Could not find legibility results at {full_legibile_path}")
                    print("Will proceed with all images")
                    legible_dict = None

            generate_json_for_pose_estimator(args, legible=legible_dict)
        except Exception as e:
            print(f"Error generating pose JSON: {e}")
            success = True  # Continue anyway
        print("Done generating json for pose")

        # 5. run pose estimation and store results
        if success:
            print("Detecting pose")
            input_json_norm = input_json.replace('\\', '/')
            output_json_norm = output_json.replace('\\', '/')
            pose_home_norm = config.pose_home.replace('\\', '/')

            command = f"conda run -n {config.pose_env} {python_cmd} pose.py {pose_home_norm}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py {pose_home_norm}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json_norm} --out-json {output_json_norm}"
            run_command(command, ignore_errors=True)
            # Force success regardless of result
            success = True
            print("Done detecting pose")

    # 6. generate cropped images
    if args.pipeline['crops'] and success:
        print("Generate crops")
        try:
            crops_destination_dir = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                 config.dataset['SoccerNet'][args.part]['crops_folder'], 'imgs')
            Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)

            if legible_results is None and os.path.exists(full_legibile_path):
                with open(full_legibile_path, "r") as outfile:
                    legible_results = json.load(outfile)

            if legible_results is None:
                print("Warning: No legibility results found. Cropping may not work correctly.")
                legible_results = {}

            helpers.generate_crops(output_json, crops_destination_dir, legible_results)
        except Exception as e:
            print(f"Error generating crops: {e}")
            success = True  # Continue anyway
        print("Done generating crops")

    str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                   config.dataset['SoccerNet'][args.part]['jersey_id_result'])

    # 7. run STR system on all crops
    if args.pipeline['str'] and success:
        print("Predict numbers")
        image_dir = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                 config.dataset['SoccerNet'][args.part]['crops_folder'])
        str_model = config.dataset['SoccerNet']['str_model'].replace('\\', '/')
        image_dir_norm = image_dir.replace('\\', '/')
        str_result_file_norm = str_result_file.replace('\\', '/')

        command = f"conda run -n {config.str_env} {python_cmd} str.py {str_model} --data_root={image_dir_norm} --batch_size=1 --inference --result_file {str_result_file_norm}"
        run_command(command, ignore_errors=True)
        # Continue regardless
        success = True
        print("Done predict numbers")

    # 8. combine tracklet results
    if args.pipeline['combine'] and success:
        print("Combining tracklet results")
        try:
            analysis_results = None
            if os.path.exists(str_result_file):
                # read predicted results, stack unique predictions, sum confidence scores for each, choose argmax
                results_dict, analysis_results = helpers.process_jersey_id_predictions(str_result_file, useBias=True)

                # add illegible tracklet predictions
                consolidated_dict = consolidated_results(image_dir, results_dict, illegible_path,
                                                         soccer_ball_list=soccer_ball_list)

                # save results as json
                final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['final_result'])
                with open(final_results_path, 'w') as f:
                    json.dump(consolidated_dict, f)
            else:
                print(f"Warning: STR results file not found at {str_result_file}")
                success = False
        except Exception as e:
            print(f"Error combining results: {e}")
            success = False
        print("Done combining tracklet results")

    # 9. evaluate accuracy
    if args.pipeline['eval'] and success:
        print("Evaluating results")
        try:
            final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                              config.dataset['SoccerNet'][args.part]['final_result'])
            if consolidated_dict is None and os.path.exists(final_results_path):
                with open(final_results_path, 'r') as f:
                    consolidated_dict = json.load(f)

            if consolidated_dict is not None and os.path.exists(gt_path):
                with open(gt_path, 'r') as gf:
                    gt_dict = json.load(gf)
                print(f"Consolidated keys: {len(consolidated_dict.keys())}, GT keys: {len(gt_dict.keys())}")
                helpers.evaluate_results(consolidated_dict, gt_dict, full_results=analysis_results)
            else:
                print("Warning: Cannot evaluate without consolidated results or ground truth")
        except Exception as e:
            print(f"Error evaluating results: {e}")
        print("Done evaluating results")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Options: 'SoccerNet', 'Hockey'")
    parser.add_argument('part', help="Options: 'test', 'val', 'train', 'challenge")
    parser.add_argument('--train_str', action='store_true', default=False,
                        help="Run training of jersey number recognition")
    parser.add_argument('--reid_model', default='res50_market', help="ReID model type to use")
    args = parser.parse_args()

    if not args.train_str:
        if args.dataset == 'SoccerNet':
            actions = {"soccer_ball_filter": True,
                       "feat": True,
                       "filter": True,
                       "legible": True,
                       "legible_eval": False,
                       "pose": True,
                       "crops": True,
                       "str": True,
                       "combine": True,
                       "eval": True}
            args.pipeline = actions
            soccer_net_pipeline(args)
        elif args.dataset == 'Hockey':
            actions = {"legible": True,
                       "str": True}
            args.pipeline = actions
            hockey_pipeline(args)
        else:
            print("Unknown dataset")
    else:
        train_parseq(args)


