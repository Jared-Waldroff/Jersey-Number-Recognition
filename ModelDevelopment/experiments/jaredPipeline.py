import sys
from pathlib import Path
import os
import random

sys.path.append(str(Path.cwd().parent.parent))
print(str(Path.cwd().parent.parent))
print("Current working directory: ", os.getcwd())

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ModelDevelopment.CentralPipeline import CentralPipeline
from ModelDevelopment.ImageBatchPipeline import ImageBatchPipeline
from DataProcessing.DataPreProcessing import DataPaths, DataPreProcessing

# Configuration variables
USE_RANDOM_TRACKLETS = False  # Set to True to use random tracklets, False to process all tracklets
NUM_RANDOM_TRACKLETS = 4  # Number of random tracklets to process if USE_RANDOM_TRACKLETS is True

# Optional: Specify specific tracklets to process if you want a fixed set
SPECIFIC_TRACKLETS = []  # Example tracklet IDs

def main():
    # First, get all tracks
    data_preprocessor = DataPreProcessing()
    tracks, max_track = data_preprocessor.get_tracks(DataPaths.TEST_DATA_DIR.value)

    # Determine which tracklets to process
    if USE_RANDOM_TRACKLETS:
        # Randomly select a subset of tracks
        tracklets_to_process = random.sample(tracks, NUM_RANDOM_TRACKLETS)
        print("Processing the following randomly selected tracklets:")
        print(tracklets_to_process)
    elif SPECIFIC_TRACKLETS:
        # Use predefined specific tracklets
        tracklets_to_process = SPECIFIC_TRACKLETS
        print("Processing the following specific tracklets:")
        print(tracklets_to_process)
    else:
        # Process all tracklets
        tracklets_to_process = tracks
        print("Processing ALL tracklets")

    pipeline = CentralPipeline(
        tracklets_to_process_override=tracklets_to_process,  # change to ["34", "666", "1044", "1166"] format for specific tracklets
        # num_tracklets=3,
        # num_images_per_tracklet=50,
        input_data_path=DataPaths.TEST_DATA_DIR.value,
        output_processed_data_path=DataPaths.PROCESSED_DATA_OUTPUT_DIR_TEST.value,
        common_processed_data_dir=DataPaths.COMMON_PROCESSED_OUTPUT_DATA_TEST.value,
        gt_data_path=DataPaths.TEST_DATA_GT.value,
        single_image_pipeline=False,
        display_transformed_image_sample=False,
        # NOTE: DO NOT USE. Code is parallelized so we cannot show images anymore. Code breaks, but first one will show if True.
        num_image_samples=1,
        use_cache=False,  # Set to false if you encounter data inconsistencies.
        suppress_logging=False,

        # --- PARALLELIZATION PARAMS --- These settings are optimal for an NVIDIA RTX 3070 Ti Laptop GPU.
        num_workers=2,  # CRITICAL optimisation param. Adjust accordingly. 6
        tracklet_batch_size=16,  # CRITICAL optimisation param. Adjust accordingly. 32
        image_batch_size=24,  # CRITICAL optimisation param. Adjust accordingly. 200
        num_threads_multiplier=1,
        # CRITICAL optimisation param. Adjust accordingly. 3. Interpretation: num_threads = num_workers * num_threads_multiplier
    )

    pipeline.run_soccernet(
      run_soccer_ball_filter=False,
      generate_features=False,
      run_filter=False,
      run_legible=False,
      run_legible_eval=False,
      run_pose=True,
      run_crops=True,
      run_str=False,
      run_combine=False,
      run_eval=False,
      use_clip4str=False,
      pyscrippt=True)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
