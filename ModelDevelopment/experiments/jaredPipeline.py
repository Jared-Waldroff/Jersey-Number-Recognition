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


def main():
    # First, get all tracks
    data_preprocessor = DataPreProcessing()
    tracks, max_track = data_preprocessor.get_tracks(DataPaths.TEST_DATA_DIR.value)

    # Randomly select a subset of tracks
    num_tracklets_to_process = 4  # Change this to the number of random tracklets you want
    random_tracklets = random.sample(tracks, num_tracklets_to_process)

    pipeline = CentralPipeline(
        tracklets_to_process_override=random_tracklets,  # change to ["34", "666", "1044", "1166"] format for specific tracklets
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
        use_cache=True,  # Set to false if you encounter data inconsistencies.
        suppress_logging=False,

        # --- PARALLELIZATION PARAMS --- These settings are optimal for an NVIDIA RTX 3070 Ti Laptop GPU.
        num_workers=1,  # CRITICAL optimisation param. Adjust accordingly. 6
        tracklet_batch_size=32,  # CRITICAL optimisation param. Adjust accordingly. 32
        image_batch_size=128,  # CRITICAL optimisation param. Adjust accordingly. 200
        num_threads_multiplier=1,
        # CRITICAL optimisation param. Adjust accordingly. 3. Interpretation: num_threads = num_workers * num_threads_multiplier
    )

    # Print out the randomly selected tracklets
    print("Processing the following randomly selected tracklets:")
    print(random_tracklets)

    pipeline.run_soccernet(
      run_soccer_ball_filter=True,
      generate_features=True,
      run_filter=True,
      run_legible=True,
      run_legible_eval=True,
      run_pose=True,
      run_crops=True,
      run_str=True,
      run_combine=True,
      run_eval=True,
      use_clip4str=True,
      pyscrippt=True)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
