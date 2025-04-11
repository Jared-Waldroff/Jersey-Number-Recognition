#!/usr/bin/env python3
"""
SoccerNet Jersey Number Recognition Pipeline Runner
"""

import sys
import os
from pathlib import Path
import multiprocessing


def main():
    # Add parent directory to path to ensure imports work properly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    print(f"Current working directory: {os.getcwd()}")
    print(f"Added to path: {parent_dir}")

    try:
        from ModelDevelopment.CentralPipeline import CentralPipeline
        from DataProcessing.DataPreProcessing import DataPaths
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please make sure you're running this script from the correct directory.")
        sys.exit(1)

    # Define which tracklets to process
    # tracklets_to_process = [i for i in range(0, 712)]  # First batch
    # tracklets_to_process = [i for i in range(0, 1210)]
    tracklets_to_process = [i for i in range(0, 1210)] # back up batch
    num_tracklets = 1426

    # Initialize the pipeline
    pipeline = CentralPipeline(
        tracklets_to_process_override=tracklets_to_process,
        num_tracklets = num_tracklets,
        input_data_path=DataPaths.TEST_DATA_DIR.value,
        output_processed_data_path=DataPaths.PROCESSED_DATA_OUTPUT_DIR_TEST.value,  # Change this
        common_processed_data_dir=DataPaths.COMMON_PROCESSED_OUTPUT_DATA_TEST.value,  # Change this
        gt_data_path=DataPaths.TEST_DATA_GT.value,  # Replace with actual path
        display_transformed_image_sample=False,
        use_cache=False,
        suppress_logging=False,
        use_image_enhancement=False,

        # Parallelization parameters - optimized for GPU
        num_workers=2,
        tracklet_batch_size=24,
        image_batch_size=24,
        num_threads_multiplier=2
    )

    # Run the SoccerNet pipeline
    pipeline.run_soccernet(
        run_soccer_ball_filter=False,
        generate_features=False,
        run_filter=False,
        run_legible=False,
        run_legible_eval=False,
        run_pose=False,
        run_crops=False,
        run_str=True,
        run_combine=True,
        run_eval=True,
        use_clip4str=True,
        pyscript=True,
    )

    print("Pipeline processing complete!")


if __name__ == "__main__":
    # This is critical for Windows multiprocessing
    multiprocessing.freeze_support()

    # Call main function - this prevents multiprocessing issues on Windows
    main()
