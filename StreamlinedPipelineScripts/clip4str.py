#!/usr/bin/env python3
# CLIP4STR processing module
# Handles processing output from the CLIP4STR model

import os
os.environ["MPLBACKEND"] = "Agg"

import json
import sys
import subprocess
from pathlib import Path
import shutil
import concurrent.futures
from tqdm import tqdm
import threading

#GPU_SEMAPHORE = threading.Semaphore(value=1)


def parse_output(stdout_text):
    """
    Parse the output from the CLIP4STR model.
    First tries to extract JSON results, then falls back to text parsing if needed.

    Args:
        stdout_text (str): The stdout output from running the CLIP4STR model

    Returns:
        dict: A dictionary of results in the format expected by process_jersey_id_predictions
    """
    # Try to extract JSON results first
    json_start = stdout_text.find("JSON_RESULTS_BEGIN")
    json_end = stdout_text.find("JSON_RESULTS_END")

    if json_start >= 0 and json_end > json_start:
        # Extract JSON text
        json_text = stdout_text[json_start + len("JSON_RESULTS_BEGIN"):json_end].strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output: {e}. Falling back to text parsing.")

    # Fall back to original parsing if no JSON found or if JSON parsing failed
    results_dict = {}

    for line in stdout_text.splitlines():
        # Only process lines that look like image file predictions
        if line and ('.jpg:' in line or '.jpeg:' in line or '.png:' in line):
            try:
                parts = line.split(':', 1)
                filename = parts[0].strip()
                rest = parts[1].strip()

                # Extract confidence if present in the new format
                confidence_array = None
                if "→ Confidence:" in rest:
                    conf_part = rest.split("→ Confidence:", 1)[1].strip()
                    try:
                        # Try to parse as list
                        confidence_array = json.loads(conf_part)
                    except json.JSONDecodeError:
                        # If that fails, use old default
                        pass

                # Extract jersey number - the part after "Jersey Number:" if it exists
                jersey_number = "-1"  # Default
                if "→ Jersey Number:" in rest:
                    jersey_part = rest.split("→ Jersey Number:", 1)[1]
                    if "→" in jersey_part:
                        jersey_number = jersey_part.split("→", 1)[0].strip()
                    else:
                        jersey_number = jersey_part.strip()
                else:
                    # Fallback - just extract digits
                    digits = ''.join([c for c in rest if c.isdigit()])
                    if digits:
                        jersey_number = digits

                # Generate confidence values if not extracted above
                if confidence_array is None:
                    if jersey_number.isdigit() and jersey_number != "-1":
                        # Create an array with 0.9 confidence for each character
                        confidence_array = [0.9] * len(jersey_number)
                    else:
                        # For invalid predictions, use a single low confidence value
                        confidence_array = [0.1]

                # Store in the format expected by process_jersey_id_predictions
                results_dict[filename] = {
                    "label": jersey_number,
                    "confidence": confidence_array
                }

            except Exception as e:
                print(f"Error parsing line: {line}, error: {e}")

    return results_dict


def log_sample_results(results_dict, logger, sample_count=5):
    """
    Log only the count of results, not the samples themselves.
    """
    if results_dict:
        logger.info(f"Successfully processed {len(results_dict)} predictions")
    else:
        logger.warning("No results were produced by the model")


def run_clip4str_inference(python_path, read_script_path, model_path, clip_pretrained_path,
                           images_dir, result_file, logger, env=None, batch_size=50):
    """
    Run the CLIP4STR model for inference in batches.
    
    Args:
        python_path (str): Path to the Python executable.
        read_script_path (str): Path to the read.py script.
        model_path (str): Path to the CLIP4STR model.
        clip_pretrained_path (str): Path to the pretrained CLIP model.
        images_dir (str): Path to the directory containing images.
        result_file (str): Path to save the aggregated results.
        logger: Logger for output.
        env (dict): Environment variables.
        batch_size (int): Number of images to process per batch.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if env is None:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Get all image files in the provided directory.
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        logger.error(f"No image files found in {images_dir}")
        return False

    # Split images into batches.
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    logger.info(f"Processing {len(image_files)} images in {len(batches)} batches (batch size = {batch_size})")

    aggregated_results = {}

    # Process each batch sequentially.
    for i, batch in enumerate(tqdm(batches, desc="Processing batches", unit="batch")):
        # Create a temporary directory for the current batch.
        batch_temp_dir = os.path.join(images_dir, f"temp_batch_{i}")
        os.makedirs(batch_temp_dir, exist_ok=True)

        # Copy the batch images to the temporary directory.
        for img in batch:
            shutil.copy(os.path.join(images_dir, img), os.path.join(batch_temp_dir, img))
        
        # Build the command for inference on the current batch.
        command = [
            python_path,
            read_script_path,
            model_path,
            f"--images_path={batch_temp_dir}",
            "--device=cuda",
            f"--clip_pretrained={clip_pretrained_path}"
        ]
        logger.info(f"Batch {i}: Running command: {' '.join(command)}")

        try:
            result = subprocess.run(command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    env=env)
        except subprocess.CalledProcessError as e:
            logger.error(f"Batch {i}: Error running inference: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            # Clean up and continue with the next batch.
            shutil.rmtree(batch_temp_dir, ignore_errors=True)
            continue

        # Parse the output using your existing parse_output function.
        batch_results = parse_output(result.stdout)
        aggregated_results.update(batch_results)
        
        # Clean up the temporary batch directory.
        shutil.rmtree(batch_temp_dir, ignore_errors=True)
        
        # Optionally, free up memory explicitly.
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    # Ensure the directory for result_file exists and save the aggregated results.
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)

    logger.info(f"Saved {len(aggregated_results)} results to {result_file}")
    return True


def run_parallel_clip4str_inference(python_path, read_script_path, model_path, clip_pretrained_path,
                                    images_dir, result_file, logger, num_workers=4, batch_size=50,
                                    env=None, gpu_semaphore=None):
    """Parallel version of CLIP4STR inference with GPU resource management"""

    # 1. Get all image files - this can happen outside the GPU lock
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 2. Split into batches - use smaller batches for better memory management
    #adjusted_batch_size = min(batch_size, 32)  # Cap at 32 images per batch for memory safety
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

    logger.info(f"Processing {len(image_files)} images in {len(batches)} batches of {batch_size}")

    # 3. Create temp directories for each batch - still no need for GPU
    temp_dirs = [os.path.join(os.path.dirname(result_file), f"clip4str_temp_{i}")
                 for i in range(len(batches))]
    for temp_dir in temp_dirs:
        os.makedirs(temp_dir, exist_ok=True)

    # 4. Define worker function that applies the GPU semaphore only around inference
    def process_batch(batch_idx, image_batch, temp_dir):
        # Create temp image directory (non-GPU operation)
        batch_img_dir = os.path.join(temp_dir, "imgs")
        os.makedirs(batch_img_dir, exist_ok=True)

        # Copy images to batch directory (non-GPU operation)
        for img in image_batch:
            shutil.copy(os.path.join(images_dir, img), os.path.join(batch_img_dir, img))

        # Prepare for inference
        batch_result_file = os.path.join(temp_dir, "results.json")

        # Only lock the GPU during the actual model inference
        with gpu_semaphore:
            logger.info(f"Batch {batch_idx}: Acquired GPU lock, running inference on {len(image_batch)} images")
            success = run_clip4str_inference(
                python_path=python_path,
                read_script_path=read_script_path,
                model_path=model_path,
                clip_pretrained_path=clip_pretrained_path,
                images_dir=batch_img_dir,
                result_file=batch_result_file,
                logger=logger,
                env=env
            )
            logger.info(f"Batch {batch_idx}: Released GPU lock")

        # Process results after releasing the lock (non-GPU operation)
        if success and os.path.exists(batch_result_file):
            with open(batch_result_file, 'r') as f:
                return json.load(f)
        return {}

    # 5. Process batches in parallel with managed GPU access
    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, i, batch, temp_dirs[i]): i
            for i, batch in enumerate(batches)
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_batch),
                           total=len(batches), desc="Processing CLIP4STR batches"):
            batch_results = future.result()
            results_dict.update(batch_results)
            # Manual cleanup after each batch completes
            import gc
            gc.collect()
            if torch_available:
                import torch
                torch.cuda.empty_cache()  # Explicitly clear CUDA cache

    # 6. Save combined results (non-GPU operation)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # 7. Clean up temp directories (non-GPU operation)
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return True


# Check if torch is available for memory management
torch_available = False
try:
    import torch

    torch_available = True
except ImportError:
    pass


if __name__ == "__main__":
    # This is for testing the module directly
    import argparse
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run CLIP4STR inference")
    parser.add_argument("--model", required=True, help="Path to CLIP4STR model")
    parser.add_argument("--clip_pretrained", required=True, help="Path to pretrained CLIP model")
    parser.add_argument("--images_dir", required=True, help="Path to images directory")
    parser.add_argument("--result_file", required=True, help="Path to save results")
    parser.add_argument("--python_path", help="Path to Python executable")
    args = parser.parse_args()

    # Use system Python if not specified
    python_path = args.python_path or sys.executable

    # Get path to read.py in the same directory as the model
    model_dir = os.path.dirname(args.model)
    read_script_path = os.path.join(os.path.dirname(model_dir), "read.py")

    # Run inference
    success = run_clip4str_inference(
        python_path=python_path,
        read_script_path=read_script_path,
        model_path=args.model,
        clip_pretrained_path=args.clip_pretrained,
        images_dir=args.images_dir,
        result_file=args.result_file,
        logger=logger
    )

    sys.exit(0 if success else 1)