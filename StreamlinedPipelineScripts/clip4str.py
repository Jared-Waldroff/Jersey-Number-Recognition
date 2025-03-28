#!/usr/bin/env python3
# CLIP4STR processing module
# Handles processing output from the CLIP4STR model

import os
import json
import sys
import subprocess
from pathlib import Path


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
                           images_dir, result_file, logger, env=None):
    """
    Run the CLIP4STR model for inference.

    Args:
        python_path (str): Path to the Python executable
        read_script_path (str): Path to the read.py script
        model_path (str): Path to the CLIP4STR model
        clip_pretrained_path (str): Path to the pretrained CLIP model
        images_dir (str): Path to the directory containing images
        result_file (str): Path to save the results
        logger: Logger for output
        env (dict): Environment variables

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(model_path):
        logger.error(f"CLIP4STR model not found: {model_path}")
        return False

    if not os.path.exists(clip_pretrained_path):
        logger.error(f"Pretrained CLIP model not found: {clip_pretrained_path}")
        return False

    # Set default environment if none provided
    if env is None:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Command to run the read.py script
    command = [
        python_path,
        read_script_path,
        model_path,
        f"--images_path={images_dir}",
        "--device=cuda",
        f"--clip_pretrained={clip_pretrained_path}"
    ]

    logger.info(f"Running command: {' '.join(command)}")

    try:
        # Capture stdout but don't display it
        result = subprocess.run(command,
                                stdout=subprocess.PIPE,  # Capture but don't print
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True,
                                encoding='utf-8',
                                errors='replace',
                                env=env)

        # Only log the stderr (warnings/errors) if any
        if result.stderr:
            logger.error(result.stderr)

        # Parse the output (we need result.stdout for this)
        results_dict = parse_output(result.stdout)

        # Make sure the output directory exists
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        # Save results as JSON
        with open(result_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Saved {len(results_dict)} jersey number predictions to {result_file}")

        # Log a sample of results
        log_sample_results(results_dict, logger)

        # We've removed this to keep terminal clean
        # logger.info(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running CLIP4STR model: {e}")
        logger.info(f"STDOUT: {e.stdout}" if e.stdout else "No stdout output")
        logger.error(f"STDERR: {e.stderr}" if e.stderr else "No stderr output")

        # Create an empty results file if it doesn't exist
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump({}, f)
            logger.warning(f"Created empty results file: {result_file}")

        return False


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