#!/usr/bin/env python3
# CLIP4STR Scene Text Recognition
# Implementation based on original str.py but using CLIP4STR model with digit biasing for jersey numbers

import argparse
import string
import sys
from dataclasses import dataclass
from typing import List
from pathlib import Path
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F

# Adjust paths for CLIP4STR
ROOT = os.path.dirname(os.path.abspath(__file__))
# Add the root directory to the path to allow imports from strhub
sys.path.append(ROOT)

# First check if we can use CUDA
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}")

# Try different import paths based on what's available
try:
    # Try direct import first
    from strhub.data.module import SceneTextDataModule
    from strhub.models.utils import load_from_checkpoint, parse_model_args

    print("Using direct strhub imports")
except ImportError:
    try:
        # Try with str.CLIP4STR prefix
        from str.CLIP4STR.strhub.data.module import SceneTextDataModule
        from str.CLIP4STR.strhub.models.utils import load_from_checkpoint, parse_model_args
        from str.CLIP4STR.strhub.models.vl_str.system import VL4STR

        print("Using str.CLIP4STR strhub imports")
    except ImportError:
        # Try with just str prefix
        from str.strhub.data.module import SceneTextDataModule
        from str.strhub.models.utils import load_from_checkpoint, parse_model_args

        print("Using str.strhub imports")

from PIL import Image


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def process_image_str(filename, data_root, model, img_size, digit_indices=None):
    """
    Worker function to process a single image with CLIP4STR.
    Includes digit biasing for jersey numbers.
    """
    image_path = os.path.join(data_root, 'imgs', filename)
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

    # Get transformation from the SceneTextDataModule
    transform = SceneTextDataModule.get_transform(img_size)
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension

    # Run inference in no_grad mode for efficiency
    with torch.no_grad():
        # Forward pass with CLIP4STR model
        logits = model.forward(image.to(model.device))

        # Apply digit biasing like in read.py
        biased_logits = torch.ones_like(logits) * -1000.0

        # If we have digit indices, use them to bias toward digits
        if digit_indices:
            biased_logits[:, :, digit_indices] = logits[:, :, digit_indices]
        else:
            biased_logits = logits

        # Convert to probabilities
        probs = biased_logits.softmax(-1)

        # Decode predictions
        raw_pred, prob_values = model.tokenizer.decode(probs)
        jersey_number = raw_pred[0]

        # Ensure only digits (fallback)
        if not all(c.isdigit() for c in jersey_number):
            jersey_number = ''.join([c for c in jersey_number if c.isdigit()])
            if not jersey_number:
                jersey_number = "-1"

        # Extract confidence
        try:
            confidence = prob_values[0].cpu().detach().numpy().squeeze().tolist()
        except Exception:
            confidence = None

    return {filename: {
        'label': jersey_number,
        'raw_pred': raw_pred[0],
        'confidence': confidence,
        'logits': biased_logits.cpu().detach().numpy()[0].tolist() if biased_logits is not None else None
    }}


def run_inference(model, data_root, result_file, img_size):
    """
    Parallelized inference for STR with CLIP4STR.
    Includes digit biasing logic from read.py.
    """
    file_dir = os.path.join(data_root, 'imgs')
    filenames = sorted(os.listdir(file_dir))
    results = {}

    # Define digits that can be detected
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Find the indices of digits in the model's charset
    digit_indices = []
    if hasattr(model.tokenizer, 'charset'):
        charset = model.tokenizer.charset
        for i, char in enumerate(charset):
            if char in digits:
                # Account for special tokens in the vocabulary
                if hasattr(model, 'bos_id'):
                    digit_indices.append(i + 3)  # +3 for bos, eos, pad tokens
                else:
                    digit_indices.append(i)

    print(f"Found {len(digit_indices)} digit indices in charset: {digit_indices}")

    # Define number of worker threads (adjust as needed)
    num_workers = 8

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit a task for each filename.
        future_to_filename = {
            executor.submit(process_image_str, filename, data_root, model, img_size, digit_indices): filename
            for filename in filenames
        }
        for future in tqdm(as_completed(future_to_filename),
                           total=len(future_to_filename),
                           desc="CLIP4STR Inference"):
            filename = future_to_filename[future]
            try:
                res = future.result()
                if res is not None:
                    results.update(res)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Write results to JSON file.
    with open(result_file, 'w') as f:
        json.dump(results, f)

    print(f"Results saved to {result_file}")


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="CLIP4STR model checkpoint path")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--inference', action='store_true', default=False,
                        help='Run inference and store prediction results')
    parser.add_argument('--result_file', default='outputs/preds.json')
    parser.add_argument('--model_path', default=None, help="Explicit model path")
    parser.add_argument('--model_type', default=None, help="Model type")
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    # Configure charset based on arguments - for jersey numbers, default to digits only
    charset_test = string.digits
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    # If model_path is provided, use that instead of checkpoint
    model_path = args.model_path if args.model_path else args.checkpoint
    print(f"Loading CLIP4STR model from {model_path}")

    # Determine if we're loading a VL4STR model
    is_vl4str = False
    if args.model_type == 'vl4str' or "clip4str_huge" in model_path or "vl4str" in model_path:
        is_vl4str = True

    try:
        if is_vl4str:
            # Try different ways to load a VL4STR model
            try:
                from str.CLIP4STR.strhub.models.vl_str.system import VL4STR
                print(f"Loading as VL4STR model using str.CLIP4STR import path")
                model = VL4STR.load_from_checkpoint(model_path, **kwargs).eval().to(args.device)
            except (ImportError, AttributeError):
                try:
                    from strhub.models.vl_str.system import VL4STR
                    print(f"Loading as VL4STR model using direct import path")
                    model = VL4STR.load_from_checkpoint(model_path, **kwargs).eval().to(args.device)
                except (ImportError, AttributeError):
                    print(f"Falling back to generic load_from_checkpoint")
                    model = load_from_checkpoint(model_path, **kwargs).eval().to(args.device)
        else:
            model = load_from_checkpoint(model_path, **kwargs).eval().to(args.device)
    except RuntimeError as e:
        if "CUDA" in str(e) and args.device == 'cuda':
            print(f"CUDA error encountered. Falling back to CPU.")
            args.device = 'cpu'
            model = load_from_checkpoint(model_path, **kwargs).eval().to(args.device)
        else:
            raise

    hp = model.hparams

    # Run inference mode if specified
    if args.inference:
        print(f"Running inference on images in {args.data_root}")
        run_inference(model, args.data_root, args.result_file, hp.img_size)
        return

    # If not in inference mode, set up for evaluation
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, 2, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False)

    # Run evaluation on dataset
    test_set = ['JerseyNumbers']
    results = {}
    max_width = max(map(len, test_set))

    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0

        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length

        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)
        print(f"accuracy:{accuracy}, mean_conf:{mean_conf}")

    with open(model_path + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            print(f'Evaluation results:', file=out)
            print_results_table([results[s] for s in test_set], out)
            print('\n', file=out)


if __name__ == '__main__':
    main()