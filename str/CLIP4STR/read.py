#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json  # Add this import
import sys

import torch

from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images_path', type=str, help='Images to read')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--clip_pretrained', type=str, help='Path to pretrained CLIP model')
    args, unknown = parser.parse_known_args()

    # Add clip_pretrained to kwargs if provided
    kwargs = parse_model_args(unknown)
    if args.clip_pretrained:
        kwargs['clip_pretrained'] = args.clip_pretrained

    print(f'Additional keyword arguments: {kwargs}')

    try:
        # Check what's in the checkpoint
        print("Loading checkpoint from", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Found state_dict in checkpoint")
        else:
            print("WARNING: Checkpoint format may not be compatible")
    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")

    # Replace the load_from_checkpoint call with this code
    try:
        # Load the checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model = load_from_checkpoint(
            args.checkpoint,
            name="vl4str",  # Explicitly set the model name
            charset_train="0123456789",
            charset_test="0123456789",
            max_label_length=25,  # Critical - must match checkpoint value
            context_length=26,
            decoder_length=26,
            batch_size=32,
            img_size=[224, 224],
            patch_size=[16, 16],
            embed_dim=512,
            enc_num_heads=12,
            enc_mlp_ratio=4,
            enc_depth=12,
            enc_width=768,
            dec_num_heads=8,
            dec_mlp_ratio=4,
            dec_depth=1,
            perm_num=6,
            perm_forward=True,
            perm_mirrored=True,
            decode_ar=True,
            refine_iters=1,
            dropout=0.1,
            # Include all parameters from your checkpoint
            use_language_model=True,
            freeze_backbone=True,
            freeze_language_backbone=True,
            freeze_image_backbone=True,
            cross_gt_context=True,
            cross_loss_weight=0.5,
            use_share_dim=True,
            image_detach=True,
            **kwargs
        ).eval().to(args.device)

        # Load with non-strict matching
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model with {len(missing)} missing keys and {len(unexpected)} unexpected keys")

        model = model.eval().to(args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    files = sorted(
        [x for x in os.listdir(args.images_path) if x.endswith('png') or x.endswith('jpeg') or x.endswith('jpg')])

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

    # Create a dictionary to store all results
    results_dict = {}

    for fname in files:
        # Load image and prepare for input
        filename = os.path.join(args.images_path, fname)
        try:
            image = Image.open(filename).convert('RGB')
            image = img_transform(image).unsqueeze(0).to(args.device)

            # Get raw logits from model
            logits = model(image)

            # Create a biased version that only allows digits
            biased_logits = torch.ones_like(logits) * -1000.0

            # If we find digits, use them
            if digit_indices:
                biased_logits[:, :, digit_indices] = logits[:, :, digit_indices]
            else:
                biased_logits = logits

            # Convert to probabilities
            probs = biased_logits.softmax(-1)

            # Get raw prediction
            raw_pred, confidences = model.tokenizer.decode(probs)
            jersey_number = raw_pred[0]

            # Only keep confidence scores for the actual digits
            if jersey_number.isdigit() and len(jersey_number) <= 2:
                confidences[0] = confidences[0][:len(jersey_number)]

            # Ensure only digits (fallback)
            if not all(c.isdigit() for c in jersey_number):
                jersey_number = ''.join([c for c in jersey_number if c.isdigit()])
                if not jersey_number:
                    jersey_number = "-1"

            # Still print the original info for debugging
            print(f'{fname}: {raw_pred[0]} → Jersey Number: {jersey_number} → Confidence: {confidences[0].tolist()}')

            # Store result in our dictionary
            results_dict[fname] = {
                "label": jersey_number,
                "confidence": confidences[0].tolist()
            }

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            # Add failed entry to results with error message
            results_dict[fname] = {
                "label": "-1",
                "confidence": [0.0],
                "error": str(e)
            }

    # Print the entire results dictionary as JSON at the end
    # This makes it easier for clip4str.py to parse
    print("\nJSON_RESULTS_BEGIN")
    print(json.dumps(results_dict, indent=2))
    print("JSON_RESULTS_END")


if __name__ == '__main__':
    main()