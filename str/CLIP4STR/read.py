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
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
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

    for fname in files:
        # Load image and prepare for input
        filename = os.path.join(args.images_path, fname)
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
        raw_pred, _ = model.tokenizer.decode(probs)
        jersey_number = raw_pred[0]

        # Ensure only digits (fallback)
        if not all(c.isdigit() for c in jersey_number):
            jersey_number = ''.join([c for c in jersey_number if c.isdigit()])
            if not jersey_number:
                jersey_number = "-1"

        print(f'{fname}: {raw_pred} → Jersey Number: {jersey_number}')


if __name__ == '__main__':
    main()
