import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# !/usr/bin/env python3
import os
import random
import argparse
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint path")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--visualize', type=int, default=0, help="Number of random samples to visualize")
    args = parser.parse_args()

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}")
    model = load_from_checkpoint(args.checkpoint).eval().to(args.device)
    hp = model.hparams

    print(f"Model charset: {hp.charset_test}")
    print(f"Max label length: {hp.max_label_length}")

    # Setup data module with your dataset
    datamodule = SceneTextDataModule(
        args.data_root, 'train', hp.img_size, hp.max_label_length,
        hp.charset_train, hp.charset_test, args.batch_size, args.num_workers, False
    )

    # Test on validation set only
    print("Loading validation data...")
    val_loader = datamodule.val_dataloader()

    # Evaluate the model
    total = 0
    correct = 0
    predictions = []

    print("Evaluating model...")
    for imgs, labels in tqdm(iter(val_loader), desc="Validation"):
        # Move images to the same device as model
        imgs = imgs.to(model.device)

        # Get predictions
        res = model.test_step((imgs, labels), -1)['output']

        # Store some prediction examples for visualization
        if args.visualize > 0 and len(predictions) < args.visualize:
            for i in range(min(len(labels), args.visualize - len(predictions))):
                pred = model.tokenizer.decode(model(imgs[i:i + 1]))[0]
                predictions.append((imgs[i].cpu(), labels[i], pred))

        total += res.num_samples
        correct += res.correct

    accuracy = 100 * correct / total
    print(f"\nValidation Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")

    # Visualize some random predictions if requested
    if args.visualize > 0 and predictions:
        print("\nSample predictions:")
        for i, (img, label, pred) in enumerate(predictions):
            print(f"Sample {i + 1}: True: {label}, Predicted: {pred}")


if __name__ == '__main__':
    main()