import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
import torch
import sys


def comprehensive_checkpoint_diagnostic(model_path):
    print(f"Analyzing checkpoint: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")

    try:
        # Try loading with different methods
        print("\n1. Basic torch.load:")
        checkpoint = torch.load(model_path, map_location='cpu')

        print("\nCheckpoint keys:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                print(f"- {key}")

            # Check state dict keys if present
            if 'state_dict' in checkpoint:
                print("\nState Dict Keys:")
                state_dict = checkpoint['state_dict']
                visual_layers = [k for k in state_dict.keys() if 'visual' in k and 'transformer' in k]
                text_layers = [k for k in state_dict.keys() if 'transformer' in k]

                print("\nVisual Layers:")
                print(visual_layers)

                print("\nText Layers:")
                print(text_layers)
        else:
            print("Checkpoint is not a dictionary")

    except Exception as e:
        print(f"Error during loading: {e}")
        print(f"Exception type: {type(e)}")

        # Attempt to read file contents
        try:
            with open(model_path, 'rb') as f:
                first_mb = f.read(1024 * 1024)
                print("\nFirst MB of file contents (hex):")
                print(first_mb.hex())
        except Exception as read_e:
            print(f"Could not read file contents: {read_e}")


if __name__ == "__main__":
    model_path = r"C:\Users\jared\PycharmProjects\CLIP4STR\pretrained\clip\clip4str_huge_3e942729b1.pt"
    comprehensive_checkpoint_diagnostic(model_path)