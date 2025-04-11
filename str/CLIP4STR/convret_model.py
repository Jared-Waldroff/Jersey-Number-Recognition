import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from strhub.models.utils import load_from_checkpoint

# Path to your checkpoint - use raw string (r prefix) to avoid escape character issues
ckpt_path = r'C:\Users\jared\PycharmProjects\CLIP4STR\output\vl4str_2025-04-03_10-50-56\checkpoints\epoch=19-step=14792-val_accuracy=78.1514-val_NED=81.6551.ckpt'

# Load the model
model = load_from_checkpoint(ckpt_path)

# To save as a .pt file
torch.save(model.state_dict(), 'epoch=19-step=14792-val_accuracy=78.1514-val_NED=81.6551.pt')