import torch
import os

# Load the checkpoint
checkpoint_path = "C:/Users/jared/PycharmProjects/Jersey-Number-Recognition/data/pre_trained_models/clip4str/epoch=28-step=43292-val_accuracy=89.7051-val_NED=91.1516.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check what's in the checkpoint - this helps us understand what we have
print("Checkpoint keys:", checkpoint.keys())

# See if hyperparameters are stored
if 'hyper_parameters' in checkpoint:
    print("Found hyperparameters in checkpoint!")
    hyper_params = checkpoint['hyper_parameters']
    print(hyper_params)
else:
    print("No hyperparameters found in checkpoint")

# Create a simplified checkpoint that includes model config
output_path = "C:/Users/jared/PycharmProjects/Jersey-Number-Recognition/data/pre_trained_models/clip4str/jersey_model_fixed.pt"

# Create a simplified state dict with needed configuration
simplified_checkpoint = {
    'state_dict': checkpoint['state_dict'],
    'config': {
        'charset_train': "0123456789",
        'charset_test': "0123456789",
        'max_label_length': 3,
        'context_length': 2,
        'decoder_length': 3,
        'model_type': 'vl4str',
        'cross_gt_context': False
    }
}

# Save the modified checkpoint
torch.save(simplified_checkpoint, output_path)
print(f"Saved simplified model to {output_path}")