import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from strhub.models.vl_str.system import VL4STR

# Load original CLIP weights
clip_weights_path = "C:/Users/jared/PycharmProjects/Jersey-Number-Recognition/data/pre_trained_models/clip4str/ViT-B-16.pt"
clip_weights = torch.load(clip_weights_path, map_location='cpu')

# Initialize the model with the SAME context_length as the original CLIP model
model = VL4STR(
    # Jersey number specific params
    charset_train="0123456789",
    charset_test="0123456789",
    max_label_length=3,
    # Critical: match original model's dimensions
    context_length=26,  # MUST match original CLIP model
    decode_ar=True,
    refine_iters=1,
    # Architecture params
    batch_size=32,
    lr=0.0001,
    warmup_pct=0.1,
    weight_decay=0.0005,
    img_size=[224, 224],
    patch_size=[16, 16],
    embed_dim=512,
    enc_num_heads=12,
    enc_mlp_ratio=4,
    enc_depth=12,
    dec_num_heads=8,
    dec_mlp_ratio=4,
    dec_depth=1,
    perm_num=6,
    perm_forward=True,
    perm_mirrored=True,
    dropout=0.1,
    # Only fine-tune the decoder
    image_freeze_nlayer=12,  # Freeze all visual encoder
    text_freeze_nlayer=12,   # Freeze all text encoder
    clip_pretrained=clip_weights_path
)

# Save initialized model that can be loaded during training
torch.save(model.state_dict(), "initialized_jersey_model.pt")