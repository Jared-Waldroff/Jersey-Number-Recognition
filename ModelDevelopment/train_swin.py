import os
import sys
import json
import time
import torch
import random

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def train_from_mapping(mapping_path=None, model_size='small', batch_size=32, num_workers=16,
                       epochs=5, learning_rate=5e-5, patience=5, weight_decay=0.05):
    """
    Train Swin Transformer V2 model on augmented data using the provided mapping

    Args:
        mapping_path: Path to the augmented mapping JSON (if None, uses default)
        model_size: Size of Swin V2 model ('tiny', 'small', 'base', 'large')
        batch_size: Batch size for training (smaller than ResNet due to transformer memory needs)
        num_workers: Number of data loading workers
        epochs: Maximum number of epochs
        learning_rate: Initial learning rate (lower for transformer fine-tuning)
        patience: Early stopping patience
        weight_decay: Weight decay parameter for AdamW optimizer

    Returns:
        history: Training history
    """
    start_time = time.time()
    print(f"Starting Swin Transformer V2 ({model_size}) training on augmented data...")

    try:
        from ModelDevelopment.SwinTransformerV2 import SwinTransformerV2

        # Determine default mapping path if not specified
        if mapping_path is None:
            mapping_path = os.path.join(project_root, "ModelDevelopment", "output", "augmented_tracklets.json")

        print(f"Using augmentation mapping from: {mapping_path}")

        # Check if mapping file exists
        if not os.path.exists(mapping_path):
            print(f"ERROR: Mapping file not found: {mapping_path}")
            print("Run augment_tracklets.py first to generate the mapping.")
            return None

        # Load the mapping
        with open(mapping_path, 'r') as f:
            serialized_mapping = json.load(f)

        # Convert the mapping (keys are strings, values are integers)
        augmented_mapping = {k: int(v) for k, v in serialized_mapping.items()}
        print(f"Loaded mapping with {len(augmented_mapping)} augmented tracklets")

        # Verify GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training using device: {device}")

        if device.type == 'cuda':
            # Print GPU info
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

        # Initialize and train the model
        model = SwinTransformerV2(model_size=model_size)
        history = model.train(
            augmented_mapping,
            batch_size=batch_size,
            num_workers=num_workers,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            weight_decay=weight_decay
        )

        print("Training completed successfully!")
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        return history

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run with default parameters
    train_from_mapping(batch_size=32, num_workers=16, model_size='small')