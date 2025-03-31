import os
import sys
import json
import time
import torch
import random

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def train_from_mapping(mapping_path=None, batch_size=64, num_workers=16, epochs=3, learning_rate=0.001, patience=3):
    """
    Train ResNet model on augmented data using the provided mapping

    Args:
        mapping_path: Path to the augmented mapping JSON (if None, uses default)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience

    Returns:
        history: Training history
    """
    start_time = time.time()
    print("Starting ResNet50 training on augmented data...")

    try:
        from ModelDevelopment.Resnet50 import ResNet50

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

        # # Sample a subset if dataset is too large (optional)
        # if len(augmented_mapping) > 150:
        #     sample_size = 150
        #     print(f"Sampling {sample_size} tracklets from {len(augmented_mapping)} total")
        #     keys = list(augmented_mapping.keys())
        #     sampled_keys = random.sample(keys, sample_size)
        #     augmented_mapping = {k: augmented_mapping[k] for k in sampled_keys}
        #     print(f"Reduced to {len(augmented_mapping)} tracklets")

        # Verify GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training using device: {device}")

        if device.type == 'cuda':
            # Print GPU info
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

        # Initialize and train the model
        model = ResNet50()
        history = model.train(
            augmented_mapping,
            batch_size=batch_size,
            num_workers=num_workers,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience
        )

        print("Training completed successfully!")
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        return history

    except Exception as e:
        print(f"Error during training: {e}")
        return None


if __name__ == "__main__":
    # Run with default parameters
    train_from_mapping(batch_size=64, num_workers=16)