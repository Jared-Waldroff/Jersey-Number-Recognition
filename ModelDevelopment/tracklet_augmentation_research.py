import os
import sys
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the separated modules
from augment_tracklets import augment_tracklets
from train_resnet import train_from_mapping as train_resnet
from train_swin import train_from_mapping as train_swin


def main(run_augmentation=True, run_training=True, model_type='swin', model_size='small',
         aug_workers=8, train_workers=16, batch_size=32, epochs=5,
         learning_rate=5e-5, patience=5, weight_decay=0.05):
    """
    Run the complete tracklet augmentation research pipeline

    Args:
        run_augmentation: Whether to run the augmentation step
        run_training: Whether to run the training step
        model_type: Type of model to train ('resnet' or 'swin')
        model_size: Size of Swin model if using Swin ('tiny', 'small', 'base', 'large')
        aug_workers: Number of workers for augmentation
        train_workers: Number of workers for training data loading
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience
        weight_decay: Weight decay for optimizer (used for Swin)
    """
    start_time = time.time()
    mapping_path = None

    # Run augmentation if requested
    if run_augmentation:
        _, mapping_path = augment_tracklets(num_workers=aug_workers)
    else:
        # Use default mapping path
        mapping_path = os.path.join(project_root, "ModelDevelopment", "output", "augmented_tracklets.json")
        print(f"Skipping augmentation, using existing mapping: {mapping_path}")

    # Run training if requested
    if run_training:
        if model_type.lower() == 'resnet':
            print("Training with ResNet50 model")
            train_resnet(
                mapping_path=mapping_path,
                batch_size=batch_size if batch_size else 64,  # Default for ResNet
                num_workers=train_workers,
                epochs=epochs if epochs else 3,  # Default for ResNet
                learning_rate=learning_rate if learning_rate else 0.001,  # Default for ResNet
                patience=patience if patience else 3  # Default for ResNet
            )
        elif model_type.lower() == 'swin':
            print(f"Training with Swin Transformer V2 ({model_size}) model")
            train_swin(
                mapping_path=mapping_path,
                model_size=model_size,
                batch_size=batch_size if batch_size else 32,  # Default for Swin
                num_workers=train_workers,
                epochs=epochs if epochs else 5,  # Default for Swin
                learning_rate=learning_rate if learning_rate else 5e-5,  # Default for Swin
                patience=patience if patience else 5,  # Default for Swin
                weight_decay=weight_decay
            )
        else:
            print(f"Unknown model type: {model_type}. Choose 'resnet' or 'swin'.")
    else:
        print("Skipping training as requested")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main(
        run_augmentation=False,  # Skip augmentation
        model_type='swin',       # Use Swin Transformer V2
        model_size='small',      # Use small variant
        train_workers=4,         # Set number of data loading workers
        batch_size=16,           # Set batch size (smaller for transformer)
        epochs=2,                # Set number of epochs
        learning_rate=5e-5,      # Set learning rate
        patience=5               # Set early stopping patience
    )