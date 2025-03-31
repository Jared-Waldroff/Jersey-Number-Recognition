import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
from PIL import Image
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class JerseyDataset(Dataset):
    """Dataset for jersey number images with caching to speed up repeats"""

    def __init__(self, dir_to_jersey_mapping, transform=None, cache_size=500):
        self.transform = transform
        self.samples = []
        self.cache = {}  # Cache for already loaded images
        self.cache_size = cache_size

        # Create list of samples with their jersey numbers
        for img_dir, jersey_num in dir_to_jersey_mapping.items():
            # Get all images in this directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(img_dir, ext)))

            # Add each image with its jersey number
            for img_path in image_files:
                self.samples.append((img_path, jersey_num))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Check if this image is already in cache
        if idx in self.cache:
            return self.cache[idx]

        img_path, jersey_num = self.samples[idx]

        try:
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            # Store in cache if not full
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (img, jersey_num)

            return img, jersey_num
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return empty tensor on error with correct dtype
            return torch.zeros((3, 224, 224), dtype=torch.float32), jersey_num


class ResNet50:
    def __init__(self):
        """Initialize ResNet50 model for jersey number classification"""
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Enhanced preprocessing for ResNet with more CPU-intensive operations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize larger then crop for better quality
            transforms.RandomCrop((224, 224)),  # Random crop for more variation
            transforms.RandomHorizontalFlip(p=0.3),  # Flipping augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color augmentation
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Geometric augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Model will be initialized during training
        self.model = None

    def _build_model(self, num_classes):
        """Build ResNet50 model with given number of output classes"""
        # Load pretrained model
        model = torchvision.models.resnet50(weights='DEFAULT')

        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        return model.to(self.device)

    def validate(self, val_loader, criterion):
        """Run validation on the model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Set back to training mode
        self.model.train()

        return avg_val_loss, val_accuracy

    def train(self, dir_to_jersey_mapping, batch_size=64, num_workers=16, epochs=3, learning_rate=0.001, patience=3):
        """Train the ResNet50 model on augmented data"""
        # Verify GPU is being used
        print(f"Training on device: {self.device}")
        if self.device.type == 'cuda':
            # Print GPU memory info before training
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
            print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

            # Empty cache to ensure maximum available memory
            torch.cuda.empty_cache()

        # Initialize mixed precision training
        scaler = GradScaler()

        # Create dataset from directory mapping
        dataset = JerseyDataset(dir_to_jersey_mapping, transform=self.transform)

        if len(dataset) == 0:
            print("No training data found! Aborting.")
            return

        # Split into train/validation sets
        val_size = int(0.2 * len(dataset))  # 20% for validation
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print(f"Split dataset: {train_size} training samples, {val_size} validation samples")

        # Find maximum jersey number to determine num_classes
        max_jersey = max([jersey for _, jersey in dataset.samples])
        num_classes = max_jersey + 1  # +1 because jersey numbers start at 0

        print(f"Training with {len(dataset)} images across {len(set([l for _, l in dataset.samples]))} jersey numbers")
        print(f"Setting up model with {num_classes} classes to accommodate maximum jersey number {max_jersey}")

        # Build model
        self.model = self._build_model(num_classes)

        # Setup data loaders with optimized parallel processing
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Use larger batches for validation
            shuffle=False,
            num_workers=num_workers,
        )

        print(f"DataLoader configured with {num_workers} workers, batch size {batch_size}")
        print(f"Training on {train_size} images ({len(train_loader)} batches)")
        print(f"Validating on {val_size} images ({len(val_loader)} batches)")

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler based on validation loss
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # For saving best model
        model_dir = 'D:\\ModelCheckpoints'
        os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
        best_model_path = os.path.join(model_dir, "best_resnet50_jersey.pth")

        # Training loop
        self.model.train()
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(epochs):
            # Track epoch time
            epoch_start_time = time.time()

            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Use tqdm for progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for inputs, labels in progress_bar:
                # Move tensors to device with non-blocking transfers
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Track statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': running_loss / (progress_bar.n + 1),
                    'accuracy': f"{100 * correct / total:.2f}%"
                })

            # Calculate epoch training statistics
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_accuracy = 100 * correct / total

            # Save to history
            history['train_loss'].append(epoch_train_loss)
            history['train_accuracy'].append(epoch_train_accuracy)

            # Validation phase
            val_loss, val_accuracy = self.validate(val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            # Update learning rate based on validation loss
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")

            # Early stopping and model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epochs")

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Print GPU memory usage after each epoch
            if self.device.type == 'cuda':
                print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

        # Load best model for final save
        self.model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")

        # Save final model
        final_model_path = os.path.join(model_dir, "resnet50_jersey_classifier.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

        # Plot training history
        self._plot_training_history(history)

        return history

    def _plot_training_history(self, history):
        """Plot and save training history"""
        plt.figure(figsize=(15, 6))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Training')
        plt.plot(history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save plot
        model_dir = 'D:\\ModelCheckpoints'
        plot_path = os.path.join(model_dir, "training_history.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Training plot saved to {plot_path}")

    def predict(self, img_path):
        """Predict jersey number from an image"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    def load_model(self, model_path=None, num_classes=100):
        """Load a trained model from file"""
        if model_path is None:
            # Default to the saved path on the other drive
            model_path = os.path.join('D:\\ModelCheckpoints', "resnet50_jersey_classifier.pth")

        self.model = self._build_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")