from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from jersey_number_dataset_ResNet import JerseyNumberDataset, JerseyNumberMultitaskDataset
from networks import JerseyNumberClassifier, SimpleJerseyNumberClassifier, JerseyNumberMulticlassClassifier, FPNResNet34
import time
import copy
import argparse
import os
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import random_split
from networks import ResNetSE  # Import the SE-enhanced ResNet model
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

'''
############################################################################################################
Train ResNet model with the following code:

# Train the baseline ResNet model
python Train_ResNet_Model.py resnet34 --train --data out_train\SoccerNetResults\crops --weights models

# Train the multitask ResNet model
python Train_ResNet_Model.py resnet34_multi --train --data out_train\SoccerNetResults\crops --weights models

# Train the SE-Enhanced ResNet model
python Train_ResNet_Model.py resnetse --train --data out_train\SoccerNetResults\crops --weights models

# Train the FPN ResNet model
python Train_ResNet_Model.py fpn_resnet34 --train --data out_train\SoccerNetResults\crops --weights models

############################################################################################################
'''

'''
###########################################
Train function for FPN-Enhanced ResNet model
###########################################
'''
def train_fpn_model(model, optimizer, scheduler, num_epochs=10):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA Available:", torch.cuda.is_available())
    print("Using device:", device)

    # Move the model to the selected device
    model = model.to(device)

    # Get initial GPU memory
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9} GB")
    
    # Count the number of parameters in the model
    baseline_params = count_parameters(model)
    print(f"Model parameters: {baseline_params}")

    # Dataset loading
    train_img_dir = os.path.join('out_train', 'SoccerNetResults', 'crops', 'imgs')
    annotations_file_train = os.path.join('data', 'SoccerNet', 'train', 'train_gt.json')

    # Load the full training dataset (only returning full labels now)
    full_train_dataset = JerseyNumberDataset(annotations_file_train, train_img_dir, 'train')

    # Show sample images before training
    print("Displaying sample images from training dataset...")
    show_sample_images(full_train_dataset, num_images=10)

    # Define the size of the validation split (20% for validation)
    val_size = int(0.2 * len(full_train_dataset))  # 20% of dataset for validation
    train_size = len(full_train_dataset) - val_size  # Remaining 80% for training

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create dataloaders for both training and validation
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Wrap dataloader with tqdm to add progress bar
        for inputs, labels in tqdm(dataloader_train, desc=f"Epoch [{epoch}/{num_epochs-1}]", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch}/{num_epochs-1}], Loss: {running_loss / len(dataloader_train)}")

        # Step the scheduler after training each epoch
        scheduler.step()

        # Optionally, validate the model after every epoch
        validate_fpn_model(dataloader_val, model, criterion, device)

    # Save the trained model
    save_path = "models/fpn_resnet34_epoch10_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Final GPU memory usage after training
    print(f"GPU memory after training: {torch.cuda.memory_allocated() / 1e9} GB")

    # Calculate final parameter count
    enhanced_params = count_parameters(model)
    print(f"Final model parameters: {enhanced_params}")
    print(f"Parameter Efficiency: {enhanced_params / baseline_params:.4f}")

    # Calculate FLOPs
    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Adjust shape if needed
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"FLOPs: {flops.total()}")

    return model

def validate_fpn_model(val_loader, model, criterion, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%")
    
'''
###########################################
Train function for SE-Enhanced ResNet model
###########################################
'''
def train_squeeze_model(model, optimizer, scheduler, num_epochs=10):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA Available:", torch.cuda.is_available())
    print("Using device:", device)

    # Move the model to the selected device
    model = model.to(device)

    # Get initial GPU memory
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9} GB")
    
    # Count the number of parameters in the model
    baseline_params = count_parameters(model)
    print(f"Model parameters: {baseline_params}")

    # Dataset loading
    train_img_dir = os.path.join('out_train', 'SoccerNetResults', 'crops', 'imgs')
    annotations_file_train = os.path.join('data', 'SoccerNet', 'train', 'train_gt.json')

    # Load the full training dataset (only returning full labels now)
    full_train_dataset = JerseyNumberDataset(annotations_file_train, train_img_dir, 'train')

    # Show sample images before training
    print("Displaying sample images from training dataset...")
    show_sample_images(full_train_dataset, num_images=10)

    # Define the size of the validation split (20% for validation)
    val_size = int(0.2 * len(full_train_dataset))  # 20% of dataset for validation
    train_size = len(full_train_dataset) - val_size  # Remaining 80% for training

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create dataloaders for both training and validation
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Wrap dataloader with tqdm to add progress bar
        for inputs, labels in tqdm(dataloader_train, desc=f"Epoch [{epoch}/{num_epochs-1}]", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch}/{num_epochs-1}], Loss: {running_loss / len(dataloader_train)}")

        # Step the scheduler after training each epoch
        scheduler.step()

        # Optionally, validate the model after every epoch
        validate_squeeze_model(dataloader_val, model, criterion, device)

    # Save the trained model
    save_path = "models/squeeze_Resnet_epoch10_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Final GPU memory usage after training
    print(f"GPU memory after training: {torch.cuda.memory_allocated() / 1e9} GB")

    # Calculate final parameter count
    enhanced_params = count_parameters(model)
    print(f"Final model parameters: {enhanced_params}")
    print(f"Parameter Efficiency: {enhanced_params / baseline_params:.4f}")

    # Calculate FLOPs
    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Adjust shape if needed
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"FLOPs: {flops.total()}")

    return model

def validate_squeeze_model(val_loader, model, criterion, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%")

# Function to count the number of parameters in the model
def count_parameters(model):
    """Function to count the number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_sample_images(dataset, num_images=5):
    """Display a few sample images from the dataset."""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        img, label = dataset[i]  # Get image and label
        img = img.permute(1, 2, 0).cpu().numpy()  # Convert to NumPy format for display
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    
    plt.show()

'''
########################################
Train function for Baseline ResNet model
########################################
'''
def train_model(model, optimizer, scheduler, num_epochs=14):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA Available:", torch.cuda.is_available())
    print("Using device:", device)

    # Move the model to the selected device
    model = model.to(device)

    # Get initial GPU memory
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9} GB")
    
    # Count the number of parameters in the model
    baseline_params = count_parameters(model)
    print(f"Model parameters: {baseline_params}")

    # Directories and dataset loading
    train_img_dir = os.path.join('out_train', 'SoccerNetResults', 'crops', 'imgs')
    annotations_file_train = os.path.join('data', 'SoccerNet', 'train', 'train_gt.json')
    annotations_file_test = os.path.join('data', 'SoccerNet', 'test', 'test_gt.json')

    # Load the full training dataset (only returning full labels now)
    full_train_dataset = JerseyNumberDataset(annotations_file_train, train_img_dir, 'train')  # Updated dataset class

    # Show sample images before training
    print("Displaying sample images from training dataset...")
    show_sample_images(full_train_dataset, num_images=5)

    # Define the size of the validation split (20% validation, 80% training)
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(annotations_file_test, batch_size=4, shuffle=False, num_workers=4)

    dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        print(f'Model is on device: {next(model.parameters()).device}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()  # Evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}', ncols=100):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass and loss calculation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Model now returns only one output
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step the scheduler
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save best model based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    # Compute and print training time
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save trained model
    model_save_path = "models/baseline_ResNet_train_crops_epoch14_model.pth"
    torch.save(best_model_wts, model_save_path)
    print(f"Model saved at {model_save_path}")

    # Final GPU memory usage
    print(f"GPU memory after training: {torch.cuda.memory_allocated() / 1e9} GB")

    # Calculate final parameter count
    enhanced_params = count_parameters(model)
    print(f"Final model parameters: {enhanced_params}")
    print(f"Parameter Efficiency: {enhanced_params / baseline_params:.4f}")

    # Calculate FLOPs
    input_tensor = torch.randn(1, 3, 256, 256).to(device)  # Adjust shape if needed
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"FLOPs: {flops.total()}")

    return model

'''
#########################################
Train function for multitask ResNet model
#########################################
'''
ALPHA = 0.5
BETA = 0.25
GAMMA = 0.25

def train_multitask_model(model, optimizer, scheduler, num_epochs=10):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA Available:", torch.cuda.is_available())
    print("Using device:", device)

    # Move the model to the selected device
    model = model.to(device)

    # Get initial GPU memory
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9} GB")
    
    # Count the number of parameters in the baseline model
    baseline_params = count_parameters(model)
    print(f"Baseline model parameters: {baseline_params}")

    # Directories and dataset loading
    train_img_dir = os.path.join('data', 'SoccerNet', 'train', 'images')
    annotations_file_train = os.path.join('data', 'SoccerNet', 'train', 'train_gt.json')
    annotations_file_test = os.path.join('data', 'SoccerNet', 'test', 'test_gt.json')

    # Load the full training dataset
    full_train_dataset = JerseyNumberMultitaskDataset(annotations_file_train, train_img_dir, 'train')

    # Define the size of the validation split (e.g., 20% for validation)
    val_size = int(0.2 * len(full_train_dataset))  # 20% of the dataset for validation
    train_size = len(full_train_dataset) - val_size  # Remaining 80% for training

    # Split the dataset into train and validation
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create dataloaders for both training and validation
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(annotations_file_test, batch_size=4, shuffle=False, num_workers=4)

    dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}

    # Criterion and other variables
    criterion = nn.CrossEntropyLoss()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        print(f'Model is on device: {next(model.parameters()).device}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data with tqdm for progress bar
            for inputs, labels, digits1, digits2 in tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}', ncols=100):
                inputs = inputs.to(device)
                labels = labels.to(device)
                digits1 = digits1.to(device)
                digits2 = digits2.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass and loss calculation
                with torch.set_grad_enabled(phase == 'train'):
                    out1, out2, out3 = model(inputs)
                    _, preds = torch.max(out1, 1)
                    loss = ALPHA * criterion(out1, labels) + BETA * criterion(out2, digits1) + GAMMA * criterion(out3, digits2)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step the scheduler
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation accuracy is better
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    # Compute and print time elapsed for training
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Final model
    model.load_state_dict(best_model_wts)

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Define model save path
    model_save_path = "models/baseline_MultiResNet_model.pth"

    # Save the trained model
    torch.save(best_model_wts, model_save_path)
    print(f"Model saved successfully at {model_save_path}")

    # After training, get final GPU memory usage
    print(f"GPU memory after training: {torch.cuda.memory_allocated() / 1e9} GB")

    # Calculate and print parameter count again
    enhanced_params = count_parameters(model)
    print(f"Enhanced model parameters: {enhanced_params}")

    # Parameter Efficiency comparison
    print(f"Parameter Efficiency: {enhanced_params / baseline_params:.4f}")

    # Calculate FLOPs for the model
    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Modify this based on your input shape
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"FLOPs: {flops.total()}")

    return model

'''
############################################
Main function for ResNet model on validation
############################################
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--data', help='data root directory')
    parser.add_argument('--weights', help='path to model weights')
    parser.add_argument('--original_weights', help='path to model weights')
    parser.add_argument('model_type', help='resnet34, resnet34_multi, or resnetse')

    args = parser.parse_args()

    # Initialize the ResNet model based on the selected model type
    if args.simple:
        model_ft = SimpleJerseyNumberClassifier()
    elif args.model_type == 'resnet34':
        model_ft = JerseyNumberClassifier()
    elif args.model_type == 'resnet34_multi':
        model_ft = JerseyNumberMulticlassClassifier()
    elif args.model_type == 'resnetse':
        model_ft = ResNetSE() 
    elif args.model_type == 'fpn_resnet34':
        model_ft = FPNResNet34()  # Initialize FPNResNet34 model
    else:
        raise ValueError('Invalid model type specified.')

    if args.fine_tune:
        state_dict = torch.load(args.original_weights, map_location=device)

    if args.train:
        model_ft = model_ft.to(device)
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        if args.model_type == 'resnet34_multi':
            model_ft = train_multitask_model(model_ft, optimizer_ft, exp_lr_scheduler, num_epochs=10)
        elif args.model_type == 'resnetse':
            model_ft = train_squeeze_model(model_ft, optimizer_ft, exp_lr_scheduler, num_epochs=10)
        elif args.model_type == 'fpn_resnet34':
            model_ft = train_fpn_model(model_ft, optimizer_ft, exp_lr_scheduler, num_epochs=10)
        else:
            model_ft = train_model(model_ft, optimizer_ft, exp_lr_scheduler, num_epochs=10)
        torch.save(model_ft.state_dict(), args.weights)