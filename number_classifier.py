from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from jersey_number_dataset import JerseyNumberDataset, JerseyNumberMultitaskDataset
from networks import JerseyNumberClassifier, SimpleJerseyNumberClassifier, JerseyNumberMulticlassClassifier

import time
import copy
import argparse
import os
'''
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #print(f"input and label sizes:{len(inputs), len(labels)}")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(f"output size is {len(outputs)}")
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
'''

ALPHA = 0.5
BETA = 0.25
GAMMA = 0.25
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import random_split

def count_parameters(model):
    """Function to count the number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_multitask_model(model, optimizer, scheduler, num_epochs=6):
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

def test_model(model, subset, model_type = None):
    model.eval()
    running_corrects = 0
    # Iterate over data.
    temp_max = 500
    temp_count = 0
    for inputs, labels in dataloaders[subset]:
        # print(f"input and label sizes:{len(inputs), len(labels)}")
        temp_count += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        torch.set_grad_enabled(False)
        if model_type == 'resnet34_multi':
            outputs, _, _ = model(inputs)
        else:
            outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        print(preds, labels.data)
        running_corrects += torch.sum(preds == labels.data)
        if subset == 'train' and temp_count >= temp_max:
            break

    print(temp_count, dataset_sizes[subset], running_corrects )
    epoch_acc = running_corrects.double() / temp_count

    print(f"Accuracy {subset}:{epoch_acc}")
    return epoch_acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Non-STR method for number recognition - used for comparison
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--data', help='data root directory')
    parser.add_argument('--weights', help='path to model weights')
    parser.add_argument('--original_weights', help='path to model weights')
    parser.add_argument('model_type', help='resnet34 or resnet34_multi')

    args = parser.parse_args()
    '''
    #train_img_dir = os.path.join(args.data, 'train', 'imgs')
    test_img_dir = os.path.join(args.data, 'test', 'imgs')
    val_img_dir = os.path.join(args.data, 'val', 'imgs')

    train_img_dir = os.path.join(args.data, 'out', 'SoccerNetResults', 'crops', 'imgs')
    annotations_file = os.path.join(args.data, 'out', 'SoccerNetResults', 'final_results.json')

    annotations_file_train = os.path.join(train_img_dir, 'train_gt.txt')
    annotations_file_val = os.path.join(val_img_dir, 'val_gt.txt')
    annotations_file_test = os.path.join(test_img_dir, 'test_gt.txt')

    image_dataset_train = JerseyNumberDataset(annotations_file_train, train_img_dir, 'train')
    image_dataset_test = JerseyNumberDataset(annotations_file_test, test_img_dir, 'test')
    image_dataset_val = JerseyNumberDataset(annotations_file_val, val_img_dir, 'val')

    dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=4,
                                                   shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(image_dataset_test, batch_size=4,
                                                 shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=4,
                                                  shuffle=False, num_workers=4)
    
    train_img_dir = os.path.join('data', 'SoccerNet', 'train', 'images')
    annotations_file = os.path.join('data', 'SoccerNet', 'train', 'train_gt.json')

    image_dataset_train = JerseyNumberMultitaskDataset(annotations_file, train_img_dir, 'train')
    image_dataset_val = JerseyNumberMultitaskDataset(annotations_file, train_img_dir, 'val')
    image_dataset_test = JerseyNumberMultitaskDataset(annotations_file, train_img_dir, 'test')

    dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=4, shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(image_dataset_val, batch_size=4, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=4, shuffle=False, num_workers=4)

    image_datasets = {'train': image_dataset_train, 'val': image_dataset_val, 'test': image_dataset_test}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}
    image_datasets = {'train': image_dataset_train, 'val': image_dataset_val, 'test': image_dataset_test}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}
    '''

    if args.simple:
        model_ft = SimpleJerseyNumberClassifier()
    elif args.model_type == 'resnet34':
        model_ft = JerseyNumberClassifier()
    else:
        model_ft = JerseyNumberMulticlassClassifier()
    if args.fine_tune:
        state_dict = torch.load(args.original_weights, map_location=device)
    # create the model based on ResNet18 and train from pretrained version
    if args.train:
        model_ft = model_ft.to(device)
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        if args.model_type == 'resnet34_multi':
            model_ft = train_multitask_model(model_ft, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=6)
        else:
            criterion = nn.CrossEntropyLoss()
            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=15)

        # Decay LR by a factor of 0.1 every 7 epochs
        torch.save(model_ft.state_dict(), args.weights)

    else: # test on validation set
        #load weights
        state_dict = torch.load(args.weights, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        model_ft.load_state_dict(state_dict)
        model_ft = model_ft.to(device)
        test_model(model_ft, 'test', model_type = args.model_type)
