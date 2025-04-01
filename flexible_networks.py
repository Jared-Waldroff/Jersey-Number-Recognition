import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlexibleLegibilityConvNet(nn.Module):
    """
    A more flexible ConvNet implementation that can adapt to different model structures
    when loading pre-trained weights.
    """
    def __init__(self, input_channels=3, initial_filters=32, num_blocks=3, fc_features=256, dropout_rate=0.5):
        super().__init__()
        
        # Define the convolutional layers explicitly instead of using ModuleList
        # This makes it easier to match with pre-trained weights
        self.conv1 = nn.Conv2d(input_channels, initial_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(initial_filters, initial_filters*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(initial_filters*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(initial_filters*2, initial_filters*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(initial_filters*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Add a fourth convolutional block for more capacity
        self.conv4 = nn.Conv2d(initial_filters*4, initial_filters*8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(initial_filters*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Add attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(initial_filters*8, initial_filters*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters*2, initial_filters*8, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Calculate the flattened size based on input dimensions
        # For 224x224 input, after 4 pooling layers (each dividing dimensions by 2)
        # the spatial dimensions will be 14x14
        self.fc_input_features = (initial_filters*8) * 14 * 14
        
        # Improved fully connected layers
        self.fc1 = nn.Linear(self.fc_input_features, fc_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_features, fc_features // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(fc_features // 2, 1)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Apply channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        x = self.pool4(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return torch.sigmoid(x)

class AdaptiveConvNet(nn.Module):
    """
    An adaptive ConvNet that can handle various input sizes and tries to
    adapt to the structure of pre-trained weights.
    """
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Basic convolutional blocks with standard channel progression
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4 (optional, will be skipped if input is too small)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # Adaptive pooling to fixed size
        )
        
        # Use adaptive pooling to handle various input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier with flexible input size
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ImprovedAdaptiveConvNet(nn.Module):
    """
    An improved adaptive ConvNet with residual connections and attention mechanisms
    """
    def __init__(self, input_channels=3, initial_filters=32):
        super().__init__()
        
        # First block
        self.conv1 = nn.Conv2d(input_channels, initial_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(initial_filters, initial_filters*2)
        self.res_block2 = self._make_residual_block(initial_filters*2, initial_filters*4)
        self.res_block3 = self._make_residual_block(initial_filters*4, initial_filters*8)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(initial_filters*8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            
            # Shortcut connection
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            ),
            
            # Final activation
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self._forward_residual_block(x, self.res_block1)
        x = self._forward_residual_block(x, self.res_block2)
        x = self._forward_residual_block(x, self.res_block3)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def _forward_residual_block(self, x, block):
        # Extract the layers from the block
        conv1 = block[0]
        bn1 = block[1]
        relu1 = block[2]
        conv2 = block[3]
        bn2 = block[4]
        shortcut = block[5]
        final_relu = block[6]
        
        # Forward pass with residual connection
        identity = shortcut(x)
        
        out = conv1(x)
        out = bn1(out)
        out = relu1(out)
        
        out = conv2(out)
        out = bn2(out)
        
        out += identity
        out = final_relu(out)
        
        return out

def load_flexible_weights(model, state_dict):
    """Load weights into a flexible model, handling mismatches gracefully"""
    # Create a new state dict that will hold the compatible weights
    new_state_dict = {}
    model_state_dict = model.state_dict()
    
    # Print info about the model and state dict
    print(f"Model has {len(model_state_dict)} layers")
    print(f"State dict has {len(state_dict)} layers")
    
    # Try to load as many weights as possible
    mismatched_layers = []
    for name, param in model_state_dict.items():
        if name in state_dict and param.size() == state_dict[name].size():
            new_state_dict[name] = state_dict[name]
        else:
            mismatched_layers.append(name)
            # Keep the original initialization for this layer
            new_state_dict[name] = param
    
    # Print info about the loading process
    print(f"Successfully loaded {len(new_state_dict) - len(mismatched_layers)}/{len(model_state_dict)} layers")
    if mismatched_layers:
        print(f"Mismatched layers: {', '.join(mismatched_layers[:5])}" + 
              (f" and {len(mismatched_layers) - 5} more" if len(mismatched_layers) > 5 else ""))
    
    # Load the compatible weights
    model.load_state_dict(new_state_dict)
    return model
