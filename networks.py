import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FPNResNet34(nn.Module):
    def __init__(self, pretrained=True, num_classes=87):  # 86 classes + 1 for non-visible (-1)
        super(FPNResNet34, self).__init__()

        # Load the pre-trained ResNet34 model
        resnet = models.resnet34(pretrained=pretrained)

        # Extract the layers of ResNet34 backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Exclude fully connected layers

        # FPN levels: Extracts from different layers of the backbone
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Lateral connections for FPN
        self.lateral1 = nn.Conv2d(256, 256, kernel_size=1)  # From layer1
        self.lateral2 = nn.Conv2d(512, 256, kernel_size=1)  # From layer2
        self.lateral3 = nn.Conv2d(1024, 256, kernel_size=1)  # From layer3
        self.lateral4 = nn.Conv2d(2048, 256, kernel_size=1)  # From layer4

        # Top-down pathway and fusion
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample for top-down fusion

        # Final classifier to classify jersey numbers (or non-visible)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),  # Output number of classes (86 or 87 including -1)
            nn.AdaptiveAvgPool2d(1),  # Pooling to get a single value per feature map
            nn.Flatten(),
            nn.Softmax(dim=1)  # Output probabilities for classification
        )

    def forward(self, x):
        # Extract features from each layer of the backbone
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Apply lateral connections for FPN to each feature map
        p1 = self.lateral1(x1)  # Lateral connection for layer1
        p2 = self.lateral2(x2)  # Lateral connection for layer2
        p3 = self.lateral3(x3)  # Lateral connection for layer3
        p4 = self.lateral4(x4)  # Lateral connection for layer4

        # Top-down pathway to merge features
        p3 = self.upsample(p3)
        p2 = self.upsample(p2)
        p1 = self.upsample(p1)

        # Fusion of features from different layers (this is a basic fusion, can be expanded)
        p2 = p2 + p3
        p1 = p1 + p2
        p1 = p1 + p4  # Final merge from all levels

        # Classify the final merged features
        out = self.classifier(p1)

        return out
    
# Define the Squeeze-and-Excitation block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling (average over height and width)
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)  # Average over the height and width
        avg_pool = avg_pool.view(avg_pool.size(0), -1)  # Flatten to shape [batch_size, channel]

        # Fully connected layers
        x_se = self.fc1(avg_pool)  # fc1 expects input size [batch_size, channel]
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)
        x_se = self.sigmoid(x_se).view(x_se.size(0), x_se.size(1), 1, 1)  # Reshape to [batch_size, channel, 1, 1]

        return x * x_se.expand_as(x)


# Define ResNet with Squeeze-and-Excitation Block
class ResNetSE(nn.Module):
    def __init__(self):
        super(ResNetSE, self).__init__()
        self.model_ft = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        num_ftrs = self.model_ft.fc.in_features
        
        # Insert SE layer in the ResNet architecture for layer4 output channels
        self.se_layer = SELayer(512)  # Number of channels in layer4 of ResNet34

        # Modify final fully connected layer for 100 classes (jersey numbers)
        self.model_ft.fc = nn.Linear(num_ftrs, 100)  # Output 100 classes for jersey numbers

    def forward(self, x):
        # Forward pass through the ResNet model
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        # Apply layers in layer1, layer2, layer3
        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        
        # Apply SE layer after layer4
        x = self.model_ft.layer4(x)
        x = self.se_layer(x)  # Apply SE block here
        
        # Average pooling and fully connected layer
        x = self.model_ft.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model_ft.fc(x)
        return x


# Baseline ResNet34 model
class JerseyNumberClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 100)

    def forward(self, input):
        return self.model_ft(input)
    
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JerseyNumberMulticlassClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])

        self.head1 = nn.Linear(512, 100)
        self.head2 = nn.Linear(512, 10)
        self.head3 = nn.Linear(512, 11)

    def forward(self, input):
        # get backbone features
        backbone_feats = self.backbone(input)

        backbone_feats =backbone_feats.reshape(backbone_feats.size(0), -1)

        # pass through heads
        h1 = self.head1(backbone_feats)
        h2 = self.head2(backbone_feats)
        h3 = self.head3(backbone_feats)
        return h1, h2, h3


class SimpleJerseyNumberClassifier(nn.Module):
    def __init__(self):
        super(SimpleJerseyNumberClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(65536, 2048)
        self.linear2 = nn.Linear(2048, 100)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x



# ResNet18 based model for binary classification
class LegibilityClassifier(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.resnet18(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True
        self.model_ft.avgpool.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = F.sigmoid(x)
        return x

# ResNet34 based model for binary classification
class LegibilityClassifier34(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = F.sigmoid(x)
        return x

# ResNet18 based model for binary classification
class LegibilityClassifierTransformer(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.vit_b_16(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.heads.head.in_features
        self.model_ft.heads.head = nn.Linear(num_ftrs, 1)
        self.model_ft.heads.head.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = F.sigmoid(x)
        return x

# Classifier Model
class LegibilitySimpleClassifier(nn.Module):
    def __init__(self):
        super(LegibilitySimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(8192, 2048)
        self.linear2 = nn.Linear(2048, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.leaky_relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        return x
