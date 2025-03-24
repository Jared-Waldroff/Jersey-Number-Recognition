import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from torchvision.models import ResNet34_Weights

class JerseyNumberClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 100)

    def forward(self, input):
        return self.model_ft(input)

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


# Custom CNN
class CustomCNN(nn.Module):
    def __init__(self, feature_dim=2048):
        super(CustomCNN, self).__init__()
        # Define a simple but effective CNN architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Additional convolutional blocks
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        # Global pooling and feature projection
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, feature_dim)
        self.bn_final = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # Global feature
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        v = self.fc(v)
        v = self.bn_final(v)

        # To maintain the same interface as CTLModel from centroids-reid
        return x, v
    
    # ResNet with Feature Pyramid Network (FPN) for Jersey Number Recognition
class FPNResNet34(nn.Module):
    def __init__(self, pretrained=True, num_classes=100):  # 100 jersey numbers (0-99)
        super(FPNResNet34, self).__init__()

        # Load the pre-trained ResNet34 model
        resnet = models.resnet34(pretrained=pretrained)

        # Extract ResNet34 convolutional backbone layers
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 64 -> 64 channels
        self.layer2 = resnet.layer2  # 64 -> 128 channels
        self.layer3 = resnet.layer3  # 128 -> 256 channels
        self.layer4 = resnet.layer4  # 256 -> 512 channels (Final layer)

        # FPN lateral connections
        self.lateral1 = nn.Conv2d(64, 256, kernel_size=1)    # Layer1 (64 -> 256)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)   # Layer2 (128 -> 256)
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1)   # Layer3 (256 -> 256)
        self.lateral4 = nn.Conv2d(512, 256, kernel_size=1)   # Layer4 (512 -> 256)

        # Top-down pathway (upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Classifier for jersey number recognition
        self.classifier = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),  # 101 output classes
            nn.AdaptiveAvgPool2d(1),  # Pooling to 1x1
            nn.Flatten()
        )

    def forward(self, x):
        # Extract feature maps from ResNet backbone
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Apply lateral connections
        p4 = self.lateral4(x4)  # Highest-level feature map
        p3 = self.lateral3(x3) + self.upsample(p4)
        p2 = self.lateral2(x2) + self.upsample(p3)
        p1 = self.lateral1(x1) + self.upsample(p2)

        # Classify final feature map
        out = self.classifier(p1)  # Shape: (batch_size, num_classes)

        return out  # Softmax should be applied during inference
    
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