import os
import torch

from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights, MobileNet_V2_Weights

import misc
import config

class ClassificationResNet18(nn.Module):
    def __init__(self, load_pretrained=False):
        super(ClassificationResNet18, self).__init__()
        # Load the pretrained resnet18 model

        if load_pretrained and os.path.exists(config.WEIGHTS_PATH):
            weights = torch.load(config.WEIGHTS_PATH, map_location=misc.get_device())
        else:
            weights=ResNet18_Weights.DEFAULT

        model = models.resnet18(weights=weights)

        # Retain all layers of the original model, only change the final layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        num_features = model.fc.in_features

        # Add a new fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(num_features, len(config.CLASSES))
        )

        # Initialize weights of the new classifier layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs, features

class AttentionModule(nn.Module):
    def __init__(self, in_features):
        super(AttentionModule, self).__init__()
        self.attention = nn.Linear(in_features, in_features)
        
    def forward(self, x):
        b, _ = x.size()
        attn_weights = F.softmax(self.attention(x), dim=1)
        out = attn_weights * x
        return out

class ClassificationResNet50WithAttention(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(ClassificationResNet50WithAttention, self).__init__()
        # Load the pretrained resnet50 model
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Retain all layers of the original model, only change the final layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.dropout = nn.Dropout(p=dropout_rate)
        num_features = model.fc.in_features

        # Attention Module
        self.attention = AttentionModule(num_features)
        
        # Additional Linear layer to reduce feature size to FEATURE_VECTOR_SIZE
        fv_size = config.FEATURE_VECTOR_SIZE
        self.additional_fc = nn.Linear(num_features, fv_size)
        
        # Add a new fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fv_size, len(config.CLASSES))  # Assuming config.CLASSES is predefined
        )

        # Initialize weights of the new classifier layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        features = self.dropout(features)
        b, c, h, w = features.size()
        features = features.view(b, -1)
        features = self.attention(features)
        features = F.relu(self.additional_fc(features))
        outputs = self.classifier(features)
        return outputs, features

class ClassificationResNet(nn.Module):
    def __init__(self, model = 'ResNet-18', dropout_rate=0.0):
        super(ClassificationResNet, self).__init__()

        # Load the pretrained resnet model
        if model == 'ResNet-18':
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model == 'ResNet-50':
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError('Invalid model type')

        # Retain all layers of the original model, only change the final layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.dropout = nn.Dropout(p=dropout_rate)
        num_features = model.fc.in_features

        # Additional Linear layer to reduce feature size to FEATURE_VECTOR_SIZE
        fv_size = config.FEATURE_VECTOR_SIZE
        # self.additional_fc = nn.Sequential(
        #     nn.Linear(num_features, fv_size),
        #     nn.BatchNorm1d(fv_size)
        # )

        self.additional_fc = nn.Linear(num_features, fv_size)
 
        # Add a new fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fv_size, len(config.CLASSES))
        )

        # Initialize weights of the new classifier layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        features = self.dropout(features)
        b, c, h, w = features.size()  # get the size of the feature tensor
        features = features.view(b, -1)  # flatten the feature tensor
        features = F.relu(self.additional_fc(features))
        outputs = self.classifier(features)
        return outputs, features

class ClassificationDenseNet121(nn.Module):
    def __init__(self):
        super(ClassificationDenseNet121, self).__init__()
        # Load the pretrained densenet121 model
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Retain all layers of the original model, only change the final layer
        self.features = model.features
        num_features = model.classifier.in_features

        # Add a new fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, len(config.CLASSES))  # Assuming binary classification
        )

        # Initialize weights of the new classifier layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs, features

class ClassificationEfficientNet(nn.Module):
    def __init__(self):
        super(ClassificationEfficientNet, self).__init__()
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])  # Removing the classifier layer
        num_features = model._fc.in_features  # Accessing the in_features of the last FC layer

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, len(config.CLASSES))
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs, features

class ClassificationMobileNet(nn.Module):
    def __init__(self):
        super(ClassificationMobileNet, self).__init__()
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights)
        self.features = model.features
        num_features = model.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, len(config.CLASSES))
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs, features

