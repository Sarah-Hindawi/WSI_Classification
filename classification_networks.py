import os
import torch
import misc
import config
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

class ClassificationResNet18(nn.Module):
    def __init__(self, load_local_weights=False):
        super(ClassificationResNet18, self).__init__()
        
        # Load the pretrained resnet18 model
        if load_local_weights and os.path.exists(config.WEIGHTS_PATH):
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



