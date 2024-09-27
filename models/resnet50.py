import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def get_model(num_classes):
    # pre-trained ResNet50 model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # replacing the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

