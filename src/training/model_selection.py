"""
NagrikAI - Model Architecture Selection & Factory
Supports MobileNetV3-Small, EfficientNet-B0, and ResNet-18 architectures.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models

MODEL_REGISTRY = {
    "mobilenet_v3": "mobilenet_v3_small",
    "mobilenet_v3_small": "mobilenet_v3_small",
    "efficientnet_b0": "efficientnet_b0",
    "resnet18": "resnet18"
}

def build_classifier_model(
    model_name: str = "mobilenet_v3_small",
    num_classes: int = 10,
    pretrained: bool = True
) -> nn.Module:
    """
    Constructs transfer learning model adapted to NagrikAI waste classes.
    Replaces final classification head with appropriate output projection.
    """
    norm_name = model_name.lower().replace("-", "_")
    target_arch = MODEL_REGISTRY.get(norm_name, "mobilenet_v3_small")
    
    if target_arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
    elif target_arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
    elif target_arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown architecture: {model_name}")
        
    return model

