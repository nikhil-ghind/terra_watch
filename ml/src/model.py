import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

NUM_CLASSES = 5  # no_change, deforestation, urban_expansion, water_shift, agriculture


def build_resnet50_extractor() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Freeze early layers
    for name, param in model.named_parameters():
        if not any(k in name for k in ["layer3", "layer4", "fc"]):
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


def build_embedding_extractor(model: nn.Module) -> nn.Module:
    layers = list(model.children())[:-1]
    return nn.Sequential(*layers, nn.Flatten())
