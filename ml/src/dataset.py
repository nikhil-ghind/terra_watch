import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

LABEL_TO_INT = {
    "no_change": 0,
    "deforestation": 1,
    "urban_expansion": 2,
    "water_shift": 3,
    "agriculture": 4,
}


def get_transforms(train: bool) -> T.Compose:
    base = [T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if train:
        base = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(0.2, 0.2)] + base
    return T.Compose(base)


class ChangeDetectionDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # For synthetic data, return random tensors
        tensor = torch.randn(9, 224, 224)  # 3 bands × 3 views (before, after, diff)
        label = LABEL_TO_INT.get(str(row.get("label", "no_change")), 0)
        return tensor, label
