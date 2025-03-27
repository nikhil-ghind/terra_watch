import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score

from ml.src.dataset import ChangeDetectionDataset, get_transforms
from ml.src.model import build_resnet50_extractor


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, criterion, device) -> tuple:
    model.eval()
    total, preds_all, labels_all = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += criterion(logits, y).item()
            preds_all.extend(logits.argmax(1).cpu().tolist())
            labels_all.extend(y.cpu().tolist())
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total / len(loader), f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="ml/data/sample.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="ml/checkpoints")
    args = parser.parse_args()

    ds = ChangeDetectionDataset(args.data_csv, transform=get_transforms(train=True))
    train_size = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet50_extractor().to(device)
    # Adapt first conv to 9 input channels (3 bands × before/after/diff)
    model.conv1 = nn.Conv2d(9, 64, 7, 2, 3, bias=False).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch} | loss={loss:.4f} | val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
    print(f"Best macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
