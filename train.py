import os, random, csv
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

CSV_PATH = "labels.csv"
OUT_DIR = "runs_cls"
IMG_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
SEED = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(SEED)
torch.manual_seed(SEED)


class CSVDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, str]], classes: List[str], train: bool):
        self.rows = rows
        self.cls2idx = {c: i for i, c in enumerate(classes)}
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path, label = self.rows[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        y = self.cls2idx[label]
        return x, y


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ip = r["image_path"]
            lb = r["label"]
            rows.append((ip, lb))
    return rows


def split_rows(rows):
    # unknown 제외
    rows = [r for r in rows if r[1] != "unknown"]
    random.shuffle(rows)
    n = len(rows)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    test = rows[:n_test]
    val = rows[n_test : n_test + n_val]
    train = rows[n_test + n_val :]
    return train, val, test


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = read_csv(CSV_PATH)

    # 클래스 목록(사전 순 고정)
    classes = sorted(list({lb for _, lb in rows if lb != "unknown"}))
    print("Classes:", classes)

    train_rows, val_rows, test_rows = split_rows(rows)
    print(f"train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    train_ds = CSVDataset(train_rows, classes, train=True)
    val_ds = CSVDataset(val_rows, classes, train=False)
    test_ds = CSVDataset(test_rows, classes, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 모델: EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, len(classes))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_ds)

        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0

        print(f"[{epoch}/{EPOCHS}] loss={tr_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                },
                os.path.join(OUT_DIR, "best.pt"),
            )
            print("  ↑ saved best")

    # test
    ckpt = torch.load(os.path.join(OUT_DIR, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"[TEST] acc={correct/total:.4f}")


if __name__ == "__main__":
    main()
