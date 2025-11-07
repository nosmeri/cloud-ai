import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel().to(device)
opt = optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()  # 혼합정밀 학습용

# 성능 옵션
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.0+

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)

for x, y in train_loader:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    opt.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.float16):   # 혼합정밀로 속도↑, 메모리↓
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)

    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()