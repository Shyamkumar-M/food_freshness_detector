# scripts/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.baseline import get_model   # uses your baseline.py
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Config
DATA_DIR = "data"
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-3
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join("saved_models", "resnet18_freshness.pth")

# Transforms and loaders (assumes train/val/test folders already exist)
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform_train)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, optimizer
model = get_model(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE); labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        if i % 50 == 0:
            print(f"Epoch {epoch+1} Batch {i}/{len(train_loader)} Loss {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)

    # validation
    model.eval()
    val_preds = []; val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE); labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy()); val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

# save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved model to {SAVE_PATH}")
