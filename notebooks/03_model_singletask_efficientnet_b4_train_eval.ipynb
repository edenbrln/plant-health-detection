import os
import json
import collections
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Global Configuration
CLASS_IDS = [0, 9, 10, 11, 12, 15]
CLASS_MAP = {id_: idx for idx, id_ in enumerate(CLASS_IDS)}
CLASS_NAMES = {
    0: "Normal",
    9: "Cabbage Sclerotinia Rot",
    10: "Cabbage Soft Rot",
    11: "Cabbage White Butterfly",
    12: "Flea Beetle",
    15: "Brown Marmorated Stink Bug"
}
CLASS_NAMES_LST = [CLASS_NAMES[id_] for id_ in CLASS_IDS]

# -----------------
# Dataset Definition
# -----------------
class SimpleClassificationDataset(Dataset):
    def __init__(self, json_dir, image_dir, transform=None):
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        for filename in os.listdir(json_dir):
            if not filename.endswith(".json"):
                continue
                
            file_path = os.path.join(json_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                annotations = data.get("annotations", {})
                label = annotations.get("class")
                if label is None and "object" in annotations and annotations["object"]:
                    label = annotations["object"][0].get("class")
                
                if label is not None and label in CLASS_MAP:
                    img_name = data["description"]["image"]
                    self.samples.append({
                        "img_path": os.path.join(image_dir, img_name),
                        "class": CLASS_MAP[label]
                    })
            except Exception as e:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(sample["class"])

# -----------------
# Model Architecture
# -----------------
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b4(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# -----------------
# Training & Validation Routines
# -----------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, targets in tqdm(dataloader, desc="Training"):
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Validation"):
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct_preds += (preds == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = correct_preds / total_samples
    return total_loss / len(dataloader), accuracy

# -----------------
# Execution Block
# -----------------
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths (Update with actual paths before deployment)
    training_json_dir = r"path/to/Training/labels"
    training_image_dir = r"path/to/Training/images"
    validation_json_dir = r"path/to/Validation/labels"
    validation_image_dir = r"path/to/Validation/images"
    checkpoint_path = "checkpoints/simple_classifier_checkpoint.pth"

    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
    ])

    # Datasets and Loaders
    train_dataset = SimpleClassificationDataset(training_json_dir, training_image_dir, transform)
    val_dataset = SimpleClassificationDataset(validation_json_dir, validation_image_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model, Optimizer, Criterion
    model = SimpleClassifier(len(CLASS_IDS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    EPOCHS = 10 

    # Load Checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\\n[Epoch {epoch}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save Checkpoint
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint_dict, checkpoint_path)

    # -----------------
    # Final Evaluation & Visualization
    # -----------------
    print("\\nRunning Final Evaluation on Validation Set...")
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = targets.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(CLASS_IDS)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES_LST)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format=".0f")
    plt.title("Confusion Matrix (Validation Set)", fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Classification Report
    print("\\nClassification Report:")
    print(classification_report(all_targets, all_preds, labels=range(len(CLASS_IDS)), target_names=CLASS_NAMES_LST))