import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

CLASS_IDS = [0, 9, 10, 11, 12, 15]
CLASS_MAP = {id_: idx for idx, id_ in enumerate(CLASS_IDS)}

def collect_labels(json_dirs):
    """Collects unique labels across all auxiliary tasks to fit LabelEncoders."""
    crops, areas, risks, grows = set(), set(), set(), set()
    for json_dir in json_dirs:
        for file in os.listdir(json_dir):
            if not file.endswith(".json"):
                continue
            path = os.path.join(json_dir, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                ann = data.get("annotations", {})
                crop = ann.get("crop")
                area = ann.get("area")
                risk = ann.get("risk")
                grow = next((obj.get("grow") for obj in ann.get("object", []) if obj.get("grow") is not None), None)

                if None not in [crop, area, risk, grow]:
                    crops.add(crop)
                    areas.add(area)
                    risks.add(risk)
                    grows.add(grow)
            except Exception:
                pass
    return crops, areas, risks, grows

class MultitaskDataset(Dataset):
    def __init__(self, json_dir, image_dir, transform, encoders):
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.transform = transform
        self.encoders = encoders
        self.samples = []

        for file in os.listdir(json_dir):
            if file.endswith(".json"):
                path = os.path.join(json_dir, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    ann = data.get("annotations", {})
                    disease = ann.get("disease")
                    objects = ann.get("object", [])

                    label = CLASS_MAP.get(disease)
                    if label is None:
                        label = next((CLASS_MAP[obj.get("class")] for obj in objects if obj.get("class") in CLASS_MAP), None)

                    crop = ann.get("crop")
                    area = ann.get("area")
                    risk = ann.get("risk")
                    grow = next((obj.get("grow") for obj in objects if obj.get("grow") is not None), None)

                    if label is not None and None not in [crop, area, risk, grow]:
                        self.samples.append({
                            "img_path": os.path.join(image_dir, data["description"]["image"]),
                            "class": label,
                            "crop": crop, "area": area, "risk": risk, "grow": grow
                        })
                except Exception:
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        targets = {
            "class": torch.tensor(sample["class"]),
            "crop": torch.tensor(self.encoders['crop'].transform([sample["crop"]])[0]),
            "area": torch.tensor(self.encoders['area'].transform([sample["area"]])[0]),
            "risk": torch.tensor(self.encoders['risk'].transform([sample["risk"]])[0]),
            "grow": torch.tensor(self.encoders['grow'].transform([sample["grow"]])[0])
        }
        return img, targets

class MultiHeadClassifier(nn.Module):
    """
    EfficientNet-B4 Backbone with parallel heads for joint prediction 
    of disease, crop type, affected area, risk level, and growth stage.
    """
    def __init__(self, num_crop, num_area, num_risk, num_grow):
        super().__init__()
        self.backbone = efficientnet_b4(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()  # Remove FC to extract features
        
        in_features = 1792
        self.fc_class = nn.Linear(in_features, len(CLASS_IDS)) 
        self.fc_crop = nn.Linear(in_features, num_crop)
        self.fc_area = nn.Linear(in_features, num_area)
        self.fc_risk = nn.Linear(in_features, num_risk)
        self.fc_grow = nn.Linear(in_features, num_grow)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "class": self.fc_class(feat),
            "crop": self.fc_crop(feat),
            "area": self.fc_area(feat),
            "risk": self.fc_risk(feat),
            "grow": self.fc_grow(feat)
        }

def compute_loss(pred, target, criterion_dict):
    """Aggregates loss across all parallel classification heads."""
    return sum(criterion_dict[task](pred[task], target[task]) for task in criterion_dict)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Encoders Setup
    train_dir = r"path/to/Training/labels"
    val_dir = r"path/to/Validation/labels"
    crops, areas, risks, grows = collect_labels([train_dir, val_dir])
    
    encoders = {
        'crop': LabelEncoder().fit(list(crops)),
        'area': LabelEncoder().fit(list(areas)),
        'risk': LabelEncoder().fit(list(risks)),
        'grow': LabelEncoder().fit(list(grows))
    }

    # 2. Datasets & Loaders
    transform = transforms.Compose([transforms.Resize((380, 380)), transforms.ToTensor()])
    
    train_dataset = MultitaskDataset(train_dir, r"path/to/Training/images", transform, encoders)
    val_dataset = MultitaskDataset(val_dir, r"path/to/Validation/images", transform, encoders)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 3. Model & Loss setup
    model = MultiHeadClassifier(
        num_crop=len(encoders['crop'].classes_),
        num_area=len(encoders['area'].classes_),
        num_risk=len(encoders['risk'].classes_),
        num_grow=len(encoders['grow'].classes_)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion_dict = {task: nn.CrossEntropyLoss() for task in ["class", "crop", "area", "risk", "grow"]}

    # 4. Training Execution (Skeleton)
    EPOCHS = 10
    checkpoint_path = "checkpoints/multitask_classifier_latest.pth"
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            imgs = imgs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(imgs)
            loss = compute_loss(outputs, targets, criterion_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch} - Train Loss: {total_loss / len(train_loader):.4f}")
        # Note: Validation omitted here due to identified leakage issue.
        # Ensure validation is run on an uncorrupted, strictly isolated split.