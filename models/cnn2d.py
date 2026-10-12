"""
2D CNN Baseline for ECG Waveform Images.
Input: 2D waveform RGB image [batch_size, 3, H, W]
Output: Calibrated probability P(unusable)
"""

from typing import Optional, List
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from config import RANDOM_SEED
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

_IMAGE_TENSOR_CACHE = {}

class ECGImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: Optional[np.ndarray] = None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if path not in _IMAGE_TENSOR_CACHE:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            _IMAGE_TENSOR_CACHE[path] = img
        img = _IMAGE_TENSOR_CACHE[path]
            
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        return img


class ECG2DCNN(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits.squeeze(-1)


class ECG2DCNNClassifier:
    def __init__(self, epochs: int = 20, batch_size: int = 32, lr: float = 1e-3, device: Optional[str] = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        torch.manual_seed(RANDOM_SEED)
        self.net = ECG2DCNN().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.95, 0.95, 0.95], std=[0.15, 0.15, 0.15])
        ])
        
    def fit(self, image_paths: List[str], y: np.ndarray):
        dataset = ECGImageDataset(image_paths, y, transform=self.transform)
        drop_last = (len(dataset) % self.batch_size == 1)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)
        
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        pos_weight = torch.tensor([float(n_neg / max(1, n_pos))], device=self.device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        self.net.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        return self

    def predict_proba(self, image_paths: List[str]) -> np.ndarray:
        self.net.eval()
        dataset = ECGImageDataset(image_paths, transform=self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                logits = self.net(batch_x)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p.tolist())
                
        return np.array(probs, dtype=np.float32)

    def predict(self, image_paths: List[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(image_paths) >= threshold).astype(int)
