"""
Single-Pass Multimodal Decision Network (MultimodalDecisionNet)
Combines ECG Waveform Images and Numerical Signal Quality Indicators (SQIs) in a single forward pass.

Ablation Modes Supported:
1. 'image_only': Vision stream active, tabular zeroed.
2. 'sqi_only': Tabular stream active, vision zeroed.
3. 'multimodal': Both vision and numerical SQI streams active with gated fusion.
"""

from typing import Optional, List, Literal
import numpy as np
from PIL import Image
import torch

from config import RANDOM_SEED
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

from features.sqi_extractor import SQI_FEATURE_NAMES

_MM_IMAGE_CACHE = {}

class MultimodalECGDataset(Dataset):
    def __init__(self, image_paths: List[str], sqi_vectors: np.ndarray, labels: Optional[np.ndarray] = None, transform=None):
        self.image_paths = image_paths
        self.sqi_vectors = torch.from_numpy(np.nan_to_num(sqi_vectors, nan=0.0).astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32)) if labels is not None else None
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if path not in _MM_IMAGE_CACHE:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            _MM_IMAGE_CACHE[path] = img
        img = _MM_IMAGE_CACHE[path]
            
        sqi = self.sqi_vectors[idx]
        if self.labels is not None:
            return img, sqi, self.labels[idx]
        return img, sqi


class MultimodalDecisionNet(nn.Module):
    def __init__(self, num_sqi_features: int = len(SQI_FEATURE_NAMES), embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Vision Stream (2D CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. Tabular SQI Stream (MLP)
        self.sqi_encoder = nn.Sequential(
            nn.Linear(num_sqi_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3. Gated Multimodal Fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.Sigmoid()
        )
        
        # 4. Decision Head (outputs P(unusable))
        self.decision_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(
        self,
        img: torch.Tensor,
        sqi: torch.Tensor,
        mode: Literal["multimodal", "image_only", "sqi_only"] = "multimodal"
    ) -> torch.Tensor:
        b_size = img.size(0)
        
        if mode == "sqi_only":
            v_feat = torch.zeros(b_size, self.embed_dim, device=img.device)
            t_feat = self.sqi_encoder(sqi)
        elif mode == "image_only":
            v_feat = self.vision_encoder(img)
            t_feat = torch.zeros(b_size, self.embed_dim, device=img.device)
        else: # multimodal
            v_feat = self.vision_encoder(img)
            t_feat = self.sqi_encoder(sqi)
            
        combined = torch.cat([v_feat, t_feat], dim=-1)
        gate_weights = self.fusion_gate(combined)
        gated_combined = combined * gate_weights
        logits = self.decision_head(gated_combined)
        return logits.squeeze(-1)


class SinglePassMultimodalClassifier:
    def __init__(
        self,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 1e-3,
        mode: Literal["multimodal", "image_only", "sqi_only"] = "multimodal",
        device: Optional[str] = None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mode = mode
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
        self.net = MultimodalDecisionNet().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.95, 0.95, 0.95], std=[0.15, 0.15, 0.15])
        ])
        self.sqi_mean = None
        self.sqi_scale = None
        
    def fit(self, image_paths: List[str], sqi_vectors: np.ndarray, y: np.ndarray):
        sqi_vectors = np.nan_to_num(sqi_vectors, nan=0.0)
        self.sqi_mean = np.mean(sqi_vectors, axis=0)
        self.sqi_scale = np.std(sqi_vectors, axis=0) + 1e-8
        sqi_scaled = (sqi_vectors - self.sqi_mean) / self.sqi_scale
        
        dataset = MultimodalECGDataset(image_paths, sqi_scaled, y, transform=self.transform)
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
            for batch_img, batch_sqi, batch_y in loader:
                batch_img = batch_img.to(self.device)
                batch_sqi = batch_sqi.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.net(batch_img, batch_sqi, mode=self.mode)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        return self

    def predict_proba(
        self,
        image_paths: List[str],
        sqi_vectors: np.ndarray,
        override_mode: Optional[str] = None
    ) -> np.ndarray:
        self.net.eval()
        mode_to_use = override_mode if override_mode is not None else self.mode
        
        sqi_vectors = np.nan_to_num(sqi_vectors, nan=0.0)
        if self.sqi_mean is not None and self.sqi_scale is not None:
            sqi_scaled = (sqi_vectors - self.sqi_mean) / self.sqi_scale
        else:
            sqi_scaled = sqi_vectors
            
        dataset = MultimodalECGDataset(image_paths, sqi_scaled, transform=self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for batch_img, batch_sqi in loader:
                batch_img = batch_img.to(self.device)
                batch_sqi = batch_sqi.to(self.device)
                logits = self.net(batch_img, batch_sqi, mode=mode_to_use)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p.tolist())
                
        return np.array(probs, dtype=np.float32)

    def predict(self, image_paths: List[str], sqi_vectors: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(image_paths, sqi_vectors) >= threshold).astype(int)


class IndependentSQINet(nn.Module):
    def __init__(self, in_features: int = len(SQI_FEATURE_NAMES), hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class IndependentSQIClassifier:
    def __init__(self, in_features: int = len(SQI_FEATURE_NAMES), epochs: int = 35, lr: float = 1e-3,
                 batch_size: int = 64, device: Optional[str] = None):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
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
        self.net = IndependentSQINet(in_features=in_features).to(self.device)
        self.scaler_mean = None
        self.scaler_scale = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.nan_to_num(X, nan=0.0)
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_scale = np.std(X, axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        
        t_x = torch.tensor(X_scaled, dtype=torch.float32)
        t_y = torch.tensor(y, dtype=torch.float32)

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        pos_weight = torch.tensor([float(n_neg / max(1, n_pos))], device=self.device)

        # Mini-batch training: one epoch = one pass over the training set (not a single full-batch step)
        dataset = TensorDataset(t_x, t_y)
        drop_last = (len(dataset) % self.batch_size == 1) # BatchNorm1d cannot handle a trailing batch of size 1
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)

        self.net.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        t_x = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net(t_x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
