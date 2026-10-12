"""
1D CNN Baseline for Raw ECG Signal Quality Assessment.
Input: 1D waveform tensor [batch_size, 1, signal_length]
Output: Calibrated probability P(unusable)
"""

from typing import Optional, Tuple
import numpy as np
import torch

from config import RANDOM_SEED
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(self.act(self.bn(self.conv(x)))))


class RawECG1DCNN(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.b1 = Conv1DBlock(in_channels, hidden_dim, kernel_size=15, pool_size=4)
        self.b2 = Conv1DBlock(hidden_dim, hidden_dim * 2, kernel_size=11, pool_size=4)
        self.b3 = Conv1DBlock(hidden_dim * 2, hidden_dim * 4, kernel_size=7, pool_size=4)
        self.b4 = Conv1DBlock(hidden_dim * 4, hidden_dim * 4, kernel_size=5, pool_size=4)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, L]
        feat = self.b4(self.b3(self.b2(self.b1(x))))
        pooled = self.gap(feat).squeeze(-1)
        logits = self.classifier(pooled)
        return logits.squeeze(-1) # [B]


class ECG1DCNNClassifier:
    def __init__(self, epochs: int = 25, batch_size: int = 32, lr: float = 1e-3, device: Optional[str] = None):
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
        self.net = RawECG1DCNN().to(self.device)
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        X = np.stack(X) if isinstance(X, list) else np.array(X)
        # Vectorized z-score normalization
        means = np.mean(X, axis=1, keepdims=True)
        stds = np.std(X, axis=1, keepdims=True)
        stds[stds < 1e-6] = 1.0
        X_norm = (X - means) / stds
        
        X_tensor = torch.from_numpy(X_norm.astype(np.float32)).unsqueeze(1) # [B, 1, L]
        y_tensor = torch.from_numpy(y.astype(np.float32))
        
        dataset = TensorDataset(X_tensor, y_tensor)
        drop_last = (len(dataset) % self.batch_size == 1)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)
        
        # Class weighting for imbalanced unusable class
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X = np.stack(X) if isinstance(X, list) else np.array(X)
        means = np.mean(X, axis=1, keepdims=True)
        stds = np.std(X, axis=1, keepdims=True)
        stds[stds < 1e-6] = 1.0
        X_norm = (X - means) / stds
            
        X_tensor = torch.from_numpy(X_norm.astype(np.float32)).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.net(batch_x)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p.tolist())
                
        return np.array(probs, dtype=np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
