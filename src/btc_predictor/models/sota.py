from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("torch is required for N-BEATS") from exc


class FlatSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Dict, horizons: List, lookback: int):
        self.horizons = horizons
        self.lookback = lookback
        
        if len(x) < lookback:
            self.x_flat = np.empty((0, x.shape[1] * lookback), dtype=np.float32)
            self.y_vals = np.empty((0, len(horizons)), dtype=np.float32)
            return

        # Create sliding windows view first
        windows = sliding_window_view(x, window_shape=lookback, axis=0)
        # Shape: (N-L+1, L, F) -> Reshape to (N-L+1, L*F) and materialize
        # Using reshape on a non-contiguous view might trigger copy anyway, but we do it once.
        # We ensure it is contiguous float32.
        self.x_flat = np.ascontiguousarray(windows.reshape(windows.shape[0], -1), dtype=np.float32)

        # Pre-stack y
        y_stacked = np.stack([y[h] for h in horizons], axis=1)
        self.y_vals = np.ascontiguousarray(y_stacked[lookback - 1:], dtype=np.float32)

        if len(self.x_flat) != len(self.y_vals):
            min_len = min(len(self.x_flat), len(self.y_vals))
            self.x_flat = self.x_flat[:min_len]
            self.y_vals = self.y_vals[:min_len]

    def __len__(self):
        return len(self.x_flat)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x_flat[idx]), torch.from_numpy(self.y_vals[idx])


class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, theta_size: int):
        super().__init__()
        layers = []
        in_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = hidden_size
        self.fc = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        h = self.fc(x)
        return self.theta(h)


class NBeatsNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_blocks: int, num_layers: int, dropout: float, out_size: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [NBeatsBlock(input_size, hidden_size, num_layers, dropout, out_size) for _ in range(num_blocks)]
        )

    def forward(self, x):
        forecast = 0
        for block in self.blocks:
            forecast = forecast + block(x)
        return forecast


def quantile_loss(pred, target, quantiles: List[float]):
    q = torch.tensor(quantiles, dtype=torch.float32, device=pred.device).view(1, 1, -1)
    diff = target.unsqueeze(-1) - pred
    loss = torch.max(q * diff, (q - 1) * diff)
    return loss.mean()


@dataclass
class NBeatsQuantileModel:
    input_size: int
    hidden_size: int
    num_blocks: int
    num_layers: int
    dropout: float
    epochs: int
    batch_size: int
    lr: float
    quantiles: List[float]
    horizons: List
    lookback: int
    device: Optional[str] = None

    def __post_init__(self):
        out_size = len(self.horizons) * len(self.quantiles)
        flat_size = self.input_size * self.lookback
        self.net = NBeatsNet(flat_size, self.hidden_size, self.num_blocks, self.num_layers, self.dropout, out_size)
        self.device = _resolve_device(self.device)
        self.net.to(self.device)

    def fit(self, x: np.ndarray, y: Dict) -> "NBeatsQuantileModel":
        self._ensure_device()
        dataset = FlatSequenceDataset(x, y, self.horizons, self.lookback)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()

        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.net(xb)
                pred = pred.view(-1, len(self.horizons), len(self.quantiles))
                loss = quantile_loss(pred, yb, self.quantiles)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def _predict_sequences(self, x: np.ndarray) -> Dict:
        self._ensure_device()
        self.net.eval()
        dataset = FlatSequenceDataset(x, {h: np.zeros(len(x)) for h in self.horizons}, self.horizons, self.lookback)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds = {h: {q: [] for q in self.quantiles} for h in self.horizons}
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                out = self.net(xb).view(-1, len(self.horizons), len(self.quantiles))
                out_np = out.cpu().numpy()
                for hi, h in enumerate(self.horizons):
                    for qi, q in enumerate(self.quantiles):
                        preds[h][q].extend(out_np[:, hi, qi].tolist())
        return preds

    def predict(self, x: np.ndarray, context: Optional[np.ndarray] = None) -> Dict:
        if context is not None and len(context) >= self.lookback - 1:
            x_full = np.vstack([context[-(self.lookback - 1):], x])
            preds = self._predict_sequences(x_full)
            trimmed = {h: {q: np.array(v[-len(x):]) for q, v in qm.items()} for h, qm in preds.items()}
            return trimmed
        preds = self._predict_sequences(x)
        trimmed = {h: {q: np.array(v) for q, v in qm.items()} for h, qm in preds.items()}
        return trimmed

    def _ensure_device(self) -> None:
        desired = _resolve_device(self.device)
        if desired != self.device:
            self.device = desired
        self.net.to(self.device)


def _resolve_device(device: Optional[object]) -> torch.device:
    if device is not None:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            return dev
        if dev.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return dev
        if dev.type == "cpu":
            return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
