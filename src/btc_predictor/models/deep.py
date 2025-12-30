from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("torch is required for deep models") from exc


from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("torch is required for deep models") from exc


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Dict, horizons: List, lookback: int):
        # x shape: (N, Features)
        # Create sliding windows. Shape: (N - lookback + 1, lookback, Features)
        # We need to ensure we align with y.
        # If x has length N, and we need lookback L:
        # The first valid sequence ends at index L-1 (0-based).
        # The corresponding target is at index L-1.
        
        self.lookback = lookback
        self.horizons = horizons
        
        if len(x) < lookback:
             self.x_windows = np.empty((0, lookback, x.shape[1]))
             self.y_array = np.empty((0, len(horizons)))
             return

        # sliding_window_view returns (N - L + 1, F, L) when sliding over axis 0 of (N, F) ?
        # Wait, if x is (N, F).
        # Docs: "The window dimensions are added at the end of the output shape."
        # If I slide over axis 0: shape becomes (N-L+1, F, L).
        # We want (N-L+1, L, F).
        # So we need to swap the last two axes?
        # Actually no, verify this.
        # If x is 1D (N,), result is (N-L+1, L).
        # If x is 2D (N, F), and we slide axis 0.
        # The result shape is (N-L+1, F, L).
        # Yes, we need to swap axes 1 and 2.
        
        windows = sliding_window_view(x, window_shape=lookback, axis=0)
        self.x_windows = np.swapaxes(windows, 1, 2)
        
        # Pre-stack y. Each y[h] is (N,). Stack to (N, H).
        # We need to slice y to match the valid windows.
        # The sliding view's 0-th element corresponds to x[0:lookback], ending at x[lookback-1].
        # So we need y starting from lookback-1.
        y_stacked = np.stack([y[h] for h in horizons], axis=1) # (N, H)
        self.y_array = y_stacked[lookback - 1:]

        # Safety check
        if len(self.x_windows) != len(self.y_array):
            # This might happen if x and y lengths differed slightly, though they shouldn't.
            min_len = min(len(self.x_windows), len(self.y_array))
            self.x_windows = self.x_windows[:min_len]
            self.y_array = self.y_array[:min_len]

    def __len__(self):
        return len(self.x_windows)

    def __getitem__(self, idx):
        # Returns (seq_len, features), (horizons,)
        # Data is already float (usually), but ensure tensor conversion
        return (
            torch.as_tensor(self.x_windows[idx], dtype=torch.float32),
            torch.as_tensor(self.y_array[idx], dtype=torch.float32)
        )


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, out_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def quantile_loss(pred, target, quantiles: List[float]):
    q = torch.tensor(quantiles, dtype=torch.float32, device=pred.device).view(1, 1, -1)
    diff = target.unsqueeze(-1) - pred
    loss = torch.max(q * diff, (q - 1) * diff)
    return loss.mean()


@dataclass
class LSTMQuantileModel:
    input_size: int
    hidden_size: int
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
        self.net = LSTMNet(self.input_size, self.hidden_size, self.num_layers, self.dropout, out_size)
        self.device = _resolve_device(self.device)
        self.net.to(self.device)

    def fit(self, x: np.ndarray, y: Dict) -> "LSTMQuantileModel":
        self._ensure_device()
        dataset = SequenceDataset(x, y, self.horizons, self.lookback)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
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
        dataset = SequenceDataset(x, {h: np.zeros(len(x)) for h in self.horizons}, self.horizons, self.lookback)
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
            trimmed = {
                h: {q: np.array(v[-len(x):]) for q, v in qm.items()}
                for h, qm in preds.items()
            }
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
