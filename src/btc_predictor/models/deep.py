from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("torch is required for deep models") from exc


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Dict, horizons: List, lookback: int):
        self.x = x
        self.y = y
        self.horizons = horizons
        self.lookback = lookback
        self.indices = list(range(lookback - 1, len(x)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_seq = self.x[i - self.lookback + 1 : i + 1]
        y_vals = np.stack([self.y[h][i] for h in self.horizons], axis=0)
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_vals, dtype=torch.float32)


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

    def __post_init__(self):
        out_size = len(self.horizons) * len(self.quantiles)
        self.net = LSTMNet(self.input_size, self.hidden_size, self.num_layers, self.dropout, out_size)

    def fit(self, x: np.ndarray, y: Dict) -> "LSTMQuantileModel":
        dataset = SequenceDataset(x, y, self.horizons, self.lookback)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()

        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self.net(xb)
                pred = pred.view(-1, len(self.horizons), len(self.quantiles))
                loss = quantile_loss(pred, yb, self.quantiles)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def _predict_sequences(self, x: np.ndarray) -> Dict:
        self.net.eval()
        dataset = SequenceDataset(x, {h: np.zeros(len(x)) for h in self.horizons}, self.horizons, self.lookback)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds = {h: {q: [] for q in self.quantiles} for h in self.horizons}
        with torch.no_grad():
            for xb, _ in loader:
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
