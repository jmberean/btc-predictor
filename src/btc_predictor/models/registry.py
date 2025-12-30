from typing import Dict, List

from btc_predictor.config import get_base_frequency, get_horizon_labels, get_horizons, horizon_to_steps
from btc_predictor.models.baselines import HistoricalQuantileBaseline, RandomWalkBaseline


def build_base_steps(cfg) -> Dict:
    horizons = get_horizons(cfg)
    labels = get_horizon_labels(cfg)
    base = get_base_frequency(cfg)
    return {label: horizon_to_steps(h, base) for label, h in zip(labels, horizons)}


def init_model(name: str, cfg, input_size: int) -> object:
    horizons = get_horizon_labels(cfg)
    quantiles = cfg["training"]["quantiles"]
    base_steps = build_base_steps(cfg)

    if name == "naive":
        return HistoricalQuantileBaseline(quantiles=quantiles, horizons=horizons)
    if name == "random_walk":
        return RandomWalkBaseline(quantiles=quantiles, horizons=horizons)
    if name == "arima":
        from btc_predictor.models.arima import ARIMAQuantileModel

        return ARIMAQuantileModel(order=(1, 0, 1), quantiles=quantiles, horizons=horizons, base_steps=base_steps)
    if name == "garch":
        from btc_predictor.models.garch import GARCHQuantileModel

        return GARCHQuantileModel(quantiles=quantiles, horizons=horizons, base_steps=base_steps)
    if name == "lightgbm":
        from btc_predictor.models.tree import LightGBMQuantileModel

        params = cfg.get("lightgbm", {})
        params = dict(params)
        params.setdefault("random_state", cfg.get("seed", 42))
        params.setdefault("n_jobs", cfg["training"].get("n_jobs", -1))
        return LightGBMQuantileModel(params=params, quantiles=quantiles, horizons=horizons)
    if name == "lstm":
        from btc_predictor.models.deep import LSTMQuantileModel

        params = cfg.get("lstm", {})
        return LSTMQuantileModel(
            input_size=input_size,
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.1),
            epochs=params.get("epochs", 5),
            batch_size=params.get("batch_size", 128),
            lr=params.get("lr", 1e-3),
            quantiles=quantiles,
            horizons=horizons,
            lookback=cfg["features"].get("lookback_window", 48),
            device=params.get("device"),
        )
    if name == "nbeats":
        from btc_predictor.models.sota import NBeatsQuantileModel

        params = cfg.get("nbeats", {})
        return NBeatsQuantileModel(
            input_size=input_size,
            hidden_size=params.get("hidden_size", 256),
            num_blocks=params.get("num_blocks", 2),
            num_layers=params.get("num_layers", 4),
            dropout=params.get("dropout", 0.1),
            epochs=params.get("epochs", 5),
            batch_size=params.get("batch_size", 128),
            lr=params.get("lr", 1e-3),
            quantiles=quantiles,
            horizons=horizons,
            lookback=cfg["features"].get("lookback_window", 48),
            device=params.get("device"),
        )

    raise ValueError(f"Unknown model: {name}")
