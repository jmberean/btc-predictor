# Performance Optimization TODO

## High Impact
- [x] **LightGBM Parallelism**
    - [x] Update `LightGBMQuantileModel` to accept `n_jobs`.
    - [x] Pass `n_jobs` to `Parallel` execution to prevent CPU oversubscription.
    - [x] Update `train.py` to pass config value.
- [x] **Deep Learning Data Loading**
    - [x] Optimize `LSTM` `SequenceDataset` (pre-materialize contiguous arrays).
    - [x] Optimize `N-BEATS` `FlatSequenceDataset`.
    - [x] Enable `shuffle=True` for training loaders.

## Medium Impact
- [x] **Data Ingestion**
    - [x] Enforce `float32` dtypes in `binance_bulk` CSV parsing.
    - [x] Use `engine="c"` for pandas read_csv.

## Low Impact
- [x] **Feature Engineering**
    - [x] Vectorize rolling statistics (`.agg(["mean", "std"])`).

## Validation
- [x] **Unit Verification**: Verify class initialization and data shapes.
- [x] **Integration Test**: Run full training loop with `configs/verification.yaml`.