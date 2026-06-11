import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

_DATA_DIR = Path(__file__).parent

WIND_SPEED  = 6.0   # m/s — constant across the full dataset run
ALPHA_COL   = "pitch_deg"
TARGET_COLS = ["Force X (N)", "Force Y (N)", "Torque X (N-m)"]  # matches model output [F_x, F_y, Moment]


def _load_csv(path: str | Path) -> TensorDataset:
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # alpha: pitch angle in radians  →  shape (N, 1)
    alpha = np.deg2rad(df[ALPHA_COL].values).astype(np.float32)[:, None]

    # velocity: constant 6 m/s for every sample  →  shape (N, 1)
    velocity = np.full_like(alpha, WIND_SPEED)

    # targets: [F_x, F_y, Moment]  →  shape (N, 3)
    targets = df[TARGET_COLS].values.astype(np.float32)

    return TensorDataset(
        torch.from_numpy(velocity),  # batch_v     (N, 1)
        torch.from_numpy(alpha),     # batch_alpha (N, 1)
        torch.from_numpy(targets),   # (N, 3)
    )


def get_dataloaders(
    batch_size: int = 32,
    train_csv: str | Path = _DATA_DIR / "dataset_train.csv",
    val_csv:   str | Path = _DATA_DIR / "dataset_val.csv",
    # legacy kwarg kept for backwards-compat (ignored when using real CSVs)
    num_samples: int = 0,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for the airfoil force/torque dataset."""
    train_ds = _load_csv(train_csv)
    val_ds   = _load_csv(val_csv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

