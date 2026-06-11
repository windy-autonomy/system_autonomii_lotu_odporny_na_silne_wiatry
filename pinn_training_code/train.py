import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from dataset import get_dataloaders
from pin_model import NNAeroModel
from clasical import TrigAeroModel
from rbf_model import RBFAeroModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")  # for better precision in Trig/RBF models

SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_loss(model: nn.Module, batch_v, batch_alpha, batch_targets, criterion) -> torch.Tensor:
    """Run a forward pass and return the loss."""
    predictions = model(batch_v, batch_alpha)
    return criterion(predictions, batch_targets)


CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


def train_model(model: nn.Module, train_loader, val_loader, epochs=40, lr=0.005, name="Model"):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    # Compile the model for faster execution (requires PyTorch >= 2.0)
    compiled = torch.compile(model)

    print(f"\n[{name}]  (device: {DEVICE})")
    pbar = tqdm(range(epochs), desc="Epochs", unit="ep")
    train_loss = 0.0
    val_loss = 0.0
    best_val_loss = float("inf")
    best_state = None

    for epoch in pbar:
        # --- training ---
        compiled.train()
        epoch_train_loss = 0.0
        for batch_v, batch_alpha, batch_targets in train_loader:
            batch_v       = batch_v.to(DEVICE)
            batch_alpha   = batch_alpha.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            optimizer.zero_grad()
            loss = calc_loss(compiled, batch_v, batch_alpha, batch_targets, criterion)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_loss = epoch_train_loss / len(train_loader)

        # --- validation ---
        compiled.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_v, batch_alpha, batch_targets in val_loader:
                batch_v       = batch_v.to(DEVICE)
                batch_alpha   = batch_alpha.to(DEVICE)
                batch_targets = batch_targets.to(DEVICE)
                loss = calc_loss(compiled, batch_v, batch_alpha, batch_targets, criterion)
                epoch_val_loss += loss.item()
        val_loss = epoch_val_loss / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", best=f"{best_val_loss:.4f}")

    # Restore and save best weights
    model.load_state_dict(best_state)
    ckpt_path = CHECKPOINTS_DIR / f"{name.replace(' ', '_').lower()}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"  Best val loss: {best_val_loss:.4f}  →  saved to {ckpt_path}")
    return train_loss, best_val_loss


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=64)

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"Dataset  →  train: {n_train} samples  |  val: {n_val} samples  |  total: {n_train + n_val}")

    # Collect results per model across seeds: {model_name: {"train": [], "val": []}}
    all_results: dict[str, dict[str, list[float]]] = {}

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"  SEED {seed}")
        print(f"{'='*50}")
        set_seed(seed)

        models = {
            "Neural Network": NNAeroModel(),
            "Trigonometric":  TrigAeroModel(degree=2),
            "RBF Network":    RBFAeroModel(num_centers=5),
        }

        for name, model in models.items():
            train_loss, val_loss = train_model(
                model, train_loader, val_loader, epochs=40, lr=0.005, name=f"{name} [seed={seed}]"
            )
            all_results.setdefault(name, {"train": [], "val": []})
            all_results[name]["train"].append(train_loss)
            all_results[name]["val"].append(val_loss)

    # --- Statistics summary ---
    print("\n" + "=" * 62)
    print("  STATISTICS OVER 5 SEEDS")
    print("=" * 62)
    print(f"{'Model':<20} {'Train MSE':>12}  {'Val MSE':>12}")
    print(f"{'':20} {'mean ± std':>12}  {'mean ± std':>12}")
    print("-" * 62)
    for name, metrics in all_results.items():
        t = np.array(metrics["train"])
        v = np.array(metrics["val"])
        print(
            f"{name:<20} {t.mean():>7.4f}±{t.std():>6.4f}  {v.mean():>7.4f}±{v.std():>6.4f}"
        )
    print("=" * 62)

    # Per-seed breakdown
    print("\n--- Per-seed val MSE ---")
    header = f"{'Model':<20}" + "".join(f"  seed{s}" for s in SEEDS)
    print(header)
    print("-" * len(header))
    for name, metrics in all_results.items():
        row = f"{name:<20}" + "".join(f"  {v:>6.4f}" for v in metrics["val"])
        print(row)

    # Inference comparison (using last trained models)
    print("\n--- Inference at V=6 m/s, alpha=15 deg (last seed) ---")
    test_v     = torch.tensor([[6.0]]).to(DEVICE)
    test_alpha = torch.tensor([[15.0 * torch.pi / 180.0]]).to(DEVICE)

    with torch.no_grad():
        for name, model in models.items():
            model = model.to(DEVICE)
            pred = model(test_v, test_alpha)[0]
            print(f"[{name}] F_x: {pred[0]:.4f} N  |  F_y: {pred[1]:.4f} N  |  Moment: {pred[2]:.4f} N·m")