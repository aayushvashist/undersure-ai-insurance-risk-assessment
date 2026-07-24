"""Small shared helpers for saving figures and metrics consistently."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = {
    "Low": "#2E7D32",
    "Medium": "#F9A825",
    "High": "#C62828",
}


def save_fig(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure saved] {path}")


def load_metrics(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def update_metrics(path: Path, section: str, data: dict):
    """Merge `data` under `section` into the shared metrics.json file."""
    metrics = load_metrics(path)
    metrics[section] = data
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[metrics saved] {path} -> section '{section}'")
