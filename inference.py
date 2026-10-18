"""Safe weight download/loading and CPU-first inference."""

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

from model import UNet

WEIGHTS_URL = "https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/download/v1.0/unet-e012d006.pt"
WEIGHTS_PATH = Path("weights/unet-e012d006.pt")


def ensure_weights(path: Path = WEIGHTS_PATH) -> Path:
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        with urlopen(WEIGHTS_URL, timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Önceden eğitilmiş ağırlık indirilemedi: {exc}") from exc
    return path


def load_model(path: Path = WEIGHTS_PATH, device: str = "cpu") -> UNet:
    if not path.exists():
        raise FileNotFoundError(f"Model ağırlığı bulunamadı: {path}")
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    # weights_only avoids general pickle object construction on supported PyTorch versions.
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict(model: UNet, volume: np.ndarray, device: str = "cpu", batch_size: int = 8) -> np.ndarray:
    batches = []
    with torch.inference_mode():
        for start in range(0, len(volume), batch_size):
            x = torch.from_numpy(volume[start:start + batch_size].transpose(0, 3, 1, 2)).to(device)
            batches.append(model(x)[:, 0].cpu().numpy())
    return np.concatenate(batches)
