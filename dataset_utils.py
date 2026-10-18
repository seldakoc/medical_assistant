"""Dataset discovery, validation, and original volume preprocessing."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.transform import resize


@dataclass(frozen=True)
class SlicePair:
    patient: str
    image: Path
    mask: Path | None
    slice_number: int


def _slice_number(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.stem.removesuffix("_mask"))
    return int(match.group(1)) if match else -1


def index_dataset(root: Path) -> tuple[dict[str, list[SlicePair]], list[str]]:
    """Index each physical TIFF once, using relative paths and exact mask suffixes."""
    if not root.is_dir():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {root}")
    images = [p for p in root.rglob("*.tif") if not p.stem.endswith("_mask")]
    result: dict[str, list[SlicePair]] = {}
    warnings: list[str] = []
    seen: set[Path] = set()
    for image in images:
        resolved = image.resolve()
        if resolved in seen:
            warnings.append(f"Yinelenen yol atlandı: {image}")
            continue
        seen.add(resolved)
        mask = image.with_name(image.stem + "_mask" + image.suffix)
        pair = SlicePair(image.parent.name, image, mask if mask.exists() else None, _slice_number(image))
        result.setdefault(pair.patient, []).append(pair)
    for patient in result:
        result[patient].sort(key=lambda p: (p.slice_number, p.image.name))
    return dict(sorted(result.items())), warnings


def read_rgb(path_or_file) -> np.ndarray:
    try:
        with Image.open(path_or_file) as im:
            array = np.asarray(im.convert("RGB"))
    except Exception as exc:
        raise ValueError(f"TIFF görüntüsü okunamadı: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 16:
        raise ValueError(f"Uyumsuz görüntü boyutu: {array.shape}; H×W×3 bekleniyor.")
    return array


def read_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(path) as im:
            mask = np.asarray(im.convert("L")) > 0
    except Exception as exc:
        raise ValueError(f"Maske okunamadı ({path.name}): {exc}") from exc
    if mask.shape != expected_hw:
        raise ValueError(f"Görüntü/maske boyutları farklı: {expected_hw} ve {mask.shape}")
    return mask


def validate_index(index: dict[str, list[SlicePair]]) -> dict[str, object]:
    unreadable, missing, incompatible = [], [], []
    count = 0
    for pairs in index.values():
        for pair in pairs:
            count += 1
            if pair.mask is None:
                missing.append(str(pair.image))
                continue
            try:
                image = read_rgb(pair.image)
                read_mask(pair.mask, image.shape[:2])
            except ValueError as exc:
                (incompatible if "boyut" in str(exc) else unreadable).append(str(exc))
    return {"images": count, "patients": len(index), "missing_masks": missing,
            "unreadable": unreadable, "incompatible": incompatible}


def preprocess_volume(volume: np.ndarray, size: int = 256) -> tuple[np.ndarray, dict]:
    """Match the repository: crop, square-pad, cubic resize, p10/p99, channel z-score."""
    work = volume.astype(np.float32).copy()
    work[work < np.max(work) * 0.1] = 0
    active_z = np.where(np.max(work, axis=(1, 2, 3)) > 0)[0]
    active_y = np.where(np.max(work, axis=(0, 2, 3)) > 0)[0]
    active_x = np.where(np.max(work, axis=(0, 1, 3)) > 0)[0]
    if not (active_z.size and active_y.size and active_x.size):
        raise ValueError("Görüntüde ön işleme için yeterli sıfır olmayan sinyal yok.")
    bounds = (slice(active_z[0], active_z[-1] + 1), slice(active_y[0], active_y[-1] + 1), slice(active_x[0], active_x[-1] + 1))
    cropped = work[bounds]
    h, w = cropped.shape[1:3]
    dy, dx = max(0, w - h), max(0, h - w)
    pad = ((0, 0), (dy // 2, dy - dy // 2), (dx // 2, dx - dx // 2), (0, 0))
    padded = np.pad(cropped, pad, mode="constant")
    resized = resize(padded, (padded.shape[0], size, size, 3), order=2, mode="constant", cval=0, anti_aliasing=False)
    p10, p99 = np.percentile(resized, (10, 99))
    scaled = rescale_intensity(resized, in_range=(p10, p99)) if p99 > p10 else resized
    mean, std = scaled.mean(axis=(0, 1, 2)), scaled.std(axis=(0, 1, 2))
    if np.any(std == 0):
        raise ValueError("Bir veya daha fazla görüntü kanalı sabit; normalizasyon yapılamadı.")
    normalized = (scaled - mean) / std
    meta = {"bounds": bounds, "pad": pad, "padded_hw": padded.shape[1:3], "original_shape": volume.shape}
    return normalized.astype(np.float32), meta


def restore_probabilities(probabilities: np.ndarray, meta: dict) -> np.ndarray:
    z, h, w = probabilities.shape
    restored = resize(probabilities, (z, *meta["padded_hw"]), order=1, mode="constant", cval=0, anti_aliasing=False)
    pad = meta["pad"]
    restored = restored[:, pad[1][0]: restored.shape[1] - pad[1][1] if pad[1][1] else None,
                        pad[2][0]: restored.shape[2] - pad[2][1] if pad[2][1] else None]
    output = np.zeros(meta["original_shape"][:3], dtype=np.float32)
    output[meta["bounds"]] = restored
    return output
