import numpy as np


def dice_iou(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    intersection = np.logical_and(prediction, truth).sum()
    pred_sum, truth_sum = prediction.sum(), truth.sum()
    dice = 1.0 if pred_sum + truth_sum == 0 else 2.0 * intersection / (pred_sum + truth_sum)
    union = np.logical_or(prediction, truth).sum()
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = image.astype(np.float32)
    color = np.zeros_like(base); color[..., 0] = 255
    alpha = (mask.astype(np.float32) * 0.45)[..., None]
    return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
