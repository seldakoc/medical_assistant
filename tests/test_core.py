from pathlib import Path

import numpy as np
import torch

from dataset_utils import index_dataset, preprocess_volume, restore_probabilities
from inference import load_model
from metrics import dice_iou
from medical_assistant import SYSTEM_PROMPT, build_prompt


def test_dataset_pairs():
    index, _ = index_dataset(Path("kaggle_3m"))
    assert len(index) == 110
    assert sum(map(len, index.values())) == 3929
    assert all(pair.mask is not None for pairs in index.values() for pair in pairs)


def test_metrics_empty_policy():
    empty = np.zeros((4, 4), bool)
    one = empty.copy(); one[0, 0] = True
    assert dice_iou(empty, empty) == (1.0, 1.0)
    assert dice_iou(empty, one) == (0.0, 0.0)
    assert dice_iou(one, one) == (1.0, 1.0)


def test_preprocess_and_restore_shape():
    volume = np.zeros((2, 40, 30, 3), np.uint8)
    volume[:, 5:35, 4:26] = np.arange(3, dtype=np.uint8) + 10
    prepared, meta = preprocess_volume(volume)
    assert prepared.shape == (2, 256, 256, 3)
    restored = restore_probabilities(np.zeros((2, 256, 256), np.float32), meta)
    assert restored.shape == volume.shape[:3]


def test_model_loading_and_output_shape():
    model = load_model()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 256, 256))
    assert output.shape == (1, 1, 256, 256)
    assert torch.isfinite(output).all()


def test_assistant_prompt_has_safety_boundary():
    lowered = SYSTEM_PROMPT.lower()
    assert "always answer in english" in lowered
    assert "definitive diagnosis" in lowered
    assert "dosage" in lowered
    assert "differential diagnosis" in lowered
    assert "example medicines" in lowered
    prompt = build_prompt("Açıkla", 12.345)
    assert "12.35%" in prompt
    assert "not a diagnosis" in prompt
