"""Frozen ImageNet EfficientNet-B0 feature encoder."""

from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class EncoderResult:
    embedding: np.ndarray
    model_name: str = "EfficientNet-B0 (ImageNet-1K pretrained, frozen)"


def load_encoder(device: str = "cpu"):
    """Load the pretrained backbone without changing or fine-tuning its weights."""
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as exc:
        raise RuntimeError(
            "torchvision kurulu değil. `python -m pip install -r requirements.txt` çalıştırın."
        ) from exc
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.requires_grad_(False)
    return model.to(device).eval(), weights.transforms()


def extract_features(model, transform, image: np.ndarray, device: str = "cpu") -> EncoderResult:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model(tensor)[0].float().cpu().numpy()
    return EncoderResult(embedding=embedding)
