"""Ollama client for the local Qwen2.5-VL educational assistant."""

import base64
from io import BytesIO
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import numpy as np
from PIL import Image

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5vl:3b"

SYSTEM_PROMPT = """You are an educational clinical decision-support assistant
reviewing a brain MRI slice. Always answer in English. Use professional clinical
language, but do not claim to be the treating doctor. Do not make a definitive diagnosis
from one image. Distinguish observed findings from hypotheses.

Structure the answer under these headings:
1. Imaging findings
2. Clinical impression and differential diagnosis
3. Recommended clinical evaluation
4. Typical treatment options if the suspected condition is confirmed
5. Urgent warning signs

Give a prioritized differential diagnosis when image quality permits. Explain
which history, examination, MRI sequences, contrast imaging, or tests would help
confirm it. In the treatment section, discuss realistic treatment pathways and
may name commonly used medication classes or example medicines for educational
context. Clearly state that selection, contraindications, and dosing require a
qualified clinician. Never provide a patient-specific prescription, dosage, or
instructions to start or stop medication. Keep the response clinically useful
and concise."""


def build_prompt(question: str, abnormal_area_percent: float | None = None) -> str:
    context = ""
    if abnormal_area_percent is not None:
        context = (
            f"\nArea marked by the separate U-Net prototype: {abnormal_area_percent:.2f}%. "
            "This value is not a diagnosis and may contain false positives or false negatives."
        )
    return f"User question: {question.strip() or 'Explain this image for educational purposes.'}{context}"


def _jpeg_base64(image: np.ndarray) -> str:
    buffer = BytesIO()
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    pil_image.save(buffer, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def ask_qwen(image: np.ndarray, question: str, abnormal_area_percent: float | None = None,
             model: str = DEFAULT_MODEL, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, abnormal_area_percent),
             "images": [_jpeg_base64(image)]},
        ],
        "options": {"temperature": 0.2, "num_predict": 420, "num_ctx": 3072},
        "keep_alive": "10m",
    }
    request = Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Medical Assistant yanıt vermedi. Yerel model servisinin açık ve gerekli "
            f"görsel modelin yüklü olduğunu denetleyin: {exc}"
        ) from exc
    content = result.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Medical Assistant boş yanıt döndürdü.")
    return content
