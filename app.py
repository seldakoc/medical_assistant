from io import BytesIO
from pathlib import Path
import numpy as np
import streamlit as st

from dataset_utils import index_dataset, preprocess_volume, read_mask, read_rgb, restore_probabilities
from inference import WEIGHTS_PATH, ensure_weights, load_model, predict
from medical_assistant import ask_qwen
from metrics import dice_iou, overlay
from vision_encoder import extract_features, load_encoder

DATASET_ROOT = Path("kaggle_3m")
NOTICE = "Eğitim prototipidir; klinik kullanım için doğrulanmamıştır. Sonuçlar uzman değerlendirmesi gerektirir."

st.set_page_config(page_title="Medical Assistant", layout="wide")
st.title("MRI Medical Assistant")
st.warning(NOTICE)
st.caption("MRI segmentation, visual feature extraction, and educational image assessment")


@st.cache_resource
def get_segmentation_model():
    return load_model(ensure_weights(), "cpu")


@st.cache_resource
def get_encoder():
    return load_encoder("cpu")


@st.cache_data
def get_index():
    return index_dataset(DATASET_ROOT)


def run_volume(images: list[np.ndarray]):
    volume = np.stack(images)
    prepared, meta = preprocess_volume(volume)
    probabilities = predict(get_segmentation_model(), prepared, "cpu")
    return restore_probabilities(probabilities, meta)


@st.cache_data(show_spinner=False)
def run_local_patient(image_paths: tuple[str, ...]):
    """Read and segment a patient once instead of on every Streamlit rerun."""
    images = [read_rgb(path) for path in image_paths]
    if len({item.shape for item in images}) != 1:
        raise ValueError("Hasta hacminde uyumsuz kesit boyutları var.")
    return images, run_volume(images)


@st.cache_data(show_spinner=False)
def run_uploaded_image(file_bytes: bytes):
    image = read_rgb(BytesIO(file_bytes))
    return image, run_volume([image])[0]


@st.cache_data(show_spinner=False)
def cached_features(image_bytes: bytes, shape: tuple[int, ...]):
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
    encoder, transform = get_encoder()
    return extract_features(encoder, transform, image)


threshold = st.sidebar.slider("Segmentasyon eşiği", 0.0, 1.0, 0.5, 0.01)
source = st.sidebar.radio("Görüntü kaynağı", ["Yerel veri kümesi", "TIFF yükle"])

try:
    truth = None
    if source == "Yerel veri kümesi":
        index, warnings = get_index()
        if not index:
            st.error("Veri kümesinde MRI TIFF görüntüsü bulunamadı.")
            st.stop()
        patient = st.sidebar.selectbox("Hasta", list(index))
        pairs = index[patient]
        selected = st.sidebar.selectbox(
            "MRI kesiti", range(len(pairs)), format_func=lambda i: pairs[i].image.name
        )
        with st.spinner("Segmentasyon modeli çalışıyor..."):
            images, probabilities = run_local_patient(tuple(str(item.image) for item in pairs))
        pair, image, probability = pairs[selected], images[selected], probabilities[selected]
        truth = read_mask(pair.mask, image.shape[:2]) if pair.mask else None
        if warnings:
            st.info(f"Dizinleme sırasında {len(warnings)} uyarı oluştu.")
    else:
        uploaded = st.file_uploader("3 kanallı .tif/.tiff MRI görüntüsü yükleyin", type=["tif", "tiff"])
        if uploaded is None:
            st.info("Başlamak için bir TIFF görüntüsü seçin.")
            st.stop()
        image, probability = run_uploaded_image(uploaded.getvalue())

    predicted = probability >= threshold
    cols = st.columns(3)
    cols[0].image(image, caption="Orijinal MRI", use_container_width=True)
    cols[1].image((predicted * 255).astype(np.uint8), caption="U-Net tahmini", clamp=True, use_container_width=True)
    cols[2].image(overlay(image, predicted), caption="İşaretli görüntü", use_container_width=True)

    area_percent = float(predicted.mean() * 100)
    st.metric("İşaretlenen piksel oranı", f"%{area_percent:.2f}")
    st.caption("Bu oran tanı, tümör tipi veya evre değildir.")
    if truth is not None:
        dice, iou = dice_iou(predicted, truth)
        m1, m2 = st.columns(2)
        m1.metric("Dice (gösterim)", f"{dice:.4f}")
        m2.metric("IoU (gösterim)", f"{iou:.4f}")

    st.subheader("Pretrained EfficientNet-B0")
    with st.spinner("Dondurulmuş encoder görüntü özelliklerini çıkarıyor..."):
        features = cached_features(image.tobytes(), image.shape)
    st.success(f"{features.model_name}: {features.embedding.size} boyutlu özellik vektörü üretildi.")
    st.caption("Fine-tune yapılmadı. Bu vektör tıbbi sınıf veya hastalık olasılığı olarak yorumlanmaz.")

    st.subheader("Medical Assistant")
    st.caption("The assistant will assess the selected image automatically; you do not need to enter a question.")
    if st.button("Analyze image", type="primary"):
        automatic_request = (
            "Review this brain MRI slice as an educational clinical case. Describe relevant imaging "
            "findings, give a prioritized differential diagnosis with reasoning, recommend appropriate "
            "next investigations, and explain typical treatment pathways—including example medication "
            "classes or medicines—if a suspected condition is confirmed. Do not prescribe or give doses."
        )
        with st.spinner("Medical Assistant is analyzing the image..."):
            answer = ask_qwen(image, automatic_request, area_percent)
        st.markdown(answer)
        st.caption("The Medical Assistant is configured to answer in English.")
        st.warning("Bu yanıt reçete veya tanı değildir. İlaç başlamayın/bırakmayın; bir hekim ve radyoloji raporuyla doğrulayın.")

except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
    st.error(str(exc))
    if not WEIGHTS_PATH.exists():
        st.caption("U-Net ağırlıkları ilk çalıştırmada weights/ klasörüne indirilir.")
