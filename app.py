import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import base64
import requests

# -------------------------------
# 1. Configuration
# -------------------------------
st.set_page_config(page_title="Philippine Centavo Recognizer", layout="wide")

MODEL_PATH = "coin_model_3class_improved.h5"
ROBOFLOW_API_KEY = "zsOtQhpDpJk2J4HquyBJ"
WORKSPACE_NAME = "jazeels-workspace-pcssh"
WORKFLOW_ID = "detect-count-and-visualize-5"
WORKFLOW_URL = f"https://serverless.roboflow.com/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}"

# -------------------------------
# 2. Load classification model (cached)
# -------------------------------
@st.cache_resource
def load_classification_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

CLASS_NAMES = ['5c_front', '5c_back', '25c']

def predict_single_coin(image: Image.Image):
    model = load_classification_model()
    img = image.convert('RGB').resize((224, 224))
    x = np.array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    sorted_indices = np.argsort(pred)[::-1]
    top_idx = sorted_indices[0]
    second_idx = sorted_indices[1]
    raw_label_top = CLASS_NAMES[top_idx]
    confidence_top = pred[top_idx]
    final_label = '5c' if raw_label_top.startswith('5c') else '25c'
    raw_label_second = CLASS_NAMES[second_idx]
    confidence_second = pred[second_idx]
    final_label_second = '5c' if raw_label_second.startswith('5c') else '25c'
    return (final_label, confidence_top, raw_label_top,
            final_label_second, confidence_second, raw_label_second)

# -------------------------------
# 3. Roboflow multi‑coin detection (pure REST)
# -------------------------------
def detect_multiple_coins(pil_image: Image.Image):
    # Convert PIL image to Base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "inputs": {
            "image": {
                "type": "base64",
                "value": img_base64
            }
        }
    }
    headers = {"Content-Type": "application/json"}
    params = {"api_key": ROBOFLOW_API_KEY}

    response = requests.post(WORKFLOW_URL, json=payload, headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.text}")
        return {}
    return response.json()

def parse_workflow_result(result):
    annotated_image = None
    counts = {"5c": 0, "25c": 0}

    if not isinstance(result, dict):
        st.error(f"Unexpected result type: {type(result)}")
        return None, 0, 0, 0, 0

    # The response from a workflow usually contains an "outputs" list
    outputs = result.get("outputs", [])
    if outputs:
        # Take the first output (your workflow likely has one output)
        output = outputs[0]
    else:
        # Fallback: maybe the result is the output itself
        output = result

    # Extract annotated image (base64) – the field name varies
    b64_image = output.get("output_image") or output.get("image") or output.get("visualization")
    if b64_image:
        try:
            if isinstance(b64_image, str) and b64_image.startswith("data:image"):
                b64_image = b64_image.split(",")[1]
            img_bytes = base64.b64decode(b64_image)
            annotated_image = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            st.warning(f"Image decode error: {e}")

    # Extract predictions
    predictions = output.get("predictions", [])
    # If predictions is a dict with a "predictions" key (like your earlier output)
    if isinstance(predictions, dict):
        predictions = predictions.get("predictions", [])

    for det in predictions:
        class_name = det.get("class", "")
        confidence = det.get("confidence", 0)
        if confidence < 0.5:
            continue
        if class_name.startswith("5c"):
            counts["5c"] += 1
        elif class_name.startswith("25c"):
            counts["25c"] += 1

    total_coins = counts["5c"] + counts["25c"]
    total_cents = counts["5c"] * 5 + counts["25c"] * 25

    return annotated_image, counts["5c"], counts["25c"], total_coins, total_cents

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.title("🇵🇭 Philippine Centavo Coin Recognizer")
st.markdown("Upload a coin image – choose between **single coin classification** or **multi‑coin detection**.")

mode = st.radio("Select mode", ["Single Coin", "Multiple Coins"], horizontal=True)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

debug_result = None
show_debug = False

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Uploaded Image", width=400)

    if mode == "Single Coin":
        with st.spinner("Classifying single coin..."):
            (final_label, conf, raw,
             second_label, second_conf, second_raw) = predict_single_coin(original_image)
        with col2:
            st.success("### Classification Result")
            st.metric("Predicted Coin", final_label)
            st.metric("Confidence", f"{conf:.2%}")
            st.caption(f"Raw prediction: {raw}")
            st.markdown(
                f"<p style='font-size:0.8rem; color:gray;'>Other possibility: {second_label} ({second_conf:.2%}) — raw: {second_raw}</p>",
                unsafe_allow_html=True
            )

    else:   # Multiple Coins mode
        with st.spinner("Running Roboflow detection workflow..."):
            result = detect_multiple_coins(original_image)
            annotated_img, cnt5, cnt25, total_coins, total_cents = parse_workflow_result(result)

        with col2:
            st.success("### Detection Summary")
            col2a, col2b, col2c = st.columns(3)
            col2a.metric("5c Count", cnt5)
            col2b.metric("25c Count", cnt25)
            col2c.metric("Total Coins", total_coins)
            st.metric("Total Value", f"{total_cents} cents")

        if annotated_img:
            st.image(annotated_img, caption="Annotated Output", width=600)
        else:
            st.warning("Annotated image not available.")

        debug_result = result
        show_debug = True

else:
    st.info("Please upload an image to start.")

# -------------------------------
# 5. Debug output at the bottom (only in multi-coin mode)
# -------------------------------
if show_debug and debug_result is not None:
    with st.expander("🔍 Debug: Raw API Response (image omitted)"):
        result_copy = debug_result
        if isinstance(result_copy, dict):
            result_copy = result_copy.copy()
            for field in ["output_image", "visualization", "image", "base64"]:
                result_copy.pop(field, None)
        st.json(result_copy)

# -------------------------------
# 6. Sidebar
# -------------------------------
with st.sidebar:
    st.header("🏗️ System Architecture")
    st.markdown("""
    **Two independent models**:

    1. **Single Coin (Classification)**  
       - TensorFlow/Keras MobileNetV2 (frozen base)  
       - Trained on 3 classes: `5c_front`, `5c_back`, `25c`  
       - Output: `5c` or `25c`

    2. **Multiple Coins (Detection)**  
       - Roboflow YOLOv12 workflow  
       - Called via **pure REST API** – no OpenCV, no system dependencies
       - Returns annotated image + counts + total value
    """)
