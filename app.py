import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import requests
from inference_sdk import InferenceHTTPClient

# -------------------------------
# 1. Configuration
# -------------------------------
st.set_page_config(page_title="Philippine Centavo Recognizer", layout="wide")

MODEL_PATH = "coin_model_3class.h5"
ROBOFLOW_API_KEY = "zsOtQhpDpJk2J4HquyBJ"
WORKSPACE_NAME = "jazeels-workspace-pcssh"
WORKFLOW_ID = "detect-count-and-visualize-5"
API_URL = "https://serverless.roboflow.com"

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
    idx = np.argmax(pred)
    raw_label = CLASS_NAMES[idx]
    confidence = pred[idx]
    final_label = '5c' if raw_label.startswith('5c') else '25c'
    return final_label, confidence, raw_label

# -------------------------------
# 3. Roboflow multi‑coin detection
# -------------------------------
@st.cache_resource
def get_roboflow_client():
    return InferenceHTTPClient(api_url=API_URL, api_key=ROBOFLOW_API_KEY)

def detect_multiple_coins(pil_image: Image.Image):
    """Accepts a PIL Image, sends to Roboflow workflow."""
    client = get_roboflow_client()
    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": pil_image},   # pass PIL Image directly
        use_cache=True
    )
    return result

def parse_workflow_result(result):
    """
    Parses the workflow output. The result is typically a list of dictionaries.
    We assume the first element contains 'counts', 'total_coins', 'total_cents',
    and that an annotated image (base64) is in the 'visualization' field.
    """
    annotated_image = None
    counts_5c = 0
    counts_25c = 0
    total_coins = 0
    total_cents = 0

    # The workflow returns a list; take the first output
    if isinstance(result, list) and len(result) > 0:
        output = result[0]
    elif isinstance(result, dict):
        output = result
    else:
        output = {}

    # Extract counts and totals
    counts_5c = output.get("counts", {}).get("5c", 0)
    counts_25c = output.get("counts", {}).get("25c", 0)
    total_coins = output.get("total_coins", 0)
    total_cents = output.get("total_cents", 0)

    # Extract annotated image (base64 string)
    vis = output.get("visualization", {})
    if isinstance(vis, dict):
        b64_image = vis.get("base64") or vis.get("image")
    else:
        b64_image = output.get("annotated_image")

    if b64_image:
        import base64
        # Remove data URL prefix if present
        if isinstance(b64_image, str) and b64_image.startswith("data:image"):
            b64_image = b64_image.split(",")[1]
        try:
            img_bytes = base64.b64decode(b64_image)
            annotated_image = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            st.error(f"Could not decode annotated image: {e}")

    return annotated_image, counts_5c, counts_25c, total_coins, total_cents

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.title("🇵🇭 Philippine Centavo Coin Recognizer")
st.markdown("Upload a coin image – choose between **single coin classification** or **multi‑coin detection**.")

mode = st.radio("Select mode", ["Single Coin", "Multiple Coins"], horizontal=True)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        # Fix deprecation: use width instead of use_container_width
        st.image(original_image, caption="Uploaded Image", width=400)

    if mode == "Single Coin":
        with st.spinner("Classifying single coin..."):
            final_label, conf, raw = predict_single_coin(original_image)
        with col2:
            st.success("### Classification Result")
            st.metric("Predicted Coin", final_label)
            st.metric("Confidence", f"{conf:.2%}")
            st.caption(f"Raw prediction: {raw}")

    else:   # Multi‑coin mode
        with st.spinner("Running Roboflow detection workflow..."):
    result = detect_multiple_coins(original_image) # <-- Your existing line
    # --- Debugging lines (add these) ---
    with st.expander("🔍 Debug: Raw API Response"):
        st.write(result)
        if isinstance(result, list) and len(result) > 0:
            st.write("--- Output 0 Top-Level Keys ---")
            st.write(list(result[0].keys()))
    # --- End of debugging lines ---
    annotated_img, cnt5, cnt25, total_coins, total_cents = parse_workflow_result(result)
...

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
            st.warning("Annotated image not available. Raw result:")
            st.json(result)

else:
    st.info("Please upload an image to start.")

with st.sidebar:
    st.header("🏗️ System Architecture")
    st.markdown("""
    **Two independent models**:

    1. **Single Coin (Classification)**  
       - TensorFlow/Keras MobileNetV2 (frozen base)  
       - Trained on 3 classes: `5c_front`, `5c_back`, `25c`  
       - Input: 224×224, preprocessed with `mobilenet_v2.preprocess_input`  
       - Output: `5c` or `25c` (after mapping)

    2. **Multiple Coins (Detection)**  
       - Roboflow YOLOv12 workflow  
       - Detects 4 classes (front/back for each coin)  
       - Returns annotated image + JSON with counts and total value (5c=5¢, 25c=25¢)  
       - Called via `inference_sdk`

    **Streamlit** integrates both.
    """)
