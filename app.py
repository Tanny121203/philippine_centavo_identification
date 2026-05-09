import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import base64
from inference_sdk import InferenceHTTPClient

# -------------------------------
# 1. Configuration
# -------------------------------
st.set_page_config(page_title="Philippine Centavo Recognizer", layout="wide")

MODEL_PATH = "coin_model_3class_improved.h5"
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
    client = get_roboflow_client()
    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": pil_image},
        use_cache=True
    )
    return result

def parse_workflow_result(result):
    annotated_image = None
    counts = {"5c": 0, "25c": 0}

    # Unwrap if result is a list
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if not isinstance(result, dict):
        st.error(f"Unexpected result type: {type(result)}")
        return None, 0, 0, 0, 0

    # Extract annotated image
    b64_image = result.get("output_image") or result.get("image") or result.get("visualization")
    if b64_image:
        try:
            if isinstance(b64_image, str) and b64_image.startswith("data:image"):
                b64_image = b64_image.split(",")[1]
            img_bytes = base64.b64decode(b64_image)
            annotated_image = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            st.warning(f"Image decode error: {e}")

    # Navigate to the predictions list – note the nested structure
    predictions_data = result.get("predictions", {})
    predictions_list = predictions_data.get("predictions", [])
    
    # If the top-level already has a "predictions" list directly, use that
    if not predictions_list and "predictions" in result:
        predictions_list = result["predictions"]
        if isinstance(predictions_list, dict):
            predictions_list = predictions_list.get("predictions", [])

    for det in predictions_list:
        class_name = det.get("class", "")
        confidence = det.get("confidence", 0)
        # Filter low confidence detections
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

# Placeholders for results that need to be displayed later
debug_result = None
show_debug = False

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Uploaded Image", width=400)

    if mode == "Single Coin":
        with st.spinner("Classifying single coin..."):
            final_label, conf, raw = predict_single_coin(original_image)
        with col2:
            st.success("### Classification Result")
            st.metric("Predicted Coin", final_label)
            st.metric("Confidence", f"{conf:.2%}")
            st.caption(f"Raw prediction: {raw}")

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

        # Store the raw result for debug display at the bottom
        debug_result = result
        show_debug = True

else:
    st.info("Please upload an image to start.")

# -------------------------------
# 5. Debug output at the bottom (only in multi-coin mode)
# -------------------------------
if show_debug and debug_result is not None:
    with st.expander("🔍 Debug: Raw API Response (image omitted)"):
        # Create a copy without the huge image fields
        result_copy = debug_result
        if isinstance(result_copy, list) and len(result_copy) > 0:
            result_copy = result_copy[0]
        if isinstance(result_copy, dict):
            result_copy = result_copy.copy()
            for field in ["output_image", "visualization", "image", "base64"]:
                result_copy.pop(field, None)
        st.json(result_copy)

# -------------------------------
# 6. Sidebar with architecture explanation
# -------------------------------
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
    """)
