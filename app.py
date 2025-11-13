import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
import os
import time

# ==================================================
# 1️⃣ Load Model and Labels
# ==================================================
MODEL_PATH = "sign_model.keras"
LABEL_MAP_PATH = "label_map.pkl"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_label_map():
    if os.path.exists(LABEL_MAP_PATH):
        return joblib.load(LABEL_MAP_PATH)
    else:
        st.warning("⚠️ label_map.pkl not found. Using default A–Z labels.")
        return {i: chr(65 + i) for i in range(26)}

model = load_model()
label_map = load_label_map()

# ==================================================
# 2️⃣ Streamlit UI Layout
# ==================================================
st.set_page_config(page_title="ASL Sign Detector", page_icon="🖐️", layout="wide")
st.title("🖐️ Real-Time Sign Language Detection")

st.markdown("""
Place your hand **inside the green box** and wait for the system to predict your sign.  
The prediction will appear on the right side in real-time.
""")

col1, col2 = st.columns(2)
start_button = col1.button("▶ Start Detection")
stop_button = col1.button("⏹ Stop Detection")

col2.markdown("### 🔤 **Live Prediction Output**")
prediction_placeholder = col2.empty()

# ==================================================
# 3️⃣ Helper Functions
# ==================================================
def preprocess_roi(roi):
    """Preprocess the Region of Interest for model prediction."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype("float32") / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

def draw_box(frame):
    """Draw green box in the center of the frame."""
    h, w, _ = frame.shape
    box_size = 250
    start_x = w // 2 - box_size // 2
    start_y = h // 2 - box_size // 2
    end_x = start_x + box_size
    end_y = start_y + box_size
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    return start_x, start_y, end_x, end_y

# ==================================================
# 4️⃣ Real-Time Detection
# ==================================================
if start_button:
    stframe = col1.image([], channels="RGB")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("❌ Could not access webcam.")
    else:
        st.info("✅ Camera started. Press **Stop Detection** to exit.")
        prev_time = 0

        while camera.isOpened() and not stop_button:
            ret, frame = camera.read()
            if not ret:
                st.warning("⚠️ Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            x1, y1, x2, y2 = draw_box(frame)
            roi = frame[y1:y2, x1:x2]

            # Predict every 0.5 seconds for stability
            curr_time = time.time()
            if curr_time - prev_time >= 0.5:
                prev_time = curr_time
                processed = preprocess_roi(roi)
                preds = model.predict(processed)
                pred_index = np.argmax(preds)
                confidence = np.max(preds) * 100
                pred_label = label_map.get(pred_index, "?")

                # Overlay on frame
                cv2.putText(frame, f"{pred_label} ({confidence:.1f}%)",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                prediction_placeholder.markdown(
                    f"### 🟢 Predicted Sign: **{pred_label}**\nConfidence: **{confidence:.2f}%**"
                )

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        camera.release()
        cv2.destroyAllWindows()
        st.success("🛑 Detection stopped.")

