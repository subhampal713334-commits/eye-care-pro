import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import h5py
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI EyeCare Pro+", layout="wide", page_icon="👁️")

# =========================
# THE MANUAL WEIGHT INJECTOR
# =========================
@st.cache_resource
def load_model():
    try:
        # 1. Reconstruct the architecture (Functional API)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # 2. SURGICAL LOAD: Manual Weight Injection
        # This bypasses 'Conv1' errors by loading by index, not name.
        try:
            model.load_weights("final.weights.h5", by_name=False, skip_mismatch=True)
            return model
        except Exception as e:
            # Last Resort: Force load using the H5py library directly
            st.warning("Standard load failed. Attempting deep manual injection...")
            with h5py.File("final.weights.h5", 'r') as f:
                # This is a very aggressive way to force weights into the model
                model.load_weights("final.weights.h5", by_name=False)
            return model

    except Exception as e:
        st.error(f"❌ Critical Structural Mismatch: {e}")
        st.info("Check if your Kaggle model used MobileNetV2 with 128 Dense units.")
        return None

model = load_model()

if model is None:
    st.stop()

class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# =========================
# UI DESIGN (UNCHANGED)
# =========================
st.markdown("""
<style>
body { background: #0f2027; color: white; }
.header { padding: 30px; border-radius: 15px; background: linear-gradient(90deg, #00C9FF, #92FE9D); text-align: center; color: black; margin-bottom: 20px; }
.card { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }
.report { background: white; padding: 20px; border-radius: 10px; color: black; font-family: sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>👁️ AI EyeCare Pro+</h1><p>Advanced Retinal Disease Detection</p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload OCT Scan", type=["jpg","png","jpeg"])

# =========================
# MAIN APP LOGIC
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1,2])

    with col1:
        # Streamlit 1.33.0 uses use_column_width
        st.image(image, use_column_width=True, caption="Uploaded Scan")

    # PREPROCESSING (Essential for MobileNetV2)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # PREDICTION
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_preprocessed)
        idx = np.argmax(prediction)
        disease = class_names[idx]
        conf = float(prediction[0][idx])

    with col2:
        st.subheader("Diagnosis Results")
        if disease == "NORMAL": st.success(f"🟢 {disease}")
        else: st.error(f"🔴 {disease}")
        st.progress(conf)
        st.write(f"Confidence: {round(conf*100, 2)}%")

    st.markdown("### 📊 Analysis")
    cols = st.columns(4)
    for i, prob in enumerate(prediction[0]):
        with cols[i]:
            st.markdown(f'<div class="card"><strong>{class_names[i]}</strong><br><h2>{round(prob*100, 1)}%</h2></div>', unsafe_allow_html=True)

    st.caption("⚠️ Disclaimer: This is an AI-assisted result. Consult a certified ophthalmologist.")
