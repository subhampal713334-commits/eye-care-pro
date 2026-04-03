import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI EyeCare Pro+", layout="wide", page_icon="👁️")

# =========================
# LOAD MODEL (The "Positional" Fix)
# =========================
@st.cache_resource
def load_model():
    try:
        # 1. Define the input shape
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        # 2. Re-create the exact MobileNetV2 base used in Kaggle
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        
        # 3. Build the Functional structure (More stable than Sequential)
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # 4. LOAD WEIGHTS BY POSITION
        # We use by_name=False to force weights into layers by ORDER.
        # This bypasses the 'Conv1' naming mismatch entirely.
        model.load_weights("final.weights.h5", by_name=False)
        
        return model
    except Exception as e:
        st.error(f"❌ Structural Mismatch: {e}")
        st.info("Check if your Kaggle model used 128 Dense units and 4 Classes.")
        return None

model = load_model()

if model is None:
    st.stop()

class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# =========================
# REPORT GENERATOR
# =========================
def generate_report(disease, confidence):
    explanations = {
        'CNV': "Choroidal Neovascularization: abnormal blood vessel growth beneath the retina.",
        'DME': "Diabetic Macular Edema: fluid accumulation due to high blood sugar.",
        'DRUSEN': "Drusen: yellow deposits under the retina associated with AMD.",
        'NORMAL': "No abnormal retinal pathology detected. Retina appears healthy."
    }
    risk = {'CNV': "🔴 HIGH", 'DME': "🔴 HIGH", 'DRUSEN': "🟡 MODERATE", 'NORMAL': "🟢 LOW"}
    
    report = f"🔍 Diagnosis: {disease}\n📊 Confidence: {round(confidence*100,2)}%\n⚠️ Risk: {risk[disease]}\n\n{explanations[disease]}"
    return report

# =========================
# UI DESIGN
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
        st.image(image, use_column_width=True, caption="Uploaded Scan")

    # PREPROCESSING (Crucial for MobileNetV2)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    # This function scales pixels to [-1, 1], which MobileNetV2 requires
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

    st.markdown("### 🧾 Clinical Report")
    report_text = generate_report(disease, conf).replace("\n", "<br>")
    st.markdown(f'<div class="report">{report_text}</div>', unsafe_allow_html=True)

    st.caption("⚠️ Disclaimer: This is an AI-assisted result. Consult a certified ophthalmologist.")
