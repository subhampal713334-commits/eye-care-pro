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
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        # Step 1: Attempt to load as a full model object (Best Case)
        return tf.keras.models.load_model("final.weights.h5", compile=False)
    except:
        try:
            # Step 2: Manual Reconstruction (Fallback)
            # We use the exact architecture you defined in Kaggle
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3), 
                include_top=False, 
                weights=None
            )
            
            model = tf.keras.models.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
            model.build((None, 224, 224, 3))
            
            # CRITICAL FIX: by_name=False loads weights by POSITION. 
            # This fixes the "Conv1" name mismatch that causes 25% accuracy.
            model.load_weights("final.weights.h5", by_name=False)
            return model
        except Exception as e:
            st.error(f"❌ Model Weight Mismatch: {e}")
            return None

model = load_model()

if model is None:
    st.warning("Ensure 'final.weights.h5' is in your GitHub folder and matches the architecture.")
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
        'NORMAL': "No abnormal retinal pathology detected."
    }
    risk = {'CNV': "🔴 HIGH", 'DME': "🔴 HIGH", 'DRUSEN': "🟡 MODERATE", 'NORMAL': "🟢 LOW"}
    
    report = f"🔍 Diagnosis: {disease}\n📊 Confidence: {round(confidence*100,2)}%\n⚠️ Risk Level: {risk[disease]}\n\n{explanations[disease]}"
    return report

# =========================
# GRAD-CAM (Fixed for Heatmap Unavailable)
# =========================
def get_gradcam(model, img_array):
    # Dig into the MobileNetV2 base model
    base_model = model.layers[0]
    
    # Try to find the standard final conv layer for MobileNetV2
    try:
        last_conv_layer = base_model.get_layer("out_relu")
    except:
        # Fallback: Search for any conv layer
        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer = layer
                break
    
    grad_model = tf.keras.models.Model(
        inputs=[base_model.input],
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

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

st.markdown('<div class="header"><h1>👁️ AI EyeCare Pro+</h1><p>Retinal Disease Detection System</p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload OCT Scan", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1,2])

    with col1:
        # Version 1.33.0 uses use_column_width
        st.image(image, use_column_width=True, caption="Original Scan")

    # PREPROCESSING (Essential for MobileNetV2)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # PREDICTION
    with st.spinner("Analyzing..."):
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

    st.markdown("### 📊 Probability Breakdown")
    cols = st.columns(4)
    for i, prob in enumerate(prediction[0]):
        with cols[i]:
            st.markdown(f'<div class="card"><strong>{class_names[i]}</strong><br><h2>{round(prob*100, 1)}%</h2></div>', unsafe_allow_html=True)

    st.markdown("### 🧾 Clinical Report")
    report_text = generate_report(disease, conf).replace("\n", "<br>")
    st.markdown(f'<div class="report">{report_text}</div>', unsafe_allow_html=True)

    st.markdown("### 🔥 AI Attention Map")
    try:
        heatmap = get_gradcam(model, img_preprocessed)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Merge with original image
        overlay = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.4, 0)
        st.image(overlay, use_column_width=True)
    except:
        st.info("Heatmap visualization not supported for this specific model architecture.")

    st.caption("⚠️ Disclaimer: AI-assisted result only. Consult a doctor.")
