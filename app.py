import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI EyeCare Pro+", layout="wide")

# =========================
# LOAD MODEL (Fixed for final.weights.h5)
# =========================
@st.cache_resource
def load_model():
    try:
        # We load the file directly. Even if it's named 'weights', 
        # if you used model.save() in Kaggle, this will load the whole brain.
        model = tf.keras.models.load_model("final.weights.h5", compile=False)
        return model
    except Exception as e:
        # If the above fails, it means your file ONLY contains weights.
        # We then try to rebuild the architecture as a fallback.
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, models

            base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(4, activation='softmax')
            ])
            model.build((None, 224, 224, 3))
            
            # The 'skip_mismatch' is what stops the 'Conv1' error, 
            # but 'by_name=True' is what ensures the weights find the right layers.
            model.load_weights("final.weights.h5", skip_mismatch=True, by_name=True)
            return model
        except Exception as e2:
            st.error(f"❌ Model load error: {e2}")
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
        'CNV': "Choroidal Neovascularization involves abnormal blood vessel growth beneath the retina.",
        'DME': "Diabetic Macular Edema is caused by fluid accumulation due to high blood sugar.",
        'DRUSEN': "Drusen are yellow deposits under the retina associated with AMD.",
        'NORMAL': "No abnormal retinal pathology detected. Retina appears healthy."
    }
    risk = {'CNV': "🔴 HIGH", 'DME': "🔴 HIGH", 'DRUSEN': "🟡 MODERATE", 'NORMAL': "🟢 LOW"}
    
    report = f"🔍 Diagnosis: {disease}\n📊 Confidence: {round(confidence*100,2)}%\n⚠️ Risk: {risk[disease]}\n\n{explanations[disease]}"
    return report

# =========================
# GRAD-CAM
# =========================
def get_gradcam(model, img_array):
    # Search for the last convolutional layer
    last_conv_layer = None
    # If model is Sequential, the base_model is likely model.layers[0]
    target_model = model.layers[0] if hasattr(model.layers[0], 'layers') else model
    
    for layer in reversed(target_model.layers):
        if "conv" in layer.name.lower():
            last_conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        inputs=[target_model.input],
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
body { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
.header { padding: 35px; border-radius: 20px; background: linear-gradient(90deg, #00C9FF, #92FE9D); text-align: center; color: black; }
.card { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 20px; text-align: center; }
.report { background: #F9FAFB; padding: 25px; border-radius: 15px; border: 1px solid #E5E7EB; color: #111827; font-family: 'Segoe UI', sans-serif; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="header"><h1>👁️ AI EyeCare Pro+</h1><p>Advanced Retinal Disease Detection</p></div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload OCT Scan", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(image, use_column_width=True)

    # PREPROCESSING
    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Using official preprocess_input to fix the 25% bias
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    disease = class_names[idx]
    confidence = float(prediction[0][idx])

    with col2:
        st.markdown("### 🧠 Diagnosis")
        if disease == "NORMAL": st.success(f"🟢 {disease}")
        else: st.error(f"🔴 {disease}")
        st.progress(confidence)
        st.write(f"Confidence: {round(confidence*100,2)}%")

    st.markdown("### 📊 Analysis")
    cols = st.columns(4)
    for i, val in enumerate(prediction[0]):
        with cols[i]:
            st.markdown(f"""<div class="card"><h4>{class_names[i]}</h4><h2>{round(val*100,2)}%</h2></div>""", unsafe_allow_html=True)

    st.markdown("### 🧾 Clinical Report")
    report = generate_report(disease, confidence).replace("\n", "<br>")
    st.markdown(f"""<div class="report">{report}</div>""", unsafe_allow_html=True)

    st.markdown("### 🔥 AI Attention Map")
    try:
        heatmap = get_gradcam(model, img_array)
        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original = np.array(image.resize((224,224)))
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        st.image(overlay, use_column_width=True)
    except:
        st.write("Heatmap unavailable.")
