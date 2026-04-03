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
# LOAD MODEL (Revised for Full Model Load)
# =========================
@st.cache_resource
def load_model():
    try:
        # Load the complete model object directly to avoid layer naming mismatches
        # Ensure you have uploaded 'full_eye_model.h5' (saved via model.save() in Kaggle)
        model = tf.keras.models.load_model("full_eye_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        st.info("Tip: Make sure you uploaded the 'full' model file, not just the weights.")
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
        'CNV': "Choroidal Neovascularization involves abnormal blood vessel growth beneath the retina, which can leak fluid or blood and distort vision.",
        'DME': "Diabetic Macular Edema is caused by fluid accumulation in the retina due to prolonged high blood sugar levels.",
        'DRUSEN': "Drusen are yellow deposits under the retina, commonly associated with age-related macular degeneration.",
        'NORMAL': "No abnormal retinal pathology detected. Retina appears healthy."
    }

    risk = {
        'CNV': "🔴 HIGH RISK (Vision-threatening condition)",
        'DME': "🔴 HIGH RISK (Requires medical management)",
        'DRUSEN': "🟡 MODERATE RISK (Monitor progression)",
        'NORMAL': "🟢 LOW RISK"
    }

    treatment = {
        'CNV': ["Immediate consultation with an ophthalmologist", "Anti-VEGF injections", "OCT monitoring", "Possible laser therapy"],
        'DME': ["Strict blood sugar control", "Anti-VEGF therapy", "Regular retinal examinations", "Healthy diet"],
        'DRUSEN': ["Routine eye monitoring every 6–12 months", "AREDS supplements", "UV-protection", "Diet rich in antioxidants"],
        'NORMAL': ["Maintain regular eye checkups", "Balanced diet", "Limit screen time", "Protect eyes from UV"]
    }

    followup = {
        'CNV': "Urgent follow-up within 1–2 weeks recommended.",
        'DME': "Follow-up within 2–4 weeks with retinal specialist.",
        'DRUSEN': "Routine follow-up every 6 months.",
        'NORMAL': "Annual eye checkup recommended."
    }

    report = f"""
🧾 AI CLINICAL REPORT

🔍 Diagnosis: {disease}
📊 Confidence: {round(confidence*100,2)}%

🧠 Clinical Explanation:
{explanations[disease]}

⚠️ Risk Assessment:
{risk[disease]}

💊 Recommended Treatment Plan:
"""
    for t in treatment[disease]:
        report += f"\n• {t}"

    report += f"\n\n📅 Follow-up:\n{followup[disease]}\n\n⚠️ Disclaimer:\nThis is an AI-assisted report. Please consult a certified ophthalmologist."
    return report

# =========================
# GRAD-CAM (Visual Explanation)
# =========================
def get_gradcam(model, img_array):
    # Search for the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            last_conv_layer = layer
            break
            
    if not last_conv_layer:
        # Fallback for nested models (like MobileNetV2 as a base layer)
        base_model = model.layers[0]
        for layer in reversed(base_model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer = layer
                break

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
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
# UI DESIGN (Kept exactly as requested)
# =========================
st.markdown("""
<style>
body { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
.header { padding: 35px; border-radius: 20px; background: linear-gradient(90deg, #00C9FF, #92FE9D); text-align: center; color: black; }
.card { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 20px; text-align: center; }
.report { background: #F9FAFB; padding: 25px; border-radius: 15px; border: 1px solid #E5E7EB; color: #111827; font-family: 'Segoe UI', sans-serif; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="header"><h1>👁️ AI EyeCare Pro+</h1><p>Advanced Retinal Disease Detection & Clinical Decision Support</p></div>""", unsafe_allow_html=True)

st.markdown("## 📤 Upload OCT Scan")
uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

# =========================
# MAIN LOGIC
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1,2])

    with col1:
        # Fixed: Changed use_container_width to use_column_width for version 1.33.0 compatibility
        st.image(image, use_column_width=True)

    # Preprocessing
    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Fixed: Using MobileNetV2 official preprocessing to fix accuracy shift
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)
    disease = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    with col2:
        st.markdown("### 🧠 Diagnosis")
        if disease == "NORMAL":
            st.success(f"🟢 {disease}")
        else:
            st.error(f"🔴 {disease}")
        st.progress(confidence)
        st.write(f"Confidence: {round(confidence*100,2)}%")

    st.markdown("### 📊 Analysis")
    cols = st.columns(4)
    for i, val in enumerate(prediction[0]):
        with cols[i]:
            st.markdown(f"""<div class="card"><h4>{class_names[i]}</h4><h2>{round(val*100,2)}%</h2></div>""", unsafe_allow_html=True)

    st.markdown("### 🧾 Clinical Report")
    report = generate_report(disease, confidence)
    formatted_report = report.replace("\n", "<br>")
    st.markdown(f"""<div class="report">{formatted_report}</div>""", unsafe_allow_html=True)

    st.markdown("### 🔥 AI Attention Map")
    try:
        heatmap = get_gradcam(model, img_array)
        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Standardizing image for overlay
        original = np.array(image.resize((224,224)))
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        st.image(overlay, use_column_width=True)
    except Exception as e:
        st.write("Heatmap could not be generated for this image.")

    st.caption("⚠️ AI-assisted result. Always consult a certified ophthalmologist.")
