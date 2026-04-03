import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI EyeCare Pro+", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers, models

        base_model = MobileNetV2(
            input_shape=(224,224,3),
            include_top=False,
            weights=None
        )

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')
        ])

        model.build((None, 224, 224, 3))
        model.load_weights("final.weights.h5")

        return model

    except Exception as e:
        st.error(f"❌ Model load error: {e}")
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
        'CNV': [
            "Immediate consultation with an ophthalmologist",
            "Anti-VEGF injections (e.g., Ranibizumab, Aflibercept)",
            "OCT monitoring for disease progression",
            "Possible laser or photodynamic therapy",
            "Avoid smoking and maintain eye health"
        ],
        'DME': [
            "Strict blood sugar control (HbA1c monitoring)",
            "Anti-VEGF therapy or corticosteroid injections",
            "Regular retinal examinations",
            "Healthy diet and daily exercise",
            "Blood pressure and cholesterol management"
        ],
        'DRUSEN': [
            "Routine eye monitoring every 6–12 months",
            "AREDS supplements (if recommended)",
            "UV-protection sunglasses",
            "Diet rich in antioxidants (leafy greens, fish)",
            "Avoid smoking"
        ],
        'NORMAL': [
            "Maintain regular eye checkups",
            "Balanced diet (Vitamin A, Omega-3)",
            "Limit screen time (20-20-20 rule)",
            "Protect eyes from UV exposure"
        ]
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

    report += f"""

📅 Follow-up:
{followup[disease]}

⚠️ Disclaimer:
This is an AI-assisted report. Please consult a certified ophthalmologist.
"""

    return report

# =========================
# GRAD-CAM
# =========================
def get_gradcam(model, img_array):

    base_model = model.layers[0]

    last_conv = None
    for layer in base_model.layers[::-1]:
        if "conv" in layer.name:
            last_conv = layer
            break

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_array)

        x = base_output
        for layer in model.layers[1:]:
            x = layer(x)

        preds = x
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        grads = tf.ones_like(conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# =========================
# UI DESIGN
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.header {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    text-align: center;
    color: black;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
}
.report {
    background: #F9FAFB;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #E5E7EB;
    color: #111827;
    font-family: 'Segoe UI', sans-serif;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header">
    <h1>👁️ AI EyeCare Pro+</h1>
    <p>Advanced Retinal Disease Detection & Clinical Decision Support</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## 📤 Upload OCT Scan")
uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

# =========================
# MAIN
# =========================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(image, use_container_width=True)

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

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

    # =========================
    # PROBABILITY
    # =========================
    st.markdown("### 📊 Analysis")

    cols = st.columns(4)
    for i, val in enumerate(prediction[0]):
        with cols[i]:
            st.markdown(f"""
            <div class="card">
                <h4>{class_names[i]}</h4>
                <h2>{round(val*100,2)}%</h2>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # REPORT
    # =========================
    st.markdown("### 🧾 Clinical Report")

    report = generate_report(disease, confidence)
    formatted_report = report.replace("\n", "<br>")

    st.markdown(f"""
    <div class="report">
    {formatted_report}
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # HEATMAP
    # =========================
    st.markdown("### 🔥 AI Attention Map")

    heatmap = get_gradcam(model, img_array)
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.uint8(img_array[0] * 255)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    st.image(overlay, use_container_width=True)

    st.caption("⚠️ AI-assisted result. Always consult a certified ophthalmologist.")
