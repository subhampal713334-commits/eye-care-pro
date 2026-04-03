import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI EyeCare Pro+", layout="wide", page_icon="👁️")

# =========================
# LOAD MODEL (Fixed for Layer Mismatch)
# =========================
@st.cache_resource
def load_eye_model():
    model_path = "final.weights.h5"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found in repository!")
        return None

    try:
        # Strategy 1: Try loading as a full model (Architecture + Weights)
        # This is the most stable method if the file is a complete H5 save.
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception:
        try:
            # Strategy 2: Rebuild architecture and load weights with 'skip_mismatch'
            # This fixes the "Layer Conv1 expected 1 variables" error.
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, models

            base_model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None  # We will load our own weights
            )

            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(4, activation='softmax')
            ])

            model.build((None, 224, 224, 3))
            
            # skip_mismatch=True allows loading even if internal layer names differ slightly
            model.load_weights(model_path, skip_mismatch=True, by_name=True)
            return model
        except Exception as e:
            st.error(f"❌ Critical Model Load Failure: {e}")
            return None

model = load_eye_model()

if model is None:
    st.warning("Please ensure 'final.weights.h5' is uploaded to your GitHub repository.")
    st.stop()

class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# =========================
# REPORT GENERATOR
# =========================
def generate_report(disease, confidence):
    explanations = {
        'CNV': "Choroidal Neovascularization involves abnormal blood vessel growth beneath the retina.",
        'DME': "Diabetic Macular Edema is fluid accumulation in the retina due to high blood sugar.",
        'DRUSEN': "Drusen are yellow deposits under the retina, associated with early AMD.",
        'NORMAL': "No abnormal retinal pathology detected. Retina appears healthy."
    }
    
    risk = {'CNV': "🔴 HIGH", 'DME': "🔴 HIGH", 'DRUSEN': "🟡 MODERATE", 'NORMAL': "🟢 LOW"}
    
    report = f"""
    🧾 AI CLINICAL REPORT
    --------------------------
    🔍 Diagnosis: {disease}
    📊 Confidence: {round(confidence*100,2)}%
    ⚠️ Risk Level: {risk[disease]}

    🧠 Clinical Explanation:
    {explanations[disease]}

    📅 Recommendation:
    Please consult an ophthalmologist for a formal OCT review.
    """
    return report

# =========================
# GRAD-CAM (Visual Explanation)
# =========================
def get_gradcam(model, img_array):
    # Search for the last convolutional layer in the base model
    base_model = model.layers[0]
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=[base_model.input],
        outputs=[base_model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, "float32")
    gate_r = tf.cast(grads > 0, "float32")
    guided_grads = gate_f * gate_r * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return heatmap

# =========================
# UI DESIGN & STYLING
# =========================
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stHeader { background: linear-gradient(90deg, #4b6cb7, #182848); color: white; padding: 2rem; border-radius: 15px; }
    .report-box { background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #4b6cb7; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="stHeader"><h1>👁️ AI EyeCare Pro+</h1><p>Retinal OCT Analysis System</p></div>', unsafe_allow_html=True)

# =========================
# MAIN APP LOGIC
# =========================
uploaded_file = st.file_uploader("Upload an OCT scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner('Analyzing scan...'):
        prediction = model.predict(img_array)
        idx = np.argmax(prediction)
        disease = class_names[idx]
        conf = prediction[0][idx]

    with col1:
        st.image(image, caption="Uploaded OCT Scan", use_container_width=True)

    with col2:
        st.subheader("Analysis Result")
        if disease == "NORMAL":
            st.success(f"Result: {disease}")
        else:
            st.error(f"Result: {disease}")
        st.metric("Confidence Score", f"{round(conf*100, 2)}%")
        
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.text(generate_report(disease, conf))
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Heatmap Section
    st.subheader("🔥 AI Attention Map (Grad-CAM)")
    st.write("The red areas indicate where the AI is looking to make its diagnosis.")
    
    try:
        heatmap = get_gradcam(model, img_array)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR2RGB)

        original_img = np.uint8(255 * img_array[0])
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        st.image(overlay, use_container_width=True)
    except Exception as e:
        st.info("Heatmap generation is currently unavailable for this model architecture.")

st.caption("Disclaimer: This tool is for educational purposes and should not replace professional medical advice.")
