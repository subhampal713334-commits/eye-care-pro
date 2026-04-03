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
# THE "SURGICAL" LOAD FIX
# =========================
@st.cache_resource
def load_model():
    try:
        # Step 1: Manual Architecture Build
        # MobileNetV2 ko standard tarike se define karna
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

        # Step 2: Positional Weight Loading
        # 'by_name=False' matlab: Layer ka naam jo bhi ho (Conv1 ya kuch aur), 
        # bas weights ko line se bhar do.
        model.load_weights("final.weights.h5", by_name=False, skip_mismatch=True)
        
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}")
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
.header { padding: 30px; border-radius: 15px; background: linear-gradient(90deg, #00C9FF, #92FE9D); text-align: center; color: black; }
.report-box { background: white; padding: 20px; border-radius: 10px; color: black; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>👁️ AI EyeCare Pro+</h1><p>Retinal OCT Analysis</p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload OCT Scan", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 1. PREPROCESSING (Sabse Important Step)
    # Agar ye galat hua toh model 25% hi dikhayega
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) # MobileNetV2 requires -1 to 1 scaling

    # 2. PREDICTION
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    conf = float(prediction[0][idx])

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_column_width=True)
    with col2:
        st.subheader(f"Diagnosis: {class_names[idx]}")
        st.write(f"Confidence: {round(conf*100, 2)}%")
        
        # Details
        for i, name in enumerate(class_names):
            st.write(f"{name}: {round(float(prediction[0][i])*100, 2)}%")

st.caption("Disclaimer: AI-assisted result. Always consult an ophthalmologist.")
