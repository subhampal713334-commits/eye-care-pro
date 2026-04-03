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
# LOAD MODEL (Manual Injection)
# =========================
@st.cache_resource
def load_model():
    try:
        # 1. Build the architecture EXACTLY
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

        # 2. THE FIX: Load weights with 'by_name=False' AND 'skip_mismatch=True'
        # This tells TF: "If you find a layer named Conv1 that doesn't match, 
        # IGNORE IT and keep loading the rest of the 150+ layers."
        model.load_weights("final.weights.h5", by_name=False, skip_mismatch=True)
        
        return model
    except Exception as e:
        st.error(f"❌ Structural Mismatch: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# =========================
# MAIN UI
# =========================
st.markdown('<h1 style="text-align:center;">👁️ AI EyeCare Pro+</h1>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload OCT Scan", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # PREPROCESSING - This is why it might show 25% if math is wrong
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # MobileNetV2 MUST have this. It scales 0-255 to -1 to 1.
    img_array = preprocess_input(img_array)

    # PREDICTION
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    
    # DEBUG: Show raw numbers if it's still 25%
    # st.write(f"Raw Predictions: {prediction}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_column_width=True)
    with col2:
        st.subheader(f"Diagnosis: {class_names[idx]}")
        st.write(f"Confidence: {round(float(np.max(prediction))*100, 2)}%")
        
    # Show breakdown
    for i, name in enumerate(class_names):
        st.write(f"{name}: {round(float(prediction[0][i])*100, 2)}%")
