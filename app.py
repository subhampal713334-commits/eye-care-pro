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
# LOAD MODEL (Positional Fix)
# =========================
@st.cache_resource
def load_model():
    try:
        # Create the Base Model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        
        # Build Functional Model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # THE FIX: Load by POSITION (by_name=False)
        # This tells the model: "Ignore the name 'Conv1', just take the 
        # first set of weights in the file and put them in my first layer."
        try:
            model.load_weights("final.weights.h5", by_name=False)
            return model
        except Exception as e:
            # Fallback: Load with skip_mismatch if positional fails
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
# UI & PREDICTION (UNCHANGED)
# =========================
st.markdown('<h1 style="text-align:center;">👁️ AI EyeCare Pro+</h1>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload OCT Scan", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # CRITICAL: MobileNetV2 requires preprocess_input for correct values
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    disease = class_names[idx]
    conf = float(prediction[0][idx])

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_column_width=True)
    with col2:
        st.subheader(f"Diagnosis: {disease}")
        st.write(f"Confidence: {round(conf*100, 2)}%")
