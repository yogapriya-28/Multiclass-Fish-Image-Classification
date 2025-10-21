# ----------------------------------------------
# Multiclass Fish Image Classification Dashboard 
# ----------------------------------------------

import os
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

# -------------------------------
# Load model
# -------------------------------
MODEL_PATH = r'F:\fish image\best_fish_model.keras'

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

model = load_model(MODEL_PATH)

# -------------------------------
# Fish categories
# -------------------------------
class_names = [
    'animal_fish', 'animal_fish_bass', 'fish_sea_food_black_sea_sprat',
    'fish_sea_food_gilt_head_bream', 'fish_sea_food_hourse_mackerel',
    'fish_sea_food_red_mullet', 'fish_sea_food_red_sea_bream',
    'fish_sea_food_sea_bass', 'fish_sea_food_shrimp',
    'fish_sea_food_striped_red_mullet', 'fish_sea_food_trout'
]

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title=" Fish Image Classifier Dashboard",
    layout="wide",
    page_icon="🐟"
)

# -------------------------------
# App header
# -------------------------------
st.markdown(
    """
    <h1 style='text-align:center'>🐠 Multiclass Fish Image Classifier Dashboard</h1>
    <p style='text-align:center'>Upload a fish image and explore predictions interactively 🌈</p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# -------------------------------
# Prediction function (tf.function for speed)
# -------------------------------
@tf.function
def predict_fn(img_array):
    return model(img_array, training=False)

def predict(image):
    img_array = preprocess_image(image)
    predictions = predict_fn(img_array)
    scores = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(scores)]
    confidence = 100 * np.max(scores)
    return predicted_class, confidence, scores

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📤 Choose a fish image...", type=["jpg","jpeg","png"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file)
    
    # Prediction
    predicted_class, confidence, scores = predict(image)
    
    # DataFrame sorted high → low
    df = pd.DataFrame({
        'Fish Category': class_names,
        'Confidence (%)': [float(s)*100 for s in scores]
    }).sort_values('Confidence (%)', ascending=False).reset_index(drop=True)

    # -------------------------------
    # Display image + main prediction
    # -------------------------------
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(image, caption='📷 Uploaded Image', use_container_width=True)
    with col2:
        st.success(f"✅ Predicted Category: **{predicted_class}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
    
    st.markdown("---")

    # -------------------------------
    # Interactive confidence chart
    # -------------------------------
    fig = px.bar(
        df.sort_values('Confidence (%)', ascending=True),
        x='Confidence (%)',
        y='Fish Category',
        orientation='h',
        color='Confidence (%)',
        color_continuous_scale='Rainbow',
        text_auto='.2f',
        title='🌈 Model Confidence per Class'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Confidence (%)",
        yaxis_title="Fish Category",
        title_font_size=20,
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
    
    st.markdown("---")

 # -------------------------------
    # Professional color-coded confidence table with gradient
    # -------------------------------
    st.markdown("### 📋 Confidence Table ")

    styled_df = df.style.format({'Confidence (%)': '{:.2f}%'}).background_gradient(
        subset=['Confidence (%)'],
        cmap='RdYlGn',  # Red → Yellow → Green
        axis=0
    ).set_properties(**{'text-align': 'center', 'font-weight': 'bold'})

    st.dataframe(styled_df, use_container_width=True)

else:
    st.info("👆 Upload a fish image to begin classification.")