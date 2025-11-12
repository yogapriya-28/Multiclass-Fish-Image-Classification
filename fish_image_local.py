import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="🐟 Fish Classification Dashboard",
    page_icon="🐠",
    layout="wide"
)

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Home", "📤 Classify Fish","ℹ️ About"],
        icons=["house", "upload", "bar-chart-line", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f4f4ff"},
            "icon": {"color": "#4B0082", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "margin": "2px",
                "border-radius": "8px",
                "color": "#333",
                "text-align": "left",
                "padding": "8px 10px"
            },
            "nav-link-selected": {"background-color": "#7B68EE", "color": "white"},
        },
    )

# ------------------------------------------------------------
# Custom CSS for Clean Design
# ------------------------------------------------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #FAF9FF;
        }
        h1, h2, h3, h4 {
            color: #4B0082;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
if selected == "🏠 Home":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h1>🐠 Multiclass Fish Image Classification</h1>", unsafe_allow_html=True)
        st.write("""
        This deep learning project classifies **different species of fish** using **MobileNet**, 
        a lightweight and high-accuracy pre-trained model optimized for real-time predictions.
        """)

        st.markdown("### 🌊 Features")
        st.markdown("""
        - ✅ Classifies multiple fish species accurately  
        - ⚡ Fast predictions using MobileNet  
        - 📷 Easy image upload interface  
        - 📊 Displays confidence level per class  
        """)

        st.markdown("### 🧠 Tech Stack")
        st.markdown("""
        - **Frameworks:** TensorFlow, Keras, Streamlit  
        - **Architecture:** MobileNet (Transfer Learning)  
        - **Visualization:** Matplotlib, Seaborn  
        """)

        st.markdown("### 🚀 Workflow")
        st.markdown("""
        1. Upload a fish image  
        2. Model processes and predicts the species  
        3. Displays prediction and confidence chart  
        """)

    with col2:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2974/2974293.png",
            caption="AI-powered Fish Classification",
            width=350
        )

# ------------------------------------------------------------
# CLASSIFY PAGE
# ------------------------------------------------------------
elif selected == "📤 Classify Fish":
    st.markdown("<h1>📸 Upload a Fish Image for Classification</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Fish Image", use_container_width=True)

        # Preprocess the image
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Load trained model
        model_path = r"F:\multiclass_fish\mobilenet_fish_final.keras"
        model = tf.keras.models.load_model(model_path)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Fish species labels
        class_labels = [
            "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
            "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
            "fish sea_food red_mullet", "fish sea_food red_sea_bream",
            "fish sea_food sea_bass", "fish sea_food shrimp",
            "fish sea_food striped_red_mullet", "fish sea_food trout"
        ]

        predicted_label = class_labels[predicted_class]
        confidence = np.max(prediction) * 100

        # Display results
        st.success(f"🎯 Predicted Fish Species: **{predicted_label}**")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        # Confidence chart
        st.markdown("### 📊 Confidence Levels for All Classes")

        conf_df = pd.DataFrame({
            "Fish Species": class_labels,
            "Confidence (%)": prediction[0] * 100
        }).sort_values(by="Confidence (%)", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=conf_df,
            x="Confidence (%)",
            y="Fish Species",
            hue="Fish Species",
            palette="Blues_r",
            dodge=False,
            legend=False
        )

        # Add confidence values on bars
        for index, value in enumerate(conf_df["Confidence (%)"]):
            ax.text(value + 0.5, index, f"{value:.1f}%", va='center', fontsize=9, color="#333")

        plt.title("Confidence per Fish Species", fontsize=13, color="#4B0082")
        plt.xlabel("Confidence (%)")
        plt.ylabel("")
        plt.xlim(0, 105)
        plt.tight_layout()
        st.pyplot(fig)



# ------------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------------
elif selected == "ℹ️ About":
    st.markdown("<h1>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Objective
    To automatically identify fish species using AI and deep learning for applications in fisheries, marine biology, and food technology.

    ### 💡 Key Highlights
    - Based on **MobileNet Transfer Learning**
    - Achieved **97.8% accuracy** on validation data
    - Deployed as a responsive Streamlit web dashboard

    ### 🔮 Future Enhancements
    - Integrate **real-time webcam prediction**
    - Add **Vision Transformer (ViT)** models
    - Deploy on **AWS / GCP / Azure**
    - Build **Mobile App Interface**

    ### 🌊 Real-world Applications
    - Fisheries and aquaculture management  
    - Marine species research  
    - Seafood industry classification  
    - Educational AI learning tool  
    """)

   