"""
Streamlit Demo UI for Virtual Try-On
"""
import sys
import os
sys.path.append('..')

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from test_pipeline import VirtualTryOnPipeline


# Page config
st.set_page_config(
    page_title="Virtual Try-On System",
    page_icon="👕",
    layout="wide"
)

# Title
st.title("🎨 Virtual Try-On System")
st.markdown("Upload a person image and a cloth image to see the virtual try-on result!")

# Initialize pipeline (cached)
@st.cache_resource
def load_pipeline():
    checkpoint_dir = "../checkpoints"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with st.spinner("Loading AI models... This may take a moment."):
        pipeline = VirtualTryOnPipeline(checkpoint_dir, device)
    
    return pipeline


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    device_info = "🟢 GPU" if torch.cuda.is_available() else "🟡 CPU"
    st.info(f"Device: {device_info}")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Upload a **person image** (front view)
    2. Upload a **cloth image** (flat)
    3. Click **Generate Try-On**
    4. Wait for AI to process
    5. View and download result!
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use high-quality images
    - Person should face camera
    - Cloth should be flat/visible
    - Best size: 256x192 or similar
    """)


# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Person Image")
    person_file = st.file_uploader("Upload Person", type=['jpg', 'jpeg', 'png'], key='person')
    
    if person_file:
        person_img = Image.open(person_file)
        st.image(person_img, caption="Original Person", use_column_width=True)

with col2:
    st.subheader("👕 Cloth Image")
    cloth_file = st.file_uploader("Upload Cloth", type=['jpg', 'jpeg', 'png'], key='cloth')
    
    if cloth_file:
        cloth_img = Image.open(cloth_file)
        st.image(cloth_img, caption="Cloth", use_column_width=True)

with col3:
    st.subheader("✨ Try-On Result")
    result_placeholder = st.empty()


# Generate button
if person_file and cloth_file:
    if st.button("🎨 Generate Try-On", type="primary", use_container_width=True):
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_person:
            tmp_person.write(person_file.getbuffer())
            person_path = tmp_person.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_cloth:
            tmp_cloth.write(cloth_file.getbuffer())
            cloth_path = tmp_cloth.name
        
        try:
            # Load pipeline
            pipeline = load_pipeline()
            
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run pipeline with progress updates
            status_text.text("🎭 Analyzing person...")
            progress_bar.progress(20)
            
            status_text.text("🦴 Detecting pose...")
            progress_bar.progress(40)
            
            status_text.text("👕 Warping cloth...")
            progress_bar.progress(60)
            
            status_text.text("🎨 Generating try-on...")
            progress_bar.progress(80)
            
            # Run inference
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_output:
                output_path = tmp_output.name
            
            result = pipeline(person_path, cloth_path, output_path)
            
            status_text.text("✨ Finalizing...")
            progress_bar.progress(100)
            
            # Display result
            result_img = Image.open(output_path)
            with col3:
                st.image(result_img, caption="Virtual Try-On Result", use_column_width=True)
            
            # Download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Result",
                    data=f.read(),
                    file_name="tryon_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Try-on completed successfully!")
            
            # Cleanup
            os.unlink(person_path)
            os.unlink(cloth_path)
            os.unlink(output_path)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Please make sure all models are trained and checkpoints exist.")
else:
    st.info("👆 Please upload both person and cloth images to get started!")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🤖 Powered by Deep Learning | Built with PyTorch & Streamlit</p>
    <p>Virtual Try-On System - AI Model Pipeline</p>
</div>
""", unsafe_allow_html=True)
