"""
Image Viewer Component
======================

Displays CT slice images with colormap options and window/level controls.
"""

import streamlit as st
import requests
from typing import Optional
import io
from PIL import Image


def render_image_viewer(
    nodule_id: str,
    api_base_url: str = "http://localhost:8000",
    is_synthetic: bool = False,
    show_controls: bool = True
):
    """
    Render the CT image viewer component.
    
    Args:
        nodule_id: The nodule ID to display
        api_base_url: Base URL for the API
        is_synthetic: Whether the image is synthetic
        show_controls: Whether to show colormap/upscale controls
    """
    st.subheader("🫁 CT Image")
    
    # Controls
    if show_controls:
        col1, col2 = st.columns(2)
        with col1:
            colormap = st.selectbox(
                "Colormap",
                options=["gray", "bone", "hot"],
                index=0,
                key=f"colormap_{nodule_id}"
            )
        with col2:
            upscale = st.slider(
                "Zoom",
                min_value=1,
                max_value=8,
                value=4,
                key=f"upscale_{nodule_id}"
            )
    else:
        colormap = "gray"
        upscale = 4
    
    # Synthetic badge
    if is_synthetic:
        st.info("ℹ️ **Synthetic Image** - Generated for demonstration purposes")
    
    # Load and display image
    try:
        image_url = f"{api_base_url}/nodules/{nodule_id}/image?colormap={colormap}&upscale={upscale}"
        response = requests.get(image_url, timeout=10)
        
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption=f"Nodule {nodule_id}", use_container_width=True)
        else:
            st.error(f"Failed to load image: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Could not connect to API. Showing placeholder.")
        render_image_placeholder(nodule_id)
    except Exception as e:
        st.error(f"Error loading image: {e}")


def render_image_placeholder(nodule_id: str):
    """Render a placeholder when image is unavailable."""
    st.markdown(f"""
    <div style="
        background-color: #1a1a2e;
        border: 2px dashed #4a4a6a;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        color: #8a8aa0;
    ">
        <div style="font-size: 3em; margin-bottom: 10px;">🫁</div>
        <div style="font-size: 1.1em;">CT Image</div>
        <div style="font-size: 0.9em; margin-top: 5px;">Nodule {nodule_id}</div>
        <div style="font-size: 0.8em; margin-top: 15px; color: #6a6a8a;">
            Start the API server to view images
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_image_from_array(
    image_array,
    nodule_id: str,
    is_synthetic: bool = False
):
    """
    Render image directly from numpy array (for offline mode).
    
    Args:
        image_array: Numpy array of shape (H, W) with values in [0, 1]
        nodule_id: The nodule ID
        is_synthetic: Whether the image is synthetic
    """
    import numpy as np
    
    st.subheader("🫁 CT Image")
    
    if is_synthetic:
        st.info("ℹ️ **Synthetic Image** - Generated for demonstration purposes")
    
    if image_array is not None:
        # Convert to uint8 for display
        img_uint8 = (image_array * 255).astype(np.uint8)
        
        # Upscale for better display
        pil_image = Image.fromarray(img_uint8, mode='L')
        new_size = (pil_image.width * 4, pil_image.height * 4)
        pil_image = pil_image.resize(new_size, Image.Resampling.NEAREST)
        
        st.image(pil_image, caption=f"Nodule {nodule_id}", use_container_width=True)
    else:
        render_image_placeholder(nodule_id)


def render_features_table(features: dict):
    """
    Render a table of nodule features.
    
    Args:
        features: Dictionary of nodule features
    """
    st.subheader("📊 Nodule Features")
    
    # Key features to display
    display_features = [
        ("diameter_mm", "Diameter (mm)", "📏"),
        ("malignancy", "Malignancy Score", "⚠️"),
        ("malignancy_label", "Malignancy", "🏷️"),
        ("texture", "Texture Score", "🔲"),
        ("texture_label", "Texture", "🏷️"),
        ("margin", "Margin Score", "📐"),
        ("margin_label", "Margin", "🏷️"),
        ("spiculation", "Spiculation Score", "🌟"),
        ("lobulation", "Lobulation Score", "🔄"),
        ("calcification", "Calcification", "💎"),
        ("sphericity", "Sphericity", "⚽"),
        ("subtlety", "Subtlety", "👁️"),
    ]
    
    # Create two columns for features
    col1, col2 = st.columns(2)
    
    displayed = 0
    for key, label, icon in display_features:
        if key in features and features[key] is not None:
            target_col = col1 if displayed % 2 == 0 else col2
            with target_col:
                value = features[key]
                if isinstance(value, float):
                    value = f"{value:.1f}"
                st.markdown(f"{icon} **{label}:** {value}")
            displayed += 1
    
    # Source badge
    source = features.get("source", "Unknown")
    if source == "XML Import":
        st.success("✅ Real LIDC-IDRI data")
    else:
        st.warning("⚠️ Synthetic/fallback data")
