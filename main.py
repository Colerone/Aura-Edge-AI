import time

import streamlit as st
from rich import layout
from PIL import Image
import numpy as np

st.set_page_config(
    page_title= "Aura Edge",
    page_icon = "🩺",
    layout = "wide"
)

import streamlit as st

# Create 3 columns where the middle one takes the most space
# st.markdown(
#     '<h1 style="text-align: center; margin-top: 0px; color: #0077B6;">Aura Edge</h1>',
#     unsafe_allow_html=True
# )
#
# st.markdown(
#     "<h4 style='text-align: center; margin-top: 0px; padding-top:-20px; color: #0077B6;'>AI-Powered Body Skin & Wound Detection System</h4>",
#     unsafe_allow_html = True
# )

st.markdown(
    """
        <div style="text-align: center;
        background: linear-gradient(90deg,#27adaf,#004c73,#004c73,#004b72,#004b72); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        ">
            <h1 style='margin-top: 0px;'>Aura Edge</h1>
            <p style='font-size: 1.5em; font-weight: bold; margin-top: -15px;'>AI-Powered Body Skin & Wound Detection System</p>
        </div>
    """,
    unsafe_allow_html = True
)

def preprocess_image(image: Image.Image) ->np.ndarray:
    """ change photos to standard size and adjust pixel from 0 to 1"""
    resized_image  = image.resize((224,224))
    img_array = np.array(resized_image,dtype=np.float32)/255
    return img_array

def run_yolo_detection(img_array: np.ndarray):
    """ run yolo detection model to find skin and wound detection"""
    time.sleep(0.5)
    return{"box" : [50,50,150,150], "confidence" : 0.92, "label" : "Surgical Wound"}


def run_unet_segmentation(img_array: np.ndarray):
    """ U-Net: Pixel-level calculation of wound area (Mock)"""
    time.sleep(0.6)
    return{"affected_area_pixel" : 4250, "esitmated_surface_area_cm2" : 12.5}

def run_efficient_classification(img_array:np.ndarray):
    """to see skin condition and severity"""
    time.sleep(0.4)
    return {
        "severity": "Moderate",
        "condition": "Acute Inflammation",
        "recommended_ingredients": ["Centella Asiatica", "Zinc Oxide", "Panthenol (Vitamin B5)"]
    }







