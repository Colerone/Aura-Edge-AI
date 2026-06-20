import time

import streamlit as st
from rich import layout
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from streamlit import spinner

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
            <h2 style='margin-top: 0px;'>Aura Edge</h2>
            <p style='font-size: 1.5em;  margin-top: -20px;'>AI-Powered Body Skin & Wound Detection System</p>
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


def draw_detections(original_image: Image.Image, yolo_res: dict):
    """ ပုံပေါ်တွင် Bounding Box နှင့် Label ဆွဲပေးမည့် function """
    draw_img = original_image.copy()
    draw = ImageDraw.Draw(draw_img)

    # မူရင်းပုံ၏ Size ကိုယူခြင်း
    orig_w, orig_h = original_image.size

    # YOLO က 224x224 ပေါ်မှာ ရှာတာဖြစ်လို့ မူရင်းပုံ Size နဲ့ ကိုက်အောင် Scale ပြန်တွက်ခြင်း
    # (xmin, ymin, xmax, ymax)
    box = yolo_res["box"]
    xmin = (box[0] / 224) * orig_w
    ymin = (box[1] / 224) * orig_h
    xmax = (box[2] / 224) * orig_w
    ymax = (box[3] / 224) * orig_h

    # Bounding Box ဆွဲခြင်း (အနီရောင်၊ အနားသတ်အထူ ၅)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF0000", width=8)

    # Label စာသားထည့်ခြင်း
    label_text = f"{yolo_res['label']} {yolo_res['confidence'] * 100:.1f}%"

    # စာသားနောက်ခံ Box ငယ်လေးဆွဲခြင်း
    draw.rectangle([xmin, ymin - 35, xmin + 250, ymin], fill="#FF0000")
    draw.text((xmin + 5, ymin - 30), label_text, fill="white")

    return draw_img

def run_unet_segmentation(img_array: np.ndarray):
    """ U-Net: Pixel-level calculation of wound area (Mock)"""
    time.sleep(0.6)
    return{"affected_area_px" : 4250, "estimated_surface_area_cm2" : 12.5}

def run_efficient_classification(img_array:np.ndarray):
    """to see skin condition and severity"""
    time.sleep(0.4)
    return {
        "severity": "Moderate",
        "condition": "Acute Inflammation",
        "recommended_ingredients": ["Centella Asiatica", "Zinc Oxide", "Panthenol (Vitamin B5)"]
    }

col1,col2 = st.columns([1,2])

with col1:
    st.markdown(
        """
            <div style="margin-top: 15px;">
                <p  style='font-weight:bold;'>📸 Image Ingestion</p>
            </div>
        """,

        unsafe_allow_html = True
    )
    upload_Img = st.file_uploader(
        "Upload a skin & wound Image: ",
        type=["png", "jpg", "jpeg"]
    )
    if upload_Img is not None:
        image = Image.open(upload_Img)
        st.image(image, caption="Image Preview", use_container_width=True)

        with st.spinner('Processing image...'):
            processed_data = preprocess_image(image)




with col2:
   st.markdown(
       """
        <div style="margin-top: 15px;">
            <p style='font-weight: bold;'>🔬💡 Real Time AI</p>
        </div>
       """,
       unsafe_allow_html = True
   )
   if upload_Img is not None:
        with st.spinner("Running AI Multi-Model"):
            # Model များကို ခေါ်ယူခြင်း
            yolo_res = run_yolo_detection(processed_data)
            unet_res = run_unet_segmentation(processed_data)
            eff_res = run_efficient_classification(processed_data)

        st.success("Detection Complete Successfully🚀")

        annotated_image = draw_detections(image, yolo_res)
        st.image(annotated_image, caption="AI Analysis Visualization (Target Localization)", use_container_width=True)

        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown("### 🎯 1. Target Localization (YOLOv11)")
            st.metric(label="Detected Condition", value=yolo_res["label"])
            st.write(f"**Confidence Score:** {yolo_res['confidence'] * 100:.2f}%")

        with m_col2:
            st.markdown("### 📐 2. Boundary & Area Extraction (U-Net)")
            st.metric(label="Estimated Wound Surface Area", value=f"{unet_res['estimated_surface_area_cm2']} cm²")
            st.caption(f"Calculated from {unet_res['affected_area_px']} pixels at current resolution.")

        st.write("---")

        # --- FEATURE 3: CLASSIFICATION & RECOMMENDATION ---
        st.markdown("### 🧪 3. Profile Categorization (EfficientNet)")

        # Severity level အလိုက် အရောင်ပြောင်းခြင်း
        severity = eff_res["severity"]
        if severity == "Moderate":
            st.warning(f"Severity Level: {severity}")
        elif severity == "High":
            st.error(f"Severity Level: {severity}")
        else:
            st.success(f"Severity Level: {severity}")

        st.write(f"**Pathological Profile:** {eff_res['condition']}")

        st.markdown("**🔬 Ingredient-Focused Recommendations:**")
        for ingredient in eff_res["recommended_ingredients"]:
            st.markdown(f"- `{ingredient}`")

   else:
       st.info("Please upload an image from the left panel to trigger the AI analysis pipeline.")


