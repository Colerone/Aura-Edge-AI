import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Aura Edge AI - Facial Skin Detection & Recommendation System", layout="wide")
st.title("Aura Edge AI - Facial Skin Detection & Recommendation System")
st.write("Upload a photo of your face and check for skin problems.")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("`best.pt` model file cannot be found. Please check if it is in your Project Folder.")
    st.stop()

acne_thresh = 0.25
rosacea_thresh = 0.50
melasma_thresh = 0.25
iou_thresh = 0.50

class_thresholds = {
    'Acne': acne_thresh,
    'Rosacea': rosacea_thresh,
    'Melasma': melasma_thresh
}

uploaded_file = st.file_uploader("Choose a face photo to analyze...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("AI is analyzing your skin in detail..."):
        results = model.predict(source=image, conf=0.10, iou=iou_thresh)

        for result in results:
            keep_indices = []
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                cls_name = result.names[cls_id]
                conf_val = float(box.conf[0].item())

                threshold = class_thresholds.get(cls_name, 0.25)
                if conf_val >= threshold:
                    keep_indices.append(i)

            result.boxes = result.boxes[keep_indices]

            annotated_img_bgr = result.plot()
            annotated_img_rgb = annotated_img_bgr[:, :, ::-1]  # Convert BGR to RGB

            with col2:
                st.subheader(" AI Detection Result")
                st.image(annotated_img_rgb, use_container_width=True)

            st.write("---")
            st.subheader(" Detection Summary")

            boxes = result.boxes
            total_detections = len(boxes)

            if total_detections > 0:
                st.success(f"**Total skin issues detected:** {total_detections}")

                class_names = result.names
                class_ids = boxes.cls.tolist()

                counts_per_class = {}
                for cls_id in class_ids:
                    cls_name = class_names[int(cls_id)]
                    counts_per_class[cls_name] = counts_per_class.get(cls_name, 0) + 1
                cols = st.columns(len(counts_per_class))
                for idx, (name, count) in enumerate(counts_per_class.items()):
                    with cols[idx]:
                        st.metric(label=f" {name}", value=f"{count}")
            else:
                st.info("No skin issues detected. Your skin looks great!")