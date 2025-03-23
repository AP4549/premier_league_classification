import streamlit as st  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import cv2  # type: ignore
from PIL import Image  # type: ignore
from ultralytics import YOLO  # type: ignore
import base64  # type: ignore

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

st.set_page_config(page_title="Image Segmentation", layout="wide")

# Encode and set background image
background_image_base64 = encode_image("background.jpg")  
background_image = f"url(data:image/jpeg;base64,{background_image_base64})"
st.markdown(f"""
    <style>
        .stApp {{
            position: relative;
            background-image: {background_image};
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            height: 100vh;
        }}
    </style>
""", unsafe_allow_html=True)

st.title("🖼️ Image Segmentation")
st.write("Upload an image to perform segmentation using YOLO.")

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["🖼️ Image Segmentation", "🔍 Make Predictions", "🏠 Home", "🧠 Model Architecture", "📊 Benchmarks & Stats", "🚀 Object Detection", "🎨 Neural Style Transfer"])

    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)  # Dynamic confidence threshold

    if page == "🏠 Home":
        st.switch_page("Home.py")
    elif page == "🧠 Model Architecture":
        st.switch_page("pages/1_Model Architecture.py")
    elif page == "📊 Benchmarks & Stats":
        st.switch_page("pages/2_Benchmarks.py")
    elif page == "🔍 Make Predictions":
        st.switch_page("pages/3_Predictions.py")
    elif page == "🚀 Object Detection":
        st.switch_page("pages/4_Object Detection.py")
    elif page == "🎨 Neural Style Transfer":
        st.switch_page("pages/6_NST.py")

# Load YOLO model
model = YOLO("Models/best.pt")  

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
    image_cv = np.array(image)

    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Segment Image"):
        # ✅ Convert to OpenCV format (but keep colors unchanged)
        image_cv_bgr = image_cv  # Keep as is, no color conversion

        # ✅ Perform inference with YOLO using dynamic confidence threshold
        results = model(image_cv_bgr, conf=confidence_threshold)

        if not results or len(results[0].boxes) == 0:
            st.warning("⚠️ No objects detected. Try another image.")
        else:
            object_images = []
            seen_hashes = set()  # Stores unique hashable representations of images

            def remove_background(cropped_object):
                mask = np.zeros(cropped_object.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                rect = (5, 5, cropped_object.shape[1] - 10, cropped_object.shape[0] - 10)
                cv2.grabCut(cropped_object, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
                object_no_bg = cropped_object * mask[:, :, np.newaxis]
                return object_no_bg, mask

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for (x1, y1, x2, y2) in boxes:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cropped_object = image_cv[y1:y2, x1:x2]

                    # ✅ Ensure the cropped object is not empty
                    if cropped_object.size == 0:
                        continue

                    object_no_bg, mask = remove_background(cropped_object)
                    obj_img = Image.fromarray(object_no_bg).convert("RGBA")
                    alpha = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                    obj_img.putalpha(alpha)

                    # ✅ Remove duplicates by checking pixel data hash
                    img_hash = hash(obj_img.tobytes())
                    if img_hash not in seen_hashes:
                        seen_hashes.add(img_hash)
                        object_images.append(obj_img)

            # ✅ Standardize output image size
            output_size = (150, 150)  # Fixed small size for all segmented images
            object_images = [img.resize(output_size) for img in object_images]

            # ✅ Arrange segmented images in a row
            total_width = min(len(object_images) * output_size[0], 600)
            max_height = output_size[1]
            combined_image = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 0))
            x_offset = 0

            for img in object_images:
                combined_image.paste(img, (x_offset, 0), img.split()[3])
                x_offset += output_size[0]

            # Display segmented image
            st.image(combined_image, caption="Segmented Objects", width=400)
            
            if not object_images:
                st.warning("⚠️ No objects extracted. The detected boxes may be incorrect.")
