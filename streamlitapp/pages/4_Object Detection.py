import streamlit as st # type: ignore
from ultralytics import YOLO # type: ignore
from PIL import Image, ImageDraw # type: ignore
import numpy as np # type: ignore
import base64 # type: ignore
from io import BytesIO # type: ignore

st.set_page_config(page_title="Object Detection", layout="wide")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load the model from best.pt
@st.cache_resource()
def load_model():
    model = YOLO("Models/best.pt")  # Load best.pt model
    return model

model = load_model()

background_image_base64 = encode_image("background.jpg")  # Replace with your image path

# Create the background image CSS style with the base64-encoded string
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
        .centered-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["🚀 Object Detection", "🔍 Make Predictions", "🏠 Home", "🧠 Model Architecture", "📊 Benchmarks & Stats", "🖼️ Image Segmentation", "🎨 Neural Style Transfer"])

if page == "🏠 Home":
    st.switch_page("Home.py")
elif page == "🧠 Model Architecture":
    st.switch_page("pages/1_Model_Architecture.py")
elif page == "📊 Benchmarks & Stats":
    st.switch_page("pages/2_Benchmarks.py")
elif page == "🔍 Make Predictions":
    st.switch_page("pages/3_Inference.py")
elif page == "🖼️ Image Segmentation":
    st.switch_page("pages/5_Image Segmentation.py")
elif page == "🎨 Neural Style Transfer":
    st.switch_page("pages/6_NST.py")

# Streamlit UI
st.title("🚀 Object Detection")
st.write("Upload an image, and the model will detect objects.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Center input image
    st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", width=400)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Perform inference with adjusted confidence threshold
    results = model(image, conf=0.4)
    
    # Extract detected class names and draw bounding boxes
    detected_classes = set()
    draw = ImageDraw.Draw(image)
    for result in results:
        for box in result.boxes:
            detected_classes.add(result.names[int(box.cls)])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # Display results
    st.write("### Detection Results:")
    st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
    st.image(image, caption="Detected Objects", width=400)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display detected class names
    st.write("#### Detected Classes:")
    for detected_class in detected_classes:
        st.write(f"- {detected_class}")
