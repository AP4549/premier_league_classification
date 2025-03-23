import streamlit as st # type: ignore
from streamlit.web.cli import main # type: ignore
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    
st.set_page_config(page_title="Deep Learning App", layout="wide")

# Encode the image to base64
background_image_base64 = encode_image("background.jpg")  # Replace with your image path

# Create the background image CSS style with the base64-encoded string
background_image = f"url(data:image/jpeg;base64,{background_image_base64})"

st.markdown(f"""
    <style>
        .stApp {{
            position: relative;
            background-image: {background_image};
            background-size: cover;  /* Ensure the image covers the full screen */
            background-position: center center;  /* Center the image */
            background-attachment: fixed;  /* Keep the background fixed during scroll */
            height: 100vh;  /* Ensure the background covers the entire page */
        }}

        /* Football theme text colors */
        h1 {{
            color: gold;  /* Gold for heading */
            text-align: center;
        }}
        h2 {{
            color: #228B22;  /* Green for subheadings */
            text-align: center;
        }}
        p {{
            color: #FFFFFF;  /* White for text */
            font-size: 18px;
            text-align: center;
        }}
        ul {{
            color: #FFFFFF;  /* White for list items */
            font-size: 0px;
            text-align: center;
            list-style-type: none;
        }}
        ul li {{
            margin: 10px 0;
        }}
    </style>
""", unsafe_allow_html=True)

# Custom CSS to apply the background image
st.markdown(f"""
    <style>
        .stApp {{
            background-image: {background_image};
            background-size: cover;  /* Ensure the image covers the full screen */
            background-position: center center;  /* Center the image */
            background-attachment: fixed;  /* Keep the background fixed during scroll */
        }}
    </style>
""", unsafe_allow_html=True)

# Add some content to verify the background image is applied
st.markdown("<h1 style='text-align: center; color: gold; font-size: 70px; margin-bottom:100px'>⚽ Premier League Teams Crest Classification ⚽</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; font-size: 30px; margin-bottom:100px'>
        Welcome to the <strong>Premier League Teams Crest Classification Dashboard</strong>! 
        This interactive app allows you to explore deep learning models that classify the crests of Premier League teams. 
        Using various model architectures like ResNet, EfficientNet, and others, the app enables you to visualize how these models 
        recognize and classify team logos. You can also analyze training benchmarks and test model inference with new images of crests. 
        Whether you're a football fan or a deep learning enthusiast, this app provides an engaging way to explore the power of computer vision in sports.
    </p>
""", unsafe_allow_html=True)

st.markdown("<h2 style ='text-align: left; font-size:30px'>Key Features</h2>", unsafe_allow_html=True)

st.markdown("""
    <p style ='color: #000000; font-weight: bold; font-size: 20px; text-align: left' ><span style="font-size: 10px;">⚫</span>🧠 View Model Architectures</p>
    <p style ='color: #000000; font-weight: bold; font-size: 20px; text-align: left' ><span style="font-size: 10px;">⚫</span>📊 Analyze Training Benchmarks</p>
    <p style ='color: #000000; font-weight: bold; font-size: 20px; text-align: left' ><span style="font-size: 10px;">⚫</span>🔍 Test Model Inference</p>
    <p style ='color: #000000; font-weight: bold; font-size: 20px; text-align: left' ><span style="font-size: 10px;">⚫</span>🚀 Object Detection</p>
    <p style ='color: #000000; font-weight: bold; font-size: 20px; text-align: left' ><span style="font-size: 10px;">⚫</span>🖼️ Image Segmentation</p>
    <p style ='color: #000000; font-weight: bold; font-size: 20px; text-align: left' ><span style="font-size: 10px;">⚫</span>🎨 Neural Style Transfer</p>
""", unsafe_allow_html=True)

st.markdown("<h2 style ='text-align: left; font-size:30px'>New Features</h2>", unsafe_allow_html=True)

st.markdown("""
    <h3 style ='color: #228B22; text-align: left;'>🚀 Object Detection</h3>
    <p style ='text-align: left;'>
        This feature uses advanced deep learning models to detect football crests in images, even in difficult locations where classification models may fail. 
        The object detection model leverages YOLO or Faster R-CNN architectures for real-time predictions, making it robust in identifying crests from complex backgrounds.
    </p>

    <h3 style ='color: #228B22; text-align: left;'>🖼️ Image Segmentation</h3>
    <p style ='text-align: left;'>
        Image segmentation allows us to separate different football club crests within an image.
    </p>

    <h3 style ='color: #228B22; text-align: left;'>🎨 Neural Style Transfer</h3>
    <p style ='text-align: left;'>
        This feature transforms images by applying artistic styles to them. 
    </p>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["🏠 Home", "🧠 Model Architecture", "📊 Benchmarks & Stats", "🔍 Make Predictions", "🚀 Object Detection", "🖼️ Image Segmentation", "🎨 Neural Style Transfer"])

# Redirect based on selection
if page == "🧠 Model Architecture":
    st.switch_page("pages/1_Model Architecture.py")
elif page == "📊 Benchmarks & Stats":
    st.switch_page("pages/2_Benchmarks.py")
elif page == "🔍 Make Predictions":
    st.switch_page("pages/3_Inference.py")
elif page == "🚀 Object Detection":
    st.switch_page("pages/4_Object Detection.py")
elif page == "🖼️ Image Segmentation":
    st.switch_page("pages/5_Image Segmentation.py")
elif page == "🎨 Neural Style Transfer":
    st.switch_page("pages/6_NST.py")
