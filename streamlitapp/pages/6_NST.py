import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import PIL.Image # type: ignore
from tensorflow.keras.applications import vgg19 # type: ignore
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Set page configuration
st.set_page_config(page_title="Neural Style Transfer", layout="wide")

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

st.title("🎨 Neural Style Transfer")
st.write("Upload a content image and a style image to apply style transfer.")

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["🎨 Neural Style Transfer", "🔍 Make Predictions", "🏠 Home", "🧠 Model Architecture", "📊 Benchmarks & Stats", "🚀 Object Detection", "🖼️ Image Segmentation"])

    if page == "🏠 Home":
        st.switch_page("Home.py")
    elif page == "🧠 Model Architecture":
        st.switch_page("pages/1_Model Architecture.py")
    elif page == "📊 Benchmarks & Stats":
        st.switch_page("pages/2_Benchmarks.py")
    elif page == "🔍 Make Predictions":
        st.switch_page("pages/3_Inference.py")
    elif page == "🚀 Object Detection":
        st.switch_page("pages/4_Object Detection.py")
    elif page == "🖼️ Image Segmentation":
        st.switch_page("pages/5_Image Segmentation.py")

# Load VGG19 model layers
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)

# Gram matrix for style representation
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
    return result / num_locations

# Define layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Model class
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = outputs[:self.num_style_layers], outputs[self.num_style_layers:]

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

@st.cache_resource
def load_model():
    model = StyleContentModel(style_layers, content_layers)
    model.build(input_shape=(None, 512, 512, 3))
    return model

# Convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Load and preprocess image
def load_img(uploaded_file):
    max_dim = 512
    img = PIL.Image.open(uploaded_file)
    img = img.convert("RGB")
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = img[tf.newaxis, :]
    return img

# Upload images
content_file = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])

if content_file and style_file:
    content_image = load_img(content_file)
    style_image = load_img(style_file)
    
    st.image([content_file, style_file], caption=["Content Image", "Style Image"], width=250)
    
    model = load_model()
    extractor = StyleContentModel(style_layers, content_layers)
    
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    
    image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    # Updated Style & Content Weights
    style_weight = 1e4  # Increased style weight
    content_weight = 5e2  # Decreased content weight
    
    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
        style_loss *= style_weight / len(style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
        content_loss *= content_weight / len(content_layers)
        
        return style_loss + content_loss
    
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
    
    if st.button("Start Style Transfer"):  
        st.write("Processing... ⏳")
        for _ in range(15):
            train_step(image)
        
        output_image = tensor_to_image(image)
        
        # Display centered and small output image
        st.markdown("<h3 style='text-align: center;'>Styled Image</h3>", unsafe_allow_html=True)
        st.image(output_image, width=400, use_column_width=False)
        st.success("Style Transfer Complete! ✅")
