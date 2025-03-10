import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os
import base64
import streamlit.components.v1 as components

# Paths to models
YOLO_MODEL_PATH = "models/best.pt"
EFFICIENTNET_MODEL_PATH = "models/final_efficientnet_model_b0.pth"

# Define class names based on dataset
class_names = ["cocci", "healthy", "ncd", "salmo"]

# Load YOLO Model
@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL_PATH)

# Load EfficientNet Model
@st.cache_resource
def load_efficientnet():
    model = EfficientNet.from_name("efficientnet-b0")
    model._fc = torch.nn.Linear(model._fc.in_features, len(class_names))
    state_dict = torch.load(EFFICIENTNET_MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to("cpu")

# Load models
yolo_model = load_yolo()
efficientnet_model = load_efficientnet()

# Define image transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Object detection function
def detect_objects(image):
    results = yolo_model(image)
    img = np.array(image)
    detections = []
    cropped_images = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cropped = img[y1:y2, x1:x2]
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                cropped_images.append(cropped)
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf
                })
    return detections, cropped_images, img

# Classification function
def classify_object(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        outputs = efficientnet_model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Streamlit UI
st.title("🐔 Poultry Disease Detection System")
st.write("Upload an image to detect and classify diseases in poultry.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Load uploaded image and perform detection
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    detections, cropped_images, img = detect_objects(image)

    if detections:
        st.write(f"✅ Detected {len(detections)} feces areas!")
        fig, ax = plt.subplots(figsize=(8, 6))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")

# Modify the visualization section where you draw bounding boxes
        for idx, (detection, cropped) in enumerate(zip(detections, cropped_images)):
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            label = classify_object(cropped)
            
            # Update the text to include confidence
            display_text = f"{label} ({conf:.2f})"
            
            # Add rectangle for bounding box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="green", linewidth=2, fill=False))
            
            # Add text with confidence
            ax.text(x1, y1 - 10, display_text, fontsize=12, color="green", weight="bold", 
                    bbox=dict(facecolor="white", alpha=0.8))

        st.pyplot(fig)
    else:
        st.write("⚠️ No objects detected in the image.")

# Display example images from "example_images" folder in a horizontal carousel with arrows
EXAMPLES_FOLDER = "example_images"
st.write("### Example Images")

if os.path.exists(EXAMPLES_FOLDER) and os.path.isdir(EXAMPLES_FOLDER):
    example_files = [f for f in os.listdir(EXAMPLES_FOLDER) if f.lower().endswith((".jpg", "png", "jpeg"))]
    if example_files:
        image_html = """
        <style>
        .scroll-container {
            display: flex;
            overflow-x: auto;
            white-space: nowrap;
            padding: 10px 0;
            scroll-behavior: smooth;
            align-items: center;
            width: 100%;
        }
        .image-item {
            width: 150px;
            margin: 0 10px;
            flex-shrink: 0;
            text-align: center;
        }
        .image-container {
            width: 150px;
            height: 150px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            margin-bottom: 8px;
        }
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            cursor: pointer;
        }
        .image-caption {
            font-size: 12px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #555;
        }
        .scroll-button {
            cursor: pointer;
            font-size: 24px;
            border: none;
            background: rgba(0,0,0,0.1);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 5px;
            color:
        }
        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
        }
        /* For Webkit browsers like Chrome/Safari */
        .scroll-container::-webkit-scrollbar {
            height: 8px;
        }
        .scroll-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        .scroll-container::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        .scroll-container::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        </style>
        <div>
            <div class='scroll-container' id='carousel'>
        """
        for img_file in example_files:
            img_path = os.path.join(EXAMPLES_FOLDER, img_file)
            with open(img_path, "rb") as img_file_obj:
                encoded_img = base64.b64encode(img_file_obj.read()).decode()
            image_html += f"""
                <div class='image-item'>
                    <div class='image-container'>
                        <img src='data:image/png;base64,{encoded_img}' onclick='selectImage("{img_path}")'>
                    </div>
                    <div class='image-caption'>{img_file}</div>
                </div>
            """
        
        image_html += """
            </div>
            <div class='controls'>
                <button class='scroll-button' onclick='scrollLeft()'>&#9664;</button>
                <button class='scroll-button' onclick='scrollRight()'>&#9654;</button>
            </div>
        </div>
        <script>
        function scrollLeft() {
            document.getElementById('carousel').scrollBy({left: -300, behavior: 'smooth'});
        }
        function scrollRight() {
            document.getElementById('carousel').scrollBy({left: 300, behavior: 'smooth'});
        }
        function selectImage(path) {
            // You can implement selection functionality here
            console.log("Selected image:", path);
            // This would need integration with Streamlit
        }
        </script>
        """
        components.html(image_html, height=230)
    else:
        st.write("No example images found.")
else:
    st.write("Example images folder not found.")