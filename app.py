import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image

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
    """ Runs YOLO object detection on an image. """
    results = yolo_model(image)  # Run YOLO detection
    img = np.array(image)
    detections = []
    cropped_images = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cropped = img[y1:y2, x1:x2]  # Crop detected object
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                cropped_images.append(cropped)
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf
                })

    return detections, cropped_images, img

# Classification function
def classify_object(image):
    """ Classifies a single cropped object using EfficientNet. """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to("cpu")  # Move to CPU
    with torch.no_grad():
        outputs = efficientnet_model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Streamlit UI
st.title("🐔 Poultry Disease Detection System")
st.write("Upload an image to detect and classify diseases in poultry.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    
    # Run detection
    detections, cropped_images, img = detect_objects(image)

    if detections:
        st.write(f"✅ Detected {len(detections)} feces areas!")

        fig, ax = plt.subplots(figsize=(8, 6))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
        ax.imshow(img)
        ax.axis("off")

        for idx, (detection, cropped) in enumerate(zip(detections, cropped_images)):
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            label = classify_object(cropped)
            st.write(f"📍 **Object {idx+1}: {label} (Confidence: {conf:.2f})**")
            
            # Draw bounding box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="green", linewidth=2, fill=False))
            ax.text(x1, y1 - 10, f"{label}", fontsize=12, color="green", weight="bold", bbox=dict(facecolor="white", alpha=0.8))

        # Display final image with bounding boxes
        st.pyplot(fig)

    else:
        st.write("⚠️ No objects detected in the image.")