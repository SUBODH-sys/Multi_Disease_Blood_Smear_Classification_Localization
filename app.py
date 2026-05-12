# import streamlit as st
# import torch
# import torch.nn as nn
# import timm
# import cv2
# import numpy as np
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import torchvision.transforms.functional as TF

# # Page configuration
# st.set_page_config(page_title="Multi-Disease Classifier", layout="wide", page_icon="🔬")

# # Constants
# LABELS_NAMES = ['Normal', 'Sickle', 'Malaria', 'Leukemia']
# MODEL_PATH = "classifier_best_cv.pth"
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Custom CSS for black base + blue aesthetics
# st.markdown("""
# <style>
#     /* Background and Layout */
#     .stApp {
#         background-color: #0a0a0f; /* near black */
#         background-image: linear-gradient(180deg, #0a0a0f 0%, #111827 100%);
#     }
#     .main .block-container {
#         padding-top: 2rem;
#         padding-bottom: 3rem;
#         max-width: 900px;
#     }
    
#     /* Typography */
#     h1 {
#         background: -webkit-linear-gradient(45deg, #38bdf8, #3b82f6);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         font-family: 'Inter', sans-serif;
#         text-align: center;
#         margin-bottom: 0.5rem;
#         font-weight: 800;
#         letter-spacing: -0.5px;
#     }
#     h2, h3, h4 {
#         color: #93c5fd; /* light blue */
#         font-family: 'Inter', sans-serif;
#         font-weight: 600;
#     }
#     p, .stMarkdown {
#         color: #cbd5e1; /* light slate */
#     }
    
#     /* Upload Box Styling */
#     [data-testid="stFileUploadDropzone"] {
#         background-color: rgba(17, 24, 39, 0.8); /* dark slate */
#         border: 2px dashed #3b82f6;
#         border-radius: 16px;
#         padding: 2rem;
#         transition: all 0.3s ease-in-out;
#     }
#     [data-testid="stFileUploadDropzone"]:hover {
#         border-color: #60a5fa;
#         background-color: rgba(30, 58, 138, 0.3);
#         box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3), 0 8px 10px -6px rgba(59, 130, 246, 0.2);
#     }
#     [data-testid="stFileUploadDropzone"] p {
#         color: #e2e8f0 !important;
#     }

#     /* Progress bars */
#     .stProgress .st-bo {
#         background-color: #3b82f6;
#     }
    
#     /* Buttons */
#     div.stButton > button {
#         background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
#         color: white;
#         border-radius: 12px;
#         border: none;
#         padding: 0.6rem 1.2rem;
#         font-weight: 600;
#         box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.4);
#         transition: all 0.3s ease;
#     }
#     div.stButton > button:hover {
#         box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
#         transform: translateY(-2px);
#         color: white;
#     }
    
#     /* Prediction Box */
#     .prediction-box {
#         background: #1e293b; /* dark slate */
#         padding: 2rem;
#         border-radius: 16px;
#         box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
#         margin-top: 0.5rem;
#         margin-bottom: 1.5rem;
#         border-top: 4px solid #3b82f6;
#         transition: transform 0.3s ease;
#     }
#     .prediction-box:hover {
#         transform: translateY(-4px);
#     }
#     .prediction-label {
#         text-align: center; 
#         margin: 0 0 0.5rem 0; 
#         color: #93c5fd; 
#         font-size: 2.2rem; 
#         font-weight: 800;
#         letter-spacing: -0.5px;
#     }
#     .prediction-confidence {
#         text-align: center; 
#         margin: 0; 
#         font-size: 1.2rem; 
#         color: #EF4444;
#         font-weight: 600;
#         background: #1e3a8a;
#         display: inline-block;
#         padding: 0.4rem 1rem;
#         border-radius: 20px;
#     }
    
#     /* Images */
#     [data-testid="stImage"] img {
#         border-radius: 12px;
#         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
#     }
    
#     /* Dividers */
#     hr {
#         border-top: 1px solid #334155;
#         margin: 2rem 0;
#     }
    
#     /* Sidebar */
#     [data-testid="stSidebar"] {
#         background-color: #0f172a;
#     }
    
#     /* Text inputs, selectboxes */
#     .stTextInput > div > div > input, .stSelectbox > div > div {
#         background-color: #1e293b;
#         color: #e2e8f0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Define Model
# class DiseaseClassifier(nn.Module):
#     def __init__(self, num_classes=4, backbone='efficientnet_b0', pretrained=False):
#         super().__init__()
#         self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
#         self.head = nn.Sequential(
#             nn.Linear(self.backbone.num_features, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
#     def forward(self, x):
#         return self.head(self.backbone(x))

# @st.cache_resource(show_spinner=False)
# def load_model():
#     model = DiseaseClassifier(num_classes=4, pretrained=False)
#     state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
#     model.load_state_dict(state_dict)
#     model.to(DEVICE)
#     model.eval()
#     return model

# # Define Transforms
# inference_transform = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])

# def preprocess_image(image_bytes):
#     # Convert bytes to numpy array
#     img_np = np.frombuffer(image_bytes, np.uint8)
#     img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
#     # Resize for display
#     img_resized = cv2.resize(img_rgb, (224, 224))
    
#     # Apply transforms for model
#     img_tensor = inference_transform(image=img_resized)['image'].unsqueeze(0).to(DEVICE)
#     return img_resized, img_tensor

# @torch.no_grad()
# def get_prediction_with_tta(model, img_tensor):
#     model.eval()
    
    
#     # Average predictions from multiple views
#     acc_probs = torch.zeros(1, 4).to(DEVICE)
#     acc_probs += torch.softmax(model(img_tensor), dim=1)
#     acc_probs += torch.softmax(model(TF.hflip(img_tensor)), dim=1)
#     acc_probs += torch.softmax(model(TF.vflip(img_tensor)), dim=1)
#     acc_probs += torch.softmax(model(TF.rotate(img_tensor, 90)), dim=1)
#     acc_probs /= 4
    
#     probs = acc_probs[0].cpu().numpy()
#     pred_idx = np.argmax(probs)
#     return pred_idx, probs

# def get_gradcam(model, img_tensor, img_resized):
#     target_layer = model.backbone.conv_head
#     cam = GradCAM(model=model, target_layers=[target_layer])
#     grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]
#     visualization = show_cam_on_image(img_resized.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
#     return visualization

# #-----------------------------UI CODE STARTS HERE-----------------------------#

# # Streamlit UI
# st.title("🔬 Multi-Disease Classification & Localization")
# st.markdown("<p style='text-align: center; color: #FFFF; font-size: 1.15rem; max-width: 700px; margin: 0 auto 2rem auto;'>Upload a blood smear image to classify between <b style='color: #EF4444;'>Normal, Sickle Cell, Malaria, and Leukemia</b>, and visualize the infected regions using Grad-CAM.</p>", unsafe_allow_html=True)
# st.markdown("---")

# # Load model
# try:
#     with st.spinner("Loading model weights..."):
#         model = load_model()
# except Exception as e:
#     st.error(f"Failed to load the model. Ensure `{MODEL_PATH}` exists in the directory. Error: {e}")
#     st.stop()

# uploaded_file = st.file_uploader("Choose a blood smear image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Preprocess
#     image_bytes = uploaded_file.read()
#     img_resized, img_tensor = preprocess_image(image_bytes)
    
#     col1, col2 = st.columns([1, 1], gap="large")
    
#     with col1:
#         st.markdown("### Uploaded Image")
#         st.image(img_resized, use_container_width=True, channels="RGB", output_format="PNG")
        
#     # Predict
#     pred_idx, probs = get_prediction_with_tta(model, img_tensor)
#     pred_label = LABELS_NAMES[pred_idx]
#     confidence = probs[pred_idx] * 100
    
#     with col2:
#         st.markdown("### Prediction Results")
#         st.markdown(f"""
#         <div class='prediction-box'>
#             <h2 class='prediction-label'>{pred_label}</h2>
#             <div style='text-align: center;'><p class='prediction-confidence'>Confidence: {confidence:.2f}%</p></div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.write("")
#         st.markdown("#### Class Probabilities")
#         for i, name in enumerate(LABELS_NAMES):
#             st.progress(float(probs[i]), text=f"{name}: {probs[i]*100:.2f}%")
            
#     # Grad-CAM Localization
#     st.markdown("---")
#     st.markdown("### 🦠 Disease Localization (Grad-CAM)")
#     st.markdown(f"The heatmap below highlights the structural regions the model focused on to predict **{pred_label}**.")
    
#     with st.spinner("Generating Grad-CAM visualization..."):
#         try:
#             cam_image = get_gradcam(model, img_tensor, img_resized)
            
#             cam_col1, cam_col2, cam_col3 = st.columns([1, 2, 1])
#             with cam_col2:
#                 st.image(cam_image, caption=f"Grad-CAM Heatmap for {pred_label}", use_container_width=True, channels="RGB")
#         except Exception as e:
#             st.error(f"Failed to generate Grad-CAM: {e}")


import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms.functional as TF
from ultralytics import YOLO
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Multi-Disease Blood Cell Analysis", 
    layout="wide", 
    page_icon="🔬"
)

# ============================================================
# CONSTANTS & PATHS
# ============================================================
# Classification model
CLASSIFIER_LABELS = ['Normal', 'Sickle', 'Malaria', 'Leukemia']
CLASSIFIER_PATH = "model_weights/classifier_best_cv.pth"

# Localization model (YOLOv8)
YOLO_PATH = "model_weights/best.pt"
YOLO_CLASS_NAMES = {
    0: "sickle",
    1: "malaria", 
    2: "leukemia_benign",
    3: "leukemia_malignant"
}
YOLO_COLORS = {
    0: (100, 220, 255),   # sickle       → cyan
    1: (255, 60, 60),      # malaria      → red
    2: (100, 220, 100),   # benign       → green
    3: (255, 200, 30),    # malignant    → yellow
}

# Per-class confidence thresholds (from your training analysis)
PER_CLASS_CONF = {
    0: 0.25,   # sickle
    1: 0.40,   # malaria (raised to reduce false positives)
    2: 0.25,   # leukemia_benign
    3: 0.15,   # leukemia_malignant (lowered to improve recall)
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Background and Layout */
    .stApp {
        background-color: #0a0a0f;
        background-image: linear-gradient(180deg, #0a0a0f 0%, #111827 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Typography */
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3, h4 {
        color: #93c5fd;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    p, .stMarkdown {
        color: #cbd5e1;
    }
    
    /* Upload Box */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(17, 24, 39, 0.8);
        border: 2px dashed #3b82f6;
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease-in-out;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #60a5fa;
        background-color: rgba(30, 58, 138, 0.3);
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    /* Prediction Box */
    .prediction-box {
        background: #1e293b;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        border-top: 4px solid #3b82f6;
    }
    .prediction-label {
        text-align: center;
        margin: 0 0 0.5rem 0;
        color: #93c5fd;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .prediction-confidence {
        text-align: center;
        margin: 0;
        font-size: 1.2rem;
        color: #EF4444;
        font-weight: 600;
        background: #1e3a8a;
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
    }
    
    /* Detection Box */
    .detection-box {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        margin-bottom: 1rem;
        border-left: 4px solid #10b981;
    }
    
    /* Images */
    [data-testid="stImage"] img {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    hr {
        border-top: 1px solid #334155;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CLASSIFICATION MODEL
# ============================================================
class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=4, backbone='efficientnet_b0', pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load the classification model"""
    if not os.path.exists(CLASSIFIER_PATH):
        st.error(f"❌ Classification model not found at: {CLASSIFIER_PATH}")
        st.stop()
    
    model = DiseaseClassifier(num_classes=4, pretrained=False)
    state_dict = torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ============================================================
# LOCALIZATION MODEL (YOLOv8)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """Load the YOLOv8 localization model"""
    if not os.path.exists(YOLO_PATH):
        st.error(f"❌ YOLOv8 model not found at: {YOLO_PATH}")
        st.stop()
    
    model = YOLO(YOLO_PATH)
    model.to(DEVICE)
    return model

# ============================================================
# CLASSIFICATION FUNCTIONS
# ============================================================
inference_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_for_classification(image_bytes):
    """Preprocess image for classification model"""
    img_np = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_tensor = inference_transform(image=img_resized)['image'].unsqueeze(0).to(DEVICE)
    return img_resized, img_tensor

@torch.no_grad()
def get_classification_with_tta(model, img_tensor):
    """Classification with Test-Time Augmentation"""
    model.eval()
    acc_probs = torch.zeros(1, 4).to(DEVICE)
    acc_probs += torch.softmax(model(img_tensor), dim=1)
    acc_probs += torch.softmax(model(TF.hflip(img_tensor)), dim=1)
    acc_probs += torch.softmax(model(TF.vflip(img_tensor)), dim=1)
    acc_probs += torch.softmax(model(TF.rotate(img_tensor, 90)), dim=1)
    acc_probs /= 4
    probs = acc_probs[0].cpu().numpy()
    pred_idx = np.argmax(probs)
    return pred_idx, probs

def get_gradcam(model, img_tensor, img_resized):
    """Generate Grad-CAM visualization"""
    target_layer = model.backbone.conv_head
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]
    visualization = show_cam_on_image(
        img_resized.astype(np.float32) / 255.0, 
        grayscale_cam, 
        use_rgb=True
    )
    return visualization

# ============================================================
# LOCALIZATION FUNCTIONS (YOLOv8)
# ============================================================
def run_localization(yolo_model, image_bytes, per_class_conf):
    """
    Run YOLOv8 detection with per-class confidence thresholds.
    
    Returns:
        original_img: PIL Image (original)
        annotated_img: PIL Image (with bounding boxes)
        detections: list of dicts with detection info
    """
    # Decode image
    img_np = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Run detection at very low threshold to get all candidates
    results = yolo_model.predict(
        img_rgb,
        conf=0.10,      # Low global threshold
        iou=0.45,
        imgsz=640,
        verbose=False
    )[0]
    
    # Filter by per-class thresholds
    detections = []
    annotated_img = img_rgb.copy()
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Apply per-class threshold
        min_conf = per_class_conf.get(cls_id, 0.25)
        if conf < min_conf:
            continue
        
        # Extract box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Get class info
        class_name = YOLO_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color = YOLO_COLORS.get(cls_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated_img, 
            (x1, y1 - label_h - 10), 
            (x1 + label_w + 10, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_img, 
            label, 
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1, 
            cv2.LINE_AA
        )
        
        detections.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2]
        })
    
    return Image.fromarray(img_rgb), Image.fromarray(annotated_img), detections

# ============================================================
# MAIN UI
# ============================================================
st.title("🔬 Multi-Disease Blood Cell Analysis")
st.markdown("""
<p style='text-align: center; color: #cbd5e1; font-size: 1.15rem; max-width: 800px; margin: 0 auto 2rem auto;'>
Upload a blood smear image for <b style='color: #3b82f6;'>classification</b> 
or <b style='color: #10b981;'>parasite localization</b> with bounding box detection.
</p>
""", unsafe_allow_html=True)

# ============================================================
# MODE SELECTION
# ============================================================
st.markdown("### Select Analysis Mode")
analysis_mode = st.radio(
    "Choose analysis type:",
    options=["🧬 Classification (with Grad-CAM)", "🎯 Localization (with Bounding Boxes)"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader(
    "Choose a blood smear image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    
    # ========================================================
    # MODE 1: CLASSIFICATION
    # ========================================================
    if "Classification" in analysis_mode:
        st.markdown("## 🧬 Classification Analysis")
        
        with st.spinner("Loading classification model..."):
            classifier = load_classifier()
        
        # Preprocess
        img_resized, img_tensor = preprocess_for_classification(image_bytes)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### Uploaded Image")
            st.image(img_resized, use_container_width=True, channels="RGB")
        
        # Predict
        with st.spinner("Running classification..."):
            pred_idx, probs = get_classification_with_tta(classifier, img_tensor)
        
        pred_label = CLASSIFIER_LABELS[pred_idx]
        confidence = probs[pred_idx] * 100
        
        with col2:
            st.markdown("### Prediction Results")
            st.markdown(f"""
            <div class='prediction-box'>
                <h2 class='prediction-label'>{pred_label}</h2>
                <div style='text-align: center;'>
                    <p class='prediction-confidence'>Confidence: {confidence:.2f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("#### Class Probabilities")
            for i, name in enumerate(CLASSIFIER_LABELS):
                st.progress(float(probs[i]), text=f"{name}: {probs[i]*100:.2f}%")
        
        # Grad-CAM
        st.markdown("---")
        st.markdown("### 🔥 Disease Localization (Grad-CAM)")
        st.markdown(f"Heatmap highlights regions the model focused on to predict **{pred_label}**.")
        
        with st.spinner("Generating Grad-CAM..."):
            try:
                cam_image = get_gradcam(classifier, img_tensor, img_resized)
                cam_col1, cam_col2, cam_col3 = st.columns([1, 2, 1])
                with cam_col2:
                    st.image(
                        cam_image, 
                        caption=f"Grad-CAM for {pred_label}", 
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Grad-CAM generation failed: {e}")
    
    # ========================================================
    # MODE 2: LOCALIZATION (YOLOv8)
    # ========================================================
    else:
        st.markdown("## 🎯 Parasite Localization")
        
        with st.spinner("Loading YOLOv8 detection model..."):
            yolo_model = load_yolo_model()
        
        # Run detection
        with st.spinner("Detecting parasites and abnormal cells..."):
            original_img, annotated_img, detections = run_localization(
                yolo_model, 
                image_bytes, 
                PER_CLASS_CONF
            )
        
        # Display results
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### Original Image")
            st.image(original_img, use_container_width=True)
        
        with col2:
            st.markdown("### Detected Objects")
            st.image(annotated_img, use_container_width=True)
        
        # Detection summary
        st.markdown("---")
        st.markdown("### 📋 Detection Summary")
        
        if detections:
            st.markdown(f"""
            <div class='detection-box'>
                <h3 style='margin:0 0 1rem 0; color: #10b981;'>
                    ✅ Found {len(detections)} object(s)
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Group by class
            from collections import Counter
            class_counts = Counter([d["class_name"] for d in detections])
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### Detected Classes")
                for class_name, count in class_counts.items():
                    st.markdown(f"- **{class_name}**: {count} instance(s)")
            
            with col_b:
                st.markdown("#### Detection Details")
                for i, det in enumerate(detections, 1):
                    st.markdown(
                        f"**{i}.** {det['class_name']} "
                        f"(conf: {det['confidence']:.3f})"
                    )
            
            # Detailed table
            with st.expander("📊 View Full Detection Table"):
                import pandas as pd
                df = pd.DataFrame(detections)
                df = df[["class_name", "confidence", "bbox"]]
                df.columns = ["Class", "Confidence", "Bounding Box [x1,y1,x2,y2]"]
                st.dataframe(df, use_container_width=True)
        
        else:
            st.markdown("""
            <div class='detection-box' style='border-left-color: #ef4444;'>
                <h3 style='margin:0; color: #ef4444;'>
                    ℹ️ No objects detected above confidence threshold
                </h3>
                <p style='margin: 0.5rem 0 0 0; color: #94a3b8;'>
                    Try a different image or adjust detection thresholds.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Legend
        st.markdown("---")
        st.markdown("### 🎨 Bounding Box Color Legend")
        legend_cols = st.columns(4)
        for i, (cls_id, color) in enumerate(YOLO_COLORS.items()):
            with legend_cols[i]:
                color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem; 
                            background: {color_hex}; border-radius: 8px; 
                            color: #000; font-weight: 600;'>
                    {YOLO_CLASS_NAMES[cls_id]}
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>
Built by Group 3 CSE(AI)
</p>
""", unsafe_allow_html=True)
