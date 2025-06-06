import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd

# --- Config ---
MODEL_PATH = 'deepglobe_unet_jay.keras'  # Your model file
CLASS_DICT_CSV = 'class_dict.csv'
IMG_SIZE = (224, 224)

# --- Load class info ---
@st.cache_data
def load_class_colors():
    df = pd.read_csv(CLASS_DICT_CSV)
    return df[['r', 'g', 'b']].values.astype(np.uint8)

class_colors = load_class_colors()
NUM_CLASSES = len(class_colors)

# --- Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# --- Preprocessing ---
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img.astype(np.float32)

# --- Decode segmentation mask ---
def decode_mask(mask):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(class_colors):
        rgb_mask[mask == i] = color
    return rgb_mask

# --- Calculate metrics (accuracy, mIoU) ---
def calculate_metrics(pred, true_mask, num_classes):
    pred_flat = pred.flatten()
    true_flat = true_mask.flatten()

    acc = np.mean(pred_flat == true_flat)

    iou_scores = []
    for i in range(num_classes):
        intersection = np.logical_and(pred_flat == i, true_flat == i).sum()
        union = np.logical_or(pred_flat == i, true_flat == i).sum()
        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union
        iou_scores.append(iou)

    mean_iou = np.nanmean(iou_scores)
    return acc, mean_iou

# --- Streamlit UI ---
st.title("🌍 DeepGlobe Land Cover Segmentation")
st.write("Upload a satellite image to see segmented land cover types.")

uploaded_file = st.file_uploader("📷 Upload Satellite Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("🖼️ Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    img_input = preprocess(img)
    img_input_exp = np.expand_dims(img_input, axis=0)

    preds = model.predict(img_input_exp)
    pred_mask = np.argmax(preds[0], axis=-1)

    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_mask = decode_mask(pred_mask_resized)

    st.subheader("🎯 Predicted Segmentation Mask")
    st.image(color_mask, use_container_width=True)

    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.6, color_mask, 0.4, 0)
    st.subheader("🌀 Overlay on Original")
    st.image(overlay, use_container_width=True)

    # Optional: Ground truth mask upload
    st.subheader("📊 Upload Ground Truth Mask (Optional for Accuracy)")
    gt_file = st.file_uploader("Upload ground truth mask (grayscale, class labels)", type=['png', 'jpg'])

    if gt_file is not None:
        gt_bytes = np.asarray(bytearray(gt_file.read()), dtype=np.uint8)
        gt_img = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)
        gt_img_resized = cv2.resize(gt_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        acc, miou = calculate_metrics(pred_mask_resized, gt_img_resized, NUM_CLASSES)
        st.subheader("📈 Model Performance")
        st.write(f"✅ **Pixel Accuracy**: `{acc * 100:.2f}%`")
        st.write(f"✅ **Mean IoU**: `{miou * 100:.2f}%`")

# --- Class Legend with actual colors ---
st.subheader("📘 Class Color Legend")

color_map = {
    0: (0, 255, 255),     # Urban land - Cyan
    1: (255, 255, 0),     # Agriculture land - Yellow
    2: (255, 0, 255),     # Rangeland - Magenta
    3: (0, 255, 0),       # Forest land - Green
    4: (0, 0, 255),       # Water - Blue
    5: (255, 255, 255),   # Barren land - White
    6: (0, 0, 0),         # Unknown - Black
}

class_names = [
    "Urban land",
    "Agriculture land",
    "Rangeland",
    "Forest land",
    "Water",
    "Barren land",
    "Unknown"
]

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

legend_html = "<table>"
legend_html += "<tr><th>Class</th><th>Color</th></tr>"
for idx, name in enumerate(class_names):
    hex_color = rgb_to_hex(color_map[idx])
    legend_html += f"<tr><td>{name}</td><td style='background-color:{hex_color}; width:100px;'>&nbsp;</td></tr>"
legend_html += "</table>"

st.markdown(legend_html, unsafe_allow_html=True)
