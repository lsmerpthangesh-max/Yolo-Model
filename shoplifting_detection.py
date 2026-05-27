import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

st.title("🛒 Shoplifting Detection System")

# -----------------------------
# Load Model (FIXED PATH)
# -----------------------------
MODEL_PATH = "configs/shoplifting_weights.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

model = YOLO(MODEL_PATH)

# -----------------------------
# Upload Video
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save temp video
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_video_path)

    if not cap.isOpened():
        st.error("❌ Cannot open video")
        st.stop()

    stframe = st.empty()

    # -----------------------------
    # Process Video Frame-by-Frame
    # -----------------------------
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # YOLO prediction
        results = model(frame)

        # Draw results
        annotated_frame = results[0].plot()

        # Convert BGR → RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display in Streamlit
        stframe.image(annotated_frame, channels="RGB")

    cap.release()
    st.success("✅ Video processing completed")

else:
    st.info("⬆️ Please upload a video to start detection")
