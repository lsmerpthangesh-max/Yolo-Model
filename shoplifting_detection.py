from ultralytics import YOLO
import numpy as np
import imutils
import cv2
import streamlit as st
import os
import time

# import parameters
from config.parameters import WIDTH, start_status, shoplifting_status, not_shoplifting_status
from config.parameters import cls0_rect_color, cls1_rect_color, conf_color, status_color

# -----------------------------
# ✅ SAFE PATH HANDLING
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mymodel=YOLO("configs/shoplifting_wights.pt")
video_path = os.path.join(BASE_DIR, "res", "inout1.mp4")

# Debug check
st.write("Model path:", model_path)
st.write("Model exists:", os.path.exists(model_path))

if not os.path.exists(model_path):
    st.error("❌ Model file NOT FOUND. Check configs folder.")
    st.stop()

# -----------------------------
# ✅ LOAD MODEL
# -----------------------------
model = YOLO(model_path)

# -----------------------------
# ✅ LOAD VIDEO
# -----------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("❌ Video not found or cannot open")
    st.stop()

# -----------------------------
# ✅ STREAMLIT UI
# -----------------------------
st.title("🛒 Shoplifting Detection System")
frame_placeholder = st.empty()

# -----------------------------
# ✅ PERFORMANCE SETTINGS
# -----------------------------
frame_count = 0

# -----------------------------
# ✅ MAIN LOOP
# -----------------------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 SKIP FRAMES (IMPORTANT FOR SMOOTHNESS)
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    # ⚡ RESIZE (FASTER)
    frame = imutils.resize(frame, width=480)

    # 🤖 YOLO INFERENCE
    results = model(frame, verbose=False)

    status = start_status

    if len(results[0].boxes) > 0:
        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), conf, clas in zip(xyxy, confs, classes):

            w = x2 - x1
            h = y2 - y1

            if clas == 1:
                # 🔴 SHOPLIFTING
                cv2.rectangle(frame, (x1, y1), (x2, y2), cls1_rect_color, 2)

                center_x = int(x1 + w / 2)
                cv2.circle(frame, (center_x, y1), 6, (0, 0, 255), 6)

                text = f"{conf*100:.2f}%"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)

                status = shoplifting_status

            elif clas == 0 and conf > 0.5:
                # 🟢 NORMAL
                cv2.rectangle(frame, (x1, y1), (x2, y2), cls0_rect_color, 1)

                text = f"{conf*100:.2f}%"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)

                status = not_shoplifting_status

    # 📊 STATUS TEXT
    cv2.putText(frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # 🎨 Convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🖥️ Display in Streamlit
    frame_placeholder.image(frame, channels="RGB")

    # ⏱️ CONTROL PLAYBACK SPEED (IMPORTANT)
    time.sleep(0.03)

cap.release()
