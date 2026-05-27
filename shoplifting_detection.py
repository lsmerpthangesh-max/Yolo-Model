from ultralytics import YOLO
import numpy as np
import imutils
import cv2
import streamlit as st

# import parameters
from config.parameters import WIDTH, start_status, shoplifting_status, not_shoplifting_status
from config.parameters import cls0_rect_color, cls1_rect_color, conf_color, status_color

input_path = "res/inout1.mp4"

# ✅ FIXED MODEL PATH (IMPORTANT)
mymodel=YOLO("configs/shoplifting_wights.pt")

cap = cv2.VideoCapture(input_path)

# ✅ Streamlit UI
st.title("Shoplifting Detection System")
frame_placeholder = st.empty()

# ✅ ADD FRAME SKIP (VERY IMPORTANT)
frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # ✅ SKIP FRAMES (reduces lag)
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    # ✅ RESIZE SMALLER (faster)
    frame = imutils.resize(frame, width=480)

    # ✅ FAST YOLO INFERENCE
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

    # ✅ STATUS TEXT
    cv2.putText(frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # ✅ Convert to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ Display frame
    frame_placeholder.image(frame, channels="RGB")

cap.release()
