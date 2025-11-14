import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

# Tải model
model = load_model("face_recognition_model_v2.h5")

# Tải cấu hình nhận dạng
with open("class_indices.json", 'r') as f:
    class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
CONFIDENCE_THRESHOLD = 0.855

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # Mở camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang ảnh xám để phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Cắt vùng khuôn mặt
        face_img = frame[y:y+h, x:x+w]

        try:
            # Resize và chuẩn hóa ảnh để phù hợp với input của model
            face_img = cv2.resize(face_img, (128, 128))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            # Dự đoán
            predictions = model.predict(face_img, verbose=0)[0]
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions)

            # Xử lý người lạ dựa trên độ tin cậy
        if confidence < CONFIDENCE_THRESHOLD:
            label = "Unknown"
                color = (0, 0, 255)  # Đỏ cho người lạ
        else:
                # Sử dụng try/except để tránh KeyError
                try:
            label = labels[predicted_class]
                    color = (0, 255, 0)  # Xanh cho người quen
                except KeyError:
                    label = "Unknown"
                    color = (0, 0, 255)  # Đỏ cho người lạ

            # Hiển thị tên và độ tin cậy
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        except Exception as e:
            print(f"Error processing face: {str(e)}")
    # Hiển thị frame
    cv2.imshow('Face Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

