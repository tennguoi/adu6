import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("face_recognition_model.h5")
img = cv2.imread("datatest/Minh/face_minh.jpg")  # Thay đường dẫn ảnh của bạn
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print("Prediction:", prediction)
