from flask import Flask, request, jsonify, render_template
import cv2
import base64
import numpy as np
from reg2 import FaceRecognitionSystem

app = Flask(__name__)
face_system = FaceRecognitionSystem()

def decode_image(image_data):
    """Decode base64 image to OpenCV format."""
    encoded_data = image_data.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(image):
    """Encode OpenCV image to base64 format for HTML display."""
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Lấy dữ liệu hình ảnh từ request
        image_data = request.json.get('image')
        frame = decode_image(image_data)

        # Nhận diện khuôn mặt
        processed_frame = face_system.detect_and_recognize(frame)
        result_image = encode_image(processed_frame)

        return jsonify({'status': 'success', 'image': result_image})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
