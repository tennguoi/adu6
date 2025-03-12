import torch
from PIL import Image
import cv2
import numpy as np
import json
import os
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class FaceRecognitionSystem:
    def __init__(self):
        # MTCNN và FaceNet initialization
        self.mtcnn = MTCNN(
            margin=10,
            keep_all=True,
            post_process=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet = self.facenet.to(self.device)
        self.known_embeddings = {}
        self.names = {}
        self.load_known_faces()

    def load_known_faces(self):
        # Tải embeddings và tên
        if os.path.exists('face_embeddings.npz'):
            data = np.load('face_embeddings.npz', allow_pickle=True)
            self.known_embeddings = dict(data['embeddings'][()])
        else:
            print("Tệp 'face_embeddings.npz' không tồn tại. Khởi tạo danh sách rỗng.")
        
        if os.path.exists('names.json'):
            with open('names.json', 'r') as f:
                self.names = json.load(f)
        else:
            print("Tệp 'names.json' không tồn tại. Khởi tạo danh sách tên rỗng.")

    def normalize_embedding(self, embedding):
        # Chuẩn hóa embedding trước khi so sánh
        return embedding / np.linalg.norm(embedding)

    def get_face_embedding(self, face_img):
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        face_tensor = transform(Image.fromarray(face_img)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.facenet(face_tensor)
        return embedding.cpu().numpy()

    def recognize_face(self, face_embedding, cosine_threshold=0.4, euclidean_threshold=0.9, ensemble_weight=0.5):
        if not self.known_embeddings:
            return "Unknown", 0

        # Chuẩn hóa embedding đầu vào
        normalized_input = self.normalize_embedding(face_embedding)

        best_score = 0
        best_name = "Unknown"
        
        for user_id, stored_embedding in self.known_embeddings.items():
            # Chuẩn hóa embedding được lưu trữ
            normalized_stored = self.normalize_embedding(stored_embedding)

            # Tính toán Cosine Similarity
            cosine_score = cosine_similarity(normalized_input.reshape(1, -1), 
                                            normalized_stored.reshape(1, -1))[0][0]
            
            # Tính toán Euclidean Distance (càng thấp càng tốt)
            euclidean_distance = euclidean_distances(normalized_input.reshape(1, -1), 
                                                    normalized_stored.reshape(1, -1))[0][0]
            
            # Kết hợp điểm số (ensemble)
            ensemble_score = (ensemble_weight * cosine_score + 
                              (1 - ensemble_weight) * (1 / (1 + euclidean_distance)))

            if ensemble_score > best_score:
                best_score = ensemble_score
                best_name = self.names.get(str(user_id), "Unknown")

        # Kiểm tra ngưỡng với điểm số ensemble
        if best_score > cosine_threshold:
            return best_name, best_score * 100
        
        # Kiểm tra lại nếu không chắc chắn
        if best_score > 0.2:  # Ngưỡng thấp hơn để kiểm tra
            return f"Possible {best_name}", best_score * 100
        
        return "Unknown", best_score * 100

    def detect_and_recognize(self, frame):
        # Phát hiện khuôn mặt
        faces = self.mtcnn(frame)

        # Nếu không có khuôn mặt
        if faces is None:
            return frame

        # Lấy vị trí các khuôn mặt
        boxes, _ = self.mtcnn.detect(frame)

        # Xử lý từng khuôn mặt
        for box in boxes:
            bbox = list(map(int, box.tolist()))
            face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Lấy embedding và nhận dạng
            embedding = self.get_face_embedding(face_img)
            name, confidence = self.recognize_face(embedding[0])

            # Vẽ kết quả
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.1f}%)",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


def main():
    face_system = FaceRecognitionSystem()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = face_system.detect_and_recognize(frame)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()