import torch
import cv2
import os
import json
from PIL import Image
from facenet_pytorch import MTCNN

def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_id(directory: str) -> int:
    user_ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            number = int(os.path.split(filename)[-1].split("-")[1])
            user_ids.append(number)
    user_ids = sorted(list(set(user_ids)))
    max_user_ids = 1 if len(user_ids) == 0 else max(user_ids) + 1
    for i in sorted(range(0, max_user_ids)):
        try:
            if user_ids.index(i):                                 
                face_id = i
        except ValueError as e:
            return i
    return max_user_ids

def save_name(face_id: int, face_name: str, filename: str) -> None:
    names_json = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            names_json = json.load(fs)
    names_json[str(face_id)] = face_name
    with open(filename, 'w') as fs:
        json.dump(names_json, ensure_ascii=False, indent=4, fp=fs)

def main():
    directory = 'images'
    names_json_filename = 'names.json'
    create_directory(directory)
    
    mtcnn = MTCNN(margin=20, keep_all=True, 
                device='cuda' if torch.cuda.is_available() else 'cpu')
    
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    face_name = input('\nEnter user name and press <return> -->  ')
    face_id = get_face_id(directory)
    save_name(face_id, face_name, names_json_filename)
    print('\n[INFO] Initializing face capture. Look at the camera and wait...')

    count = 0 #Số lần tìm thấy khuôn mặt
    valid_count = 0 #Số ảnh hợp lệ 

    while True: #Vòng lặp đọc dữ liệu liên tục 
        ret, frame = cam.read()
        if not ret:
            continue

        # Phát hiện khuôn mặt bằng MTCNN
        boxes, _ = mtcnn.detect(frame)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.tolist())#Chuyển đổi tọa độ khuôn mặt 
                
                # Thêm margin
                margin = int(min(x2-x1, y2-y1) * 0.1)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)

                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                # Kiểm tra kích thước khuôn mặt
                face_area = (x2-x1) * (y2-y1)
                frame_area = frame.shape[0] * frame.shape[1]

                if face_area / frame_area > 0.05:  # Khuôn mặt đủ lớn
                    save_path = f'{directory}/Users-{face_id}-{count}.jpg'
                    cv2.imwrite(save_path, face_img)
                    count += 1
                    valid_count += 1
                    print(f"[INFO] Saved image {valid_count}/100: {save_path}")

                    # Vẽ khung và thông tin
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Capture: {valid_count}/30",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

        cv2.imshow('Face Capture', frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:  # ESC key
            break
        elif valid_count >= 100:  # Đủ số lượng ảnh
            break

    print('\n[INFO] Face capture completed successfully!')
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()