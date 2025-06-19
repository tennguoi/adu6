import torch
import numpy as np
from PIL import Image
import os

from facenet_pytorch import InceptionResnetV1
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt  # Thêm import matplotlib

def create_augmentation_pipeline() -> A.Compose:
    return A.Compose([
        A.Resize(160, 160),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-20, 20), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

def load_and_preprocess_images(image_paths, transform, batch_size=32, device='cpu'):
    """Load and preprocess images in batches using Albumentations"""
    image_tensors = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            augmented = transform(image=img)
            img_tensor = augmented['image'].unsqueeze(0)
            image_tensors.append(img_tensor)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    if len(image_tensors) == 0:
        return [], []

    image_tensors = torch.cat(image_tensors).to(device)
    return image_tensors, valid_paths

def filter_embeddings_with_pca(user_embeddings, n_components=3, eps=0.5, min_samples=3):
    """Filter embeddings using PCA and DBSCAN"""
    if len(user_embeddings) < min_samples:
        return np.mean(user_embeddings, axis=0)
        
    # Apply PCA for dimension reduction
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(user_embeddings)
    
    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_data)
    
    # Get core samples mask
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    # Filter outliers
    valid_embeddings = user_embeddings[core_samples_mask]
    
    if len(valid_embeddings) == 0:
        return np.mean(user_embeddings, axis=0)
    
    return np.mean(valid_embeddings, axis=0)

def normalize_embeddings(embeddings_dict):
    """Normalize embeddings after training"""
    # Convert dictionary values to array
    user_ids = list(embeddings_dict.keys())
    embeddings_array = np.array([embeddings_dict[uid] for uid in user_ids])
    
    # Normalize embeddings
    normalized_embeddings = normalize(embeddings_array, norm='l2', axis=1)
    
    # Convert back to dictionary
    return {uid: emb for uid, emb in zip(user_ids, normalized_embeddings)}

def main():
    print("\n[INFO] Starting optimized face embedding generation...")

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(Device)

    images_path = './Images/'
    batch_size = 32
    n_augmentations = 5

    image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) 
                   if f.endswith('.jpg')]
    print(f"[INFO] Found {len(image_files)} images")

    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    embeddings_dict = {}
    temp_embeddings = {}
    
    # Biến để theo dõi số embeddings mỗi người dùng
    user_embedding_counts = {}

    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i + batch_size]
        
        image_tensors = []
        valid_paths = []
        
        for img_path in batch_files:
            try:
                img = np.array(Image.open(img_path).convert('RGB'))
                # Apply multiple augmentations
                for _ in range(n_augmentations):
                    augmented = transform(image=img)
                    image_tensors.append(augmented['image'].unsqueeze(0))
                    valid_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        if len(image_tensors) == 0:
            continue

        image_tensors = torch.cat(image_tensors).to(Device)

        with torch.no_grad():
            embeddings = facenet(image_tensors).cpu().numpy()

        for idx, img_path in enumerate(valid_paths):
            user_id = int(os.path.split(img_path)[-1].split('-')[1])
            if user_id not in temp_embeddings:
                temp_embeddings[user_id] = []
                user_embedding_counts[user_id] = 0
            temp_embeddings[user_id].append(embeddings[idx])
            user_embedding_counts[user_id] += 1

    # Process embeddings for each user
    for user_id, user_embeddings in temp_embeddings.items():
        # Filter embeddings using PCA and DBSCAN
        filtered_embedding = filter_embeddings_with_pca(np.array(user_embeddings))
        embeddings_dict[user_id] = filtered_embedding

    # Normalize final embeddings
    embeddings_dict = normalize_embeddings(embeddings_dict)

    # Save embeddings in NPZ format
    np.savez_compressed('face_embeddings.npz', embeddings=embeddings_dict)
    
    print(f"\n[INFO] Successfully generated embeddings for {len(embeddings_dict)} users")
    print("[INFO] Saved embeddings to 'face_embeddings.npz'")
    data = np.load('face_embeddings.npz', allow_pickle=True)
    embeddings = data['embeddings'].item()  # 'item()' vì file lưu dạng dictionary

    # Xem danh sách user_id
    print("User IDs:", embeddings.keys())

    # Kiểm tra embedding của một user cụ thể
    # (sử dụng user_id cuối cùng từ vòng lặp, hoặc thay đổi theo ý bạn)
    print(f"Embedding for User {user_id}:", embeddings[user_id])

    # Kiểm tra kích thước vector
    print(f"Shape of Embedding for User {user_id}: {embeddings[user_id].shape}")

    # ====== PHẦN HIỂN THỊ KẾT QUẢ VỚI MATPLOTLIB ======
    
    # 1. Biểu đồ số lượng embeddings cho mỗi người dùng
    plt.figure(figsize=(10, 6))
    user_ids = list(user_embedding_counts.keys())
    user_ids.sort()  # Sắp xếp ID người dùng để biểu đồ dễ đọc hơn
    
    counts = [user_embedding_counts[uid] for uid in user_ids]
    
    plt.bar(range(len(user_ids)), counts, color='skyblue', edgecolor='navy')
    plt.xticks(range(len(user_ids)), [str(uid) for uid in user_ids], rotation=90 if len(user_ids) > 10 else 0)
    plt.xlabel('User ID')
    plt.ylabel('Số lượng Embeddings')
    plt.title('Số lượng Embeddings mỗi người dùng')
    plt.tight_layout()
    plt.savefig('embedding_counts.png')
    plt.show()
    
    # 2. Biểu đồ phân bố số lượng embeddings
    plt.figure(figsize=(8, 6))
    plt.hist(counts, bins=10, color='lightgreen', edgecolor='darkgreen')
    plt.xlabel('Số lượng Embeddings')
    plt.ylabel('Số lượng người dùng')
    plt.title('Phân bố số lượng Embeddings')
    plt.grid(True, alpha=0.3)
    plt.savefig('embedding_distribution.png')
    plt.show()
    
    # 3. Biểu đồ sự khác biệt giữa các embeddings (PCA)
    user_ids = list(embeddings.keys())
    embedding_list = np.array([embeddings[uid] for uid in user_ids])

    # Sử dụng PCA để giảm các embeddings xuống 2 chiều để có thể plot
    pca_plot = PCA(n_components=2)
    embedding_2d = pca_plot.fit_transform(embedding_list)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='blue', edgecolors='k', s=100, alpha=0.7)

    # Gán nhãn cho từng điểm
    for i, uid in enumerate(user_ids):
        plt.annotate(str(uid), (embedding_2d[i, 0], embedding_2d[i, 1]), 
                     fontsize=9, fontweight='bold')

    plt.title("Face Embeddings PCA Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.savefig('embedding_pca.png')
    plt.show()
    
    # 4. Biểu đồ heatmap của ma trận tương đồng (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embedding_list)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Similarity Matrix of Face Embeddings')
    
    # Thêm nhãn cho các trục
    tick_positions = np.arange(len(user_ids))
    plt.xticks(tick_positions, [str(uid) for uid in user_ids], rotation=90)
    plt.yticks(tick_positions, [str(uid) for uid in user_ids])
    
    # Thêm chú thích giá trị cho từng ô
    for i in range(len(user_ids)):
        for j in range(len(user_ids)):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                    ha='center', va='center', color='white' if similarity_matrix[i, j] < 0.7 else 'black',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig('embedding_similarity.png')
    plt.show()

if __name__ == "__main__":
    main()