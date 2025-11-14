import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ======================
# 1. Kiểm tra dữ liệu
# ======================
data_dir = "/content/drive/MyDrive/datatest"
classes = os.listdir(data_dir)
print("\n=== Phân phối dữ liệu ===")
class_dist = {}
for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    num_images = len(os.listdir(cls_path))
    class_dist[cls] = num_images
    print(f"{cls}: {num_images} ảnh")

# ======================
# 2. Thiết lập dữ liệu
# ======================
img_size = 128
batch_size = 32

# Data augmentation chỉnh lại
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Tạo data generators với .repeat()
train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ======================
# 3. Tính class weights
# ======================
train_labels = train_gen.classes
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))
print("\n=== Class Weights ===")
for cls, weight in class_weights.items():
    print(f"Class {classes[cls]}: {weight:.2f}")

# ======================
# 4. Xây dựng mô hình
# ======================
def create_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False  # Đóng băng layer pretrained

    model = Sequential([
        base_model,
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    return model

model = create_model(len(classes))
model.summary()

# ======================
# 5. Compile mô hình
# ======================
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# 6. Callbacks
# ======================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        mode='max',
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
]

# ======================
# 7. Huấn luyện
# ======================
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    validation_data=val_gen,
    validation_steps=val_gen.samples // batch_size,
    epochs=100,
    callbacks=callbacks,
    class_weight=class_weights
)

# ======================
# 8. Lưu mô hình
# ======================
model.save("face_recognition_model_v2.h5")
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# ======================
# 9. Visualize kết quả
# ======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Evolution')
plt.legend()

plt.savefig('training_plot_v2.png')
plt.show()

# ======================
# 10. Đánh giá
# ======================
val_loss, val_acc = model.evaluate(val_gen, steps=val_gen.samples//batch_size)
print(f"\nFinal Validation Accuracy: {val_acc:.2f}")
print(f"Final Validation Loss: {val_loss:.2f}")