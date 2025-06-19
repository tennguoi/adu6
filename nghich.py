from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    "./images/",  # Đảm bảo đường dẫn đúng
    target_size=(100, 100),
    color_mode="grayscale",
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

labels = {v: k for k, v in train_generator.class_indices.items()}  # Lệnh này chỉ thực hiện sau khi train_generator đã được tạo
print(labels)  # Kiểm tra kết quả
