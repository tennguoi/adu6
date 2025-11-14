def calculate_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Độ dài của y_true và y_pred phải giống nhau.")
    
    TP = FP = TN = FN = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1:
            if predicted == 1:
                TP += 1
            else:
                FN += 1
        else:
            if predicted == 1:
                FP += 1
            else:
                TN += 1
    
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total != 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

# Nhập dữ liệu từ người dùng
y_true = eval(input("Nhập danh sách nhãn thực tế (ví dụ: [1, 0, 1, 0]): "))
y_pred = eval(input("Nhập danh sách nhãn dự đoán (ví dụ: [1, 1, 0, 0]): "))

# Tính toán và hiển thị kết quả
if len(y_true) != len(y_pred):
    print("Lỗi: Danh sách nhãn thực tế và dự đoán phải có cùng độ dài.")
else:
    metrics = calculate_metrics(y_true, y_pred)
    print("\nCác độ đo hiệu suất:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1 Score']:.4f}")