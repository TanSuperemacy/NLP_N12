import os
import re
import sys
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "1_clapAI_UIT-VSMEC_csv")

# 1. BẢNG ÁNH XẠ TEENCODE & CHUẨN HÓA TỪ VIẾT TẮT
TEENCODE_DICT = {
    'ko': 'không', 'k': 'không', 'khg': 'không', 'hông': 'không', 'hổng': 'không',
    'dc': 'được', 'đc': 'được', 'dk': 'được', 'đk': 'được',
    'j': 'gì', 'wa': 'quá', 'wá': 'quá', 'waa': 'quá',
    'bik': 'biết', 'bít': 'biết', 'vs': 'với', 'cx': 'cũng',
    'r': 'rồi', 'rùi': 'rồi', 'thui': 'thôi', 'chừ': 'giờ',
    'ae': 'anh em', 'mn': 'mọi người', 'cmt': 'bình luận',
    'sgk': 'sách giáo khoa', 'huhu': 'huhu emoji_buon', 'haha': 'haha emoji_vui',
    'ntn': 'như thế nào', 'nv': 'như vậy', 'ib': 'inbox', 'fb': 'facebook',
    'ad': 'admin', 'cty': 'công ty', 'stt': 'status', 'nyc': 'người yêu cũ'
}

def preprocess_text(text):
    """
    Hàm tiền xử lý văn bản chuyên sâu cho tiếng Việt mạng xã hội (UIT-VSMEC)
    1. Chuẩn hóa Unicode NFC
    2. Xóa URL
    3. Ánh xạ Emoticon & Emoji thành các token cảm xúc đặc trưng
    4. Chuẩn hóa Teencode
    5. Giảm thiểu ký tự lặp kéo dài (quáaaa -> quá)
    6. Tách từ tiếng Việt bằng PyVi
    """
    text = str(text)
    # 1. Chuẩn hóa Unicode NFC
    text = unicodedata.normalize('NFC', text)
    
    # 2. Xóa URL
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # 3. Ánh xạ Emoticon
    text = re.sub(r':\)+|:\-\)+|\^_\^|\^\^|:D|:d|xD|XD|;\)|;-\)', ' icon_vui ', text)
    text = re.sub(r':\(+|:\-\(+|T_T|T\.T|:\'\(|:\{+', ' icon_buon ', text)
    text = re.sub(r':O|:o|:-O|:-o|o_O|O_o', ' icon_ngac_nhien ', text)
    text = re.sub(r':3', ' icon_meo ', text)
    text = re.sub(r'<3|♥', ' icon_yeu ', text)
    
    # Ánh xạ Emoji
    text = re.sub(r'[😭😢😿😥🥺💔]', ' emoji_buon ', text)
    text = re.sub(r'[😂🤣😄😃😀😊😆😁🥰😍😘❤️💖👍]', ' emoji_vui ', text)
    text = re.sub(r'[😡😠🤬👿💢]', ' emoji_tuc_gian ', text)
    text = re.sub(r'[🤢🤮😒🙄💩]', ' emoji_kinh_tom ', text)
    text = re.sub(r'[😱😨😰👻💀]', ' emoji_so_hai ', text)
    text = re.sub(r'[😲😯🤯😳😮🙀]', ' emoji_ngac_nhien ', text)
    
    # 4. Chuyển chữ thường
    text = text.lower()
    
    # 5. Rút gọn ký tự lặp kéo dài trước (ví dụ: đẹpppp -> đẹp, waaa -> wa)
    text = re.sub(r'([a-zA-Zà-ỹÀ-Ỹ])\1{2,}', r'\1', text)
    
    # 6. Chuẩn hóa teencode sau khi đã rút gọn (ví dụ: wa -> quá)
    words = [TEENCODE_DICT.get(w, w) for w in text.split()]
    text = ' '.join(words)
    
    # 7. Giữ lại chữ cái, số, khoảng trắng và gạch dưới (để giữ các token emoji_vui...)
    text = re.sub(r'[^\w\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 8. Tách từ tiếng Việt
    return ViTokenizer.tokenize(text)

def main():
    print("=" * 60)
    print("PIPELINE NHẬN DIỆN CẢM XÚC UIT-VSMEC (TF-IDF + LINEAR SVM)")
    print("NHÓM 12 - NHẬP MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN")
    print("=" * 60)
    
    train_file = os.path.join(DATA_DIR, "train.csv")
    val_file = os.path.join(DATA_DIR, "val.csv")
    test_file = os.path.join(DATA_DIR, "test.csv")
    
    print("\n1. Nạp dữ liệu:")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    print(f" - Train: {len(train_df)} mẫu")
    print(f" - Val:   {len(val_df)} mẫu")
    print(f" - Test:  {len(test_df)} mẫu")
    
    print("\n2. Đang thực hiện tiền xử lý văn bản (Unicode NFC, Emoji, Teencode, PyVi)...")
    train_df['clean_text'] = train_df['text'].apply(preprocess_text)
    val_df['clean_text'] = val_df['text'].apply(preprocess_text)
    test_df['clean_text'] = test_df['text'].apply(preprocess_text)
    print(" -> Tiền xử lý hoàn tất!")
    
    print("\n3. Vector hóa đặc trưng bằng TF-IDF (ngram_range=(1, 2), max_features=10000)...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    # LƯU Ý: fit_transform CHỈ trên tập train để chống rò rỉ dữ liệu (Data Leakage)
    X_train = tfidf.fit_transform(train_df['clean_text'])
    X_val = tfidf.transform(val_df['clean_text'])
    X_test = tfidf.transform(test_df['clean_text'])
    
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']
    print(f" -> Kích thước ma trận Train: {X_train.shape}")
    print(f" -> Kích thước ma trận Val:   {X_val.shape}")
    print(f" -> Kích thước ma trận Test:  {X_test.shape}")
    
    print("\n4. Tinh chỉnh siêu tham số C cho Linear SVM bằng GridSearchCV (5-Fold CV)...")
    param_grid = {
        'C': [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    }
    svm_base = LinearSVC(random_state=42, max_iter=3000, class_weight='balanced')
    grid_search = GridSearchCV(
        svm_base,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\n -> Best Parameter C: {grid_search.best_params_['C']}")
    print(f" -> Best Cross-Validation Macro F1: {grid_search.best_score_:.4f}")
    
    # Bảng đối sánh hiệu năng các giá trị C
    results_df = pd.DataFrame(grid_search.cv_results_)
    comparison_table = results_df[['param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    comparison_table.columns = ['Giá trị C', 'Mean Macro F1 (CV)', 'Std (Độ lệch chuẩn)', 'Thứ hạng']
    print("\nBẢNG ĐỐI SÁNH HIỆU NĂNG CÁC GIÁ TRỊ C:")
    print(comparison_table.to_string(index=False))
    
    # Lưu bảng đối sánh C ra CSV
    comparison_table.to_csv(os.path.join(BASE_DIR, "c_comparison_results.csv"), index=False, encoding='utf-8-sig')
    
    best_model = grid_search.best_estimator_
    
    print("\n5. Đánh giá mô hình tốt nhất:")
    # Đánh giá trên Validation
    y_pred_val = best_model.predict(X_val)
    val_macro_f1 = f1_score(y_val, y_pred_val, average='macro')
    val_weighted_f1 = f1_score(y_val, y_pred_val, average='weighted')
    val_acc = accuracy_score(y_val, y_pred_val)
    print(f"--- TẬP VALIDATION ---")
    print(f"Accuracy:    {val_acc:.4f}")
    print(f"Macro F1:    {val_macro_f1:.4f}")
    print(f"Weighted F1: {val_weighted_f1:.4f}")
    
    # Đánh giá trên Test
    y_pred_test = best_model.predict(X_test)
    test_macro_f1 = f1_score(y_test, y_pred_test, average='macro')
    test_weighted_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"\n--- TẬP TEST ---")
    print(f"Accuracy:    {test_acc:.4f}")
    print(f"Macro F1:    {test_macro_f1:.4f}")
    print(f"Weighted F1: {test_weighted_f1:.4f}")
    
    print("\nBÁO CÁO CHI TIẾT THEO TỪNG NHÃN (PER-CLASS F1):")
    labels_order = sorted(list(y_test.unique()))
    report_str = classification_report(y_test, y_pred_test, digits=4)
    print(report_str)
    
    # 6. Vẽ Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred_test, labels=labels_order)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title(f"Confusion Matrix - Linear SVM (C={grid_search.best_params_['C']})\nTest Macro F1: {test_macro_f1:.4f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n -> Đã lưu biểu đồ Ma trận nhầm lẫn tại: {cm_path}")
    
    # 7. Xuất file kết quả dự đoán predictions.csv theo đúng yêu cầu đề tài
    # Cấu trúc gồm: id, text, ground_truth, predicted_label
    predictions_df = pd.DataFrame({
        'id': list(range(len(test_df))),
        'text': test_df['text'],
        'ground_truth': test_df['label'],
        'predicted_label': y_pred_test
    })
    pred_path = os.path.join(BASE_DIR, "predictions.csv")
    predictions_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f" -> Đã xuất file dự đoán: {pred_path} ({len(predictions_df)} dòng)")
    print("\n5 dòng đầu của predictions.csv:")
    print(predictions_df.head())
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH TOÀN BỘ PIPELINE THÀNH CÔNG!")
    print("=" * 60)

if __name__ == "__main__":
    main()
