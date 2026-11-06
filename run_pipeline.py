import os
import re
import sys
import time
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvi import ViTokenizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.base import clone

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "1_clapAI_UIT-VSMEC_csv")

# 1. BẢNG CHUẨN HÓA TEENCODE & TỪ VIẾT TẮT MXH (CHUẨN HÓA CHÍNH TẢ KHÁCH QUAN)
TEENCODE_DICT = {
    'ko': 'không', 'k': 'không', 'khg': 'không', 'hông': 'không', 'hổng': 'không', 
    'kô': 'không', 'hok': 'không', 'hem': 'không', 'khong': 'không',
    'dc': 'được', 'đc': 'được', 'dk': 'được', 'đk': 'được', 'đựoc': 'được', 'duoc': 'được',
    'j': 'gì', 'wa': 'quá', 'wá': 'quá', 'waa': 'quá', 'woa': 'quá', 'qua': 'quá',
    'bik': 'biết', 'bít': 'biết', 'biet': 'biết',
    'vs': 'với', 'cx': 'cũng',
    'r': 'rồi', 'rùi': 'rồi', 'thui': 'thôi', 'chừ': 'giờ',
    'ae': 'anh em', 'mn': 'mọi người', 'mng': 'mọi người', 'cmt': 'bình luận',
    'sgk': 'sách giáo khoa',
    'ntn': 'như thế nào', 'nv': 'như vậy', 'ib': 'nhắn tin', 'fb': 'facebook',
    'ad': 'admin', 'cty': 'công ty', 'stt': 'status', 'nyc': 'người yêu cũ',
    'mik': 'mình', 'mk': 'mình',
    'nhg': 'nhưng', 'lm': 'làm', 'ms': 'mới', 'kp': 'không phải',
    'bt': 'bình thường', 'uk': 'ừ', 'uh': 'ừ', 'uhm': 'ừ', 'ừm': 'ừ',
    'vl': 'vãi', 'vcl': 'vãi', 'clgt': 'cái gì thế',
    'kb': 'kết bạn', 'qtqd': 'quá trời quá đất', 'đt': 'điện thoại'
}

def preprocess_text(text):
    text = str(text)
    
    # 1. Chuẩn hóa bảng mã Unicode NFC
    text = unicodedata.normalize('NFC', text)
    
    # 2. Xóa URL rác
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # 3. Ánh xạ Emoticon cảm xúc
    text = re.sub(r':\)+|:\-\)+|\^_\^|\^\^|:D|:d|xD|XD|;\)|;-\)|:3|\(y\)', ' icon_vui ', text)
    text = re.sub(r":\(+|:\-\(+|T_T|T\.T|:'\(|:\{+|;-;", ' icon_buon ', text)
    text = re.sub(r':O|:o|:-O|:-o|o_O|O_o|O\.o|o\.O|@@', ' icon_ngac_nhien ', text)
    text = re.sub(r'<3|♥', ' icon_yeu ', text)
    
    # 4. Ánh xạ Emoji cảm xúc
    text = re.sub(r'[😭😢😿😥🥺💔🌧️🥀😞😔]', ' emoji_buon ', text)
    text = re.sub(r'[😂🤣😄😃😀😊😆😁🥰😍😘❤️💖👍🎉✨👏💐]', ' emoji_vui ', text)
    text = re.sub(r'[😡😠🤬👿💢😤🖕]', ' emoji_tuc_gian ', text)
    text = re.sub(r'[🤢🤮😒🙄💩👎]', ' emoji_kinh_tom ', text)
    text = re.sub(r'[😱😨😰👻💀🥶]', ' emoji_so_hai ', text)
    text = re.sub(r'[😲😯🤯😳😮🙀🧐‼️⁉️]', ' emoji_ngac_nhien ', text)
    
    # 5. XỬ LÝ TIẾNG CƯỜI / KHÓC ĐẶC THÙ MXH TRƯỚC KHI RÚT GỌN LẶP
    text = re.sub(r'\b(k{3,}|kkk+)\b', ' icon_vui ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b((ha){2,}|(hi){2,}|(he){2,}|h{3,})\b', ' emoji_vui ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(hu){2,}|huhu\b', ' emoji_buon ', text, flags=re.IGNORECASE)
    
    # 6. Bảo tồn dấu câu cảm xúc quan trọng (Surprise và Anger)
    text = re.sub(r'!+', ' token_than ', text)
    text = re.sub(r'\?+', ' token_hoi ', text)
    
    # 7. Thay thế dấu câu và ký tự đặc biệt còn lại bằng khoảng trắng TRƯỚC KHI tra teencode
    text = re.sub(r'[^\w\s_]', ' ', text)
    
    # 8. Chuyển về chữ thường
    text = text.lower()
    
    # 9. Rút gọn ký tự lặp kéo dài (ví dụ: đẹpppp -> đẹp, ngonnn -> ngon)
    text = re.sub(r'([a-zA-Zà-ỹÀ-Ỹ])\1{2,}', r'\1', text)
    
    # 10. Chuẩn hóa Teencode khách quan sau khi đã tách rời dấu câu
    words = [TEENCODE_DICT.get(w, w) for w in text.split()]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 11. Tách từ tiếng Việt bằng PyVi
    return ViTokenizer.tokenize(text)

def main():
    print("=" * 65)
    print("PIPELINE NHẬN DIỆN CẢM XÚC UIT-VSMEC (FEATUREUNION + LINEAR SVM)")
    print("NHÓM 12 - NHẬP MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN")
    print("=" * 65)
    
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
    
    print("\n2. Đang thực hiện tiền xử lý văn bản...")
    train_df['clean_text'] = train_df['text'].apply(preprocess_text)
    val_df['clean_text'] = val_df['text'].apply(preprocess_text)
    test_df['clean_text'] = test_df['text'].apply(preprocess_text)
    print(" -> Tiền xử lý hoàn tất!")
    
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']
    
    print("\n3. Xây dựng Pipeline (FeatureUnion: Word + Char_wb TF-IDF & LinearSVC)...")
    features = FeatureUnion([
        ('word', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            lowercase=True
        )),
        ('char', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            sublinear_tf=True,
            min_df=2,
            lowercase=True
        ))
    ])

    pipe = Pipeline([
        ('tfidf', features),
        ('svm', LinearSVC(random_state=42, max_iter=5000, class_weight='balanced'))
    ])

    param_grid = {
        'svm__C': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    print("\n4. Tinh chỉnh siêu tham số C bằng 5-Fold StratifiedKFold...")
    t_start = time.time()
    grid.fit(train_df['clean_text'], y_train)
    best_c = grid.best_params_['svm__C']
    print(f" -> Hoàn thành trong {time.time() - t_start:.2f}s | Best C = {best_c} | Best CV Macro F1 = {grid.best_score_:.4f}")
    
    results_df = pd.DataFrame(grid.cv_results_)
    c_table = results_df[['param_svm__C', 'mean_test_score', 'std_test_score', 'rank_test_score']].copy()
    c_table.columns = ['Giá trị C', 'Mean Macro F1 (CV)', 'Độ lệch chuẩn (Std)', 'Thứ hạng']
    c_table.sort_values(by='Thứ hạng', inplace=True)
    print("\nBẢNG ĐỐI SÁNH HIỆU NĂNG CÁC GIÁ TRỊ C:")
    print(c_table.to_string(index=False))
    c_table.to_csv(os.path.join(BASE_DIR, "c_comparison_results.csv"), index=False, encoding='utf-8-sig')
    
    best_model_train_only = grid.best_estimator_
    pred_val_only = best_model_train_only.predict(val_df['clean_text'])
    pred_test_only = best_model_train_only.predict(test_df['clean_text'])
    
    val_acc_only = accuracy_score(y_val, pred_val_only)
    val_mf1_only = f1_score(y_val, pred_val_only, average='macro')
    val_wf1_only = f1_score(y_val, pred_val_only, average='weighted')
    
    test_acc_only = accuracy_score(y_test, pred_test_only)
    test_mf1_only = f1_score(y_test, pred_test_only, average='macro')
    test_wf1_only = f1_score(y_test, pred_test_only, average='weighted')
    
    print("\n" + "=" * 65)
    print("GIAO THỨC 1: MÔ HÌNH TRAIN-ONLY (5,548 MẪU) - SO SÁNH CHUẨN VỚI BÀI BÁO")
    print("=" * 65)
    print(f"VAL  (686 mẫu) -> Acc: {val_acc_only*100:.2f}% | Macro F1: {val_mf1_only*100:.2f}% | Weighted F1: {val_wf1_only*100:.2f}%")
    print(f"TEST (693 mẫu) -> Acc: {test_acc_only*100:.2f}% | Macro F1: {test_mf1_only*100:.2f}% | Weighted F1: {test_wf1_only*100:.2f}%")
    
    full_train_text = pd.concat([train_df['clean_text'], val_df['clean_text']], ignore_index=True)
    full_train_y = pd.concat([y_train, y_val], ignore_index=True)
    final_model_full = clone(best_model_train_only)
    final_model_full.fit(full_train_text, full_train_y)
    pred_test_full = final_model_full.predict(test_df['clean_text'])
    
    print("\n" + "=" * 65)
    print("GIAO THỨC 2: MÔ HÌNH MỞ RỘNG TỐI ƯU (TRAIN + VAL: 6,234 MẪU)")
    print("=" * 65)
    print(f"TEST (693 mẫu) -> Acc: {accuracy_score(y_test, pred_test_full)*100:.2f}% | Macro F1: {f1_score(y_test, pred_test_full, average='macro')*100:.2f}% | Weighted F1: {f1_score(y_test, pred_test_full, average='weighted')*100:.2f}%")
    
    print("\nBÁO CÁO CHI TIẾT THEO TỪNG NHÃN (PER-CLASS F1 - TRAIN-ONLY):")
    labels_order = sorted(list(y_test.unique()))
    print(classification_report(y_test, pred_test_only, digits=4))
    
    cm = confusion_matrix(y_test, pred_test_only, labels=labels_order)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title(f"Confusion Matrix - Linear SVM (C={best_c})\nTest Macro F1 = {test_mf1_only:.4f}", fontsize=11, fontweight='bold')
    plt.xlabel("Nhãn dự đoán (Predicted Label)")
    plt.ylabel("Nhãn thực tế (True Label)")
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    predictions_df = pd.DataFrame({
        'id': list(range(len(test_df))),
        'text': test_df['text'],
        'ground_truth': test_df['label'],
        'predicted_label': pred_test_only
    })
    pred_path = os.path.join(BASE_DIR, "predictions.csv")
    predictions_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"\n -> Đã xuất file dự đoán: {pred_path} ({len(predictions_df)} dòng)")

if __name__ == "__main__":
    main()
