import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# BÀI TẬP LỚN: NHẬP MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN\n",
    "## ĐỀ TÀI: NHẬN DIỆN CẢM XÚC VĂN BẢN MẠNG XÃ HỘI TIẾNG VIỆT (UIT-VSMEC)\n",
    "**Nhóm thực hiện: Nhóm 12**  \n",
    "- Phương pháp: **TF-IDF + Linear SVM (GridSearchCV tinh chỉnh siêu tham số C)**  \n",
    "- Bộ dữ liệu: **UIT-VSMEC** (7 nhãn cảm xúc: *Enjoyment, Disgust, Other, Sadness, Anger, Fear, Surprise*)  \n",
    "- Thang đo đánh giá: **Macro F1-score, Weighted F1-score, Per-class F1 & Confusion Matrix**  \n",
    "- Sản phẩm bàn giao: File `predictions.csv`, Bảng đối sánh $C$, và Mã nguồn Notebook hoàn chỉnh.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 0. Cài đặt các thư viện phụ thuộc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Chạy lệnh này nếu bạn chạy trên Google Colab\n",
    "!pip install -q pyvi seaborn scikit-learn pandas matplotlib"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 0.1 Tự động tải dữ liệu từ Hugging Face về môi trường Colab\n",
    "(Không cần upload file thủ công từ máy tính, chạy cell này là có dữ liệu ngay lập tức!)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.makedirs('data/1_clapAI_UIT-VSMEC_csv', exist_ok=True)\n",
    "!wget -q -O data/1_clapAI_UIT-VSMEC_csv/train.csv https://huggingface.co/datasets/clapAI/UIT-VSMEC/resolve/main/train.csv\n",
    "!wget -q -O data/1_clapAI_UIT-VSMEC_csv/val.csv https://huggingface.co/datasets/clapAI/UIT-VSMEC/resolve/main/val.csv\n",
    "!wget -q -O data/1_clapAI_UIT-VSMEC_csv/test.csv https://huggingface.co/datasets/clapAI/UIT-VSMEC/resolve/main/test.csv\n",
    "print('Tải dữ liệu thành công!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Khai báo các thư viện cần thiết"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import re\n",
    "import unicodedata\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from pyvi import ViTokenizer\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score\n",
    "\n",
    "# Cấu hình font và hiển thị đồ họa\n",
    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
    "print(\"Các thư viện đã sẵn sàng!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Nạp và Khám phá Dữ liệu UIT-VSMEC\n",
    "Dữ liệu đã được chia theo tỷ lệ chuẩn:\n",
    "- Tập Train: 5,548 mẫu\n",
    "- Tập Validation: 686 mẫu\n",
    "- Tập Test: 693 mẫu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Đường dẫn tới thư mục dữ liệu CSV\n",
    "train_path = \"data/1_clapAI_UIT-VSMEC_csv/train.csv\"\n",
    "val_path   = \"data/1_clapAI_UIT-VSMEC_csv/val.csv\"\n",
    "test_path  = \"data/1_clapAI_UIT-VSMEC_csv/test.csv\"\n",
    "\n",
    "train_df = pd.read_csv(train_path)\n",
    "val_df   = pd.read_csv(val_path)\n",
    "test_df  = pd.read_csv(test_path)\n",
    "\n",
    "print(f\"Tập Train: {len(train_df)} dòng\")\n",
    "print(f\"Tập Val:   {len(val_df)} dòng\")\n",
    "print(f\"Tập Test:  {len(test_df)} dòng\")\n",
    "\n",
    "# Xem 5 mẫu đầu tiên\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Thống kê phân bố 7 nhãn cảm xúc trên tập Train\n",
    "plt.figure(figsize=(9, 4))\n",
    "order = train_df['label'].value_counts().index\n",
    "sns.countplot(data=train_df, x='label', order=order, palette='viridis')\n",
    "plt.title(\"Phân bố 7 nhãn cảm xúc trên tập Train UIT-VSMEC\")\n",
    "plt.xlabel(\"Nhãn cảm xúc\")\n",
    "plt.ylabel(\"Số lượng mẫu\")\n",
    "plt.show()\n",
    "\n",
    "print(train_df['label'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Pipeline Tiền xử lý Văn bản Tiếng Việt Mạng xã hội\n",
    "Các bước xử lý chuyên sâu nhằm bảo toàn tối đa tín hiệu cảm xúc:\n",
    "1. **Unicode NFC**: Khắc phục triệt để lỗi dấu tổ hợp bị tách rời trên mạng xã hội.\n",
    "2. **Bảo tồn Emoticon & Emoji**: Thay vì xóa bỏ (làm mất tín hiệu cảm xúc), ta chuyển thành các token đặc trưng (`icon_vui`, `emoji_buon`, `emoji_tuc_gian`...).\n",
    "3. **Chuẩn hóa Teencode**: Chuyển các từ viết tắt phổ biến về dạng chuẩn (`ko` -> `không`, `dc` -> `được`...).\n",
    "4. **Rút gọn ký tự lặp kéo dài**: `quáaaaa` -> `quá`, `đẹpppp` -> `đẹp`.\n",
    "5. **Tách từ Tiếng Việt (Word Segmentation)**: Sử dụng thư viện `pyvi` để ghép các từ phức."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "TEENCODE_DICT = {\n",
    "    'ko': 'không', 'k': 'không', 'khg': 'không', 'hông': 'không', 'hổng': 'không',\n",
    "    'dc': 'được', 'đc': 'được', 'dk': 'được', 'đk': 'được',\n",
    "    'j': 'gì', 'wa': 'quá', 'wá': 'quá', 'waa': 'quá',\n",
    "    'bik': 'biết', 'bít': 'biết', 'vs': 'với', 'cx': 'cũng',\n",
    "    'r': 'rồi', 'rùi': 'rồi', 'thui': 'thôi', 'chừ': 'giờ',\n",
    "    'ae': 'anh em', 'mn': 'mọi người', 'cmt': 'bình luận',\n",
    "    'sgk': 'sách giáo khoa', 'huhu': 'huhu emoji_buon', 'haha': 'haha emoji_vui',\n",
    "    'ntn': 'như thế nào', 'nv': 'như vậy', 'ib': 'inbox', 'fb': 'facebook',\n",
    "    'ad': 'admin', 'cty': 'công ty', 'stt': 'status', 'nyc': 'người yêu cũ'\n",
    "}\n",
    "\n",
    "def preprocess_text(text):\n",
    "    text = str(text)\n",
    "    # Chuẩn hóa Unicode NFC\n",
    "    text = unicodedata.normalize('NFC', text)\n",
    "    # Xóa URL\n",
    "    text = re.sub(r'http\\S+|www\\.\\S+', ' ', text)\n",
    "    \n",
    "    # Ánh xạ Emoticon cảm xúc\n",
    "    text = re.sub(r':\\)+|:\\-\\)+|\\^_\\^|\\^\\^|:D|:d|xD|XD|;\\)|;-\\)', ' icon_vui ', text)\n",
    "    text = re.sub(r\":\\(+|:\\-\\(+|T_T|T\\.T|:'\\(|:\\{+\", ' icon_buon ', text)\n",
    "    text = re.sub(r':O|:o|:-O|:-o|o_O|O_o', ' icon_ngac_nhien ', text)\n",
    "    text = re.sub(r':3', ' icon_meo ', text)\n",
    "    text = re.sub(r'<3|♥', ' icon_yeu ', text)\n",
    "    \n",
    "    # Ánh xạ Emoji cảm xúc\n",
    "    text = re.sub(r'[😭😢😿😥🥺💔]', ' emoji_buon ', text)\n",
    "    text = re.sub(r'[😂🤣😄😃😀😊😆😁🥰😍😘❤️💖👍]', ' emoji_vui ', text)\n",
    "    text = re.sub(r'[😡😠🤬👿💢]', ' emoji_tuc_gian ', text)\n",
    "    text = re.sub(r'[🤢🤮😒🙄💩]', ' emoji_kinh_tom ', text)\n",
    "    text = re.sub(r'[😱😨😰👻💀]', ' emoji_so_hai ', text)\n",
    "    text = re.sub(r'[😲😯🤯😳😮🙀]', ' emoji_ngac_nhien ', text)\n",
    "    \n",
    "    # Chuyển về chữ thường\n",
    "    text = text.lower()\n",
    "    \n",
    "    # Giảm thiểu ký tự lặp kéo dài trước (ví dụ: đẹpppp -> đẹp, waaa -> wa)\n",
    "    text = re.sub(r'([a-zA-Zà-ỹÀ-Ỹ])\\1{2,}', r'\\1', text)\n",
    "    \n",
    "    # Chuẩn hóa Teencode sau khi đã rút gọn (ví dụ: wa -> quá)\n",
    "    words = [TEENCODE_DICT.get(w, w) for w in text.split()]\n",
    "    text = ' '.join(words)\n",
    "    \n",
    "    # Giữ lại chữ cái, số, khoảng trắng và gạch dưới token\n",
    "    text = re.sub(r'[^\\w\\s_]', ' ', text)\n",
    "    text = re.sub(r'\\s+', ' ', text).strip()\n",
    "    \n",
    "    # Tách từ tiếng Việt bằng PyVi\n",
    "    return ViTokenizer.tokenize(text)\n",
    "\n",
    "# Kiểm thử nhanh hàm tiền xử lý\n",
    "sample = \"thời buổi bây chừ chuyện gì cũng xảy ra :((( đừng tin ai waaa 😭😭😭\"\n",
    "print(\"Trước tiền xử lý:\", sample)\n",
    "print(\"Sau tiền xử lý:  \", preprocess_text(sample))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Áp dụng tiền xử lý lên toàn bộ dữ liệu\n",
    "print(\"Đang tiền xử lý toàn bộ tập Train, Val, Test...\")\n",
    "train_df['clean_text'] = train_df['text'].apply(preprocess_text)\n",
    "val_df['clean_text']   = val_df['text'].apply(preprocess_text)\n",
    "test_df['clean_text']  = test_df['text'].apply(preprocess_text)\n",
    "print(\"Hoàn tất tiền xử lý!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Trích xuất đặc trưng bằng TF-IDF (Feature Extraction)\n",
    "- Sử dụng `ngram_range=(1, 2)` (Unigram + Bigram) để bắt các cụm từ quan trọng (như cụm phủ định: *không thích*, *rất buồn*).\n",
    "- `max_features=10000`: Giữ lại 10,000 đặc trưng có trọng số cao nhất.\n",
    "- `sublinear_tf=True`: Áp dụng hàm logarit làm mượt tần số xuất hiện của từ.\n",
    "- **QUY TẮC BẢO VỆ CHỐNG DATA LEAKAGE**: `fit_transform` chỉ trên tập `Train`. Tập `Val` và `Test` chỉ được `transform`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf = TfidfVectorizer(\n",
    "    ngram_range=(1, 2),\n",
    "    max_features=10000,\n",
    "    sublinear_tf=True\n",
    ")\n",
    "\n",
    "X_train = tfidf.fit_transform(train_df['clean_text'])\n",
    "X_val   = tfidf.transform(val_df['clean_text'])\n",
    "X_test  = tfidf.transform(test_df['clean_text'])\n",
    "\n",
    "y_train = train_df['label']\n",
    "y_val   = val_df['label']\n",
    "y_test  = test_df['label']\n",
    "\n",
    "print(f\"Ma trận đặc trưng Train: {X_train.shape}\")\n",
    "print(f\"Ma trận đặc trưng Val:   {X_val.shape}\")\n",
    "print(f\"Ma trận đặc trưng Test:  {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Huấn luyện Mô hình & Tinh chỉnh siêu tham số $C$ bằng GridSearchCV\n",
    "- Mô hình: **Linear SVM (LinearSVC)** với `class_weight='balanced'` để tự động cân bằng trọng số cho các nhãn ít mẫu (như *Surprise, Fear, Anger*).\n",
    "- Tinh chỉnh siêu tham số $C$ qua dải giá trị $[0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]$ bằng 5-Fold Cross Validation trên tập Train.\n",
    "- Tiêu chí đánh giá để chọn $C$ tốt nhất: `scoring='f1_macro'`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "param_grid = {\n",
    "    'C': [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]\n",
    "}\n",
    "\n",
    "svm_base = LinearSVC(random_state=42, max_iter=3000, class_weight='balanced')\n",
    "\n",
    "grid_search = GridSearchCV(\n",
    "    svm_base,\n",
    "    param_grid,\n",
    "    cv=5,\n",
    "    scoring='f1_macro',\n",
    "    n_jobs=-1,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "print(f\"\\nSiêu tham số C tối ưu: {grid_search.best_params_['C']}\")\n",
    "print(f\"Best Macro F1 (Cross-Validation 5-Fold): {grid_search.best_score_:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# BẢNG ĐỐI SÁNH HIỆU NĂNG CÁC GIÁ TRỊ C (Yêu cầu bắt buộc trong đề tài)\n",
    "results_df = pd.DataFrame(grid_search.cv_results_)\n",
    "comparison_table = results_df[['param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']].copy()\n",
    "comparison_table.columns = ['Giá trị C', 'Mean Macro F1 (CV)', 'Độ lệch chuẩn (Std)', 'Thứ hạng']\n",
    "comparison_table.sort_values(by='Thứ hạng', inplace=True)\n",
    "\n",
    "display(comparison_table)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Vẽ biểu đồ đối sánh hiệu năng các giá trị C\n",
    "plt.figure(figsize=(8, 4))\n",
    "plt.plot(comparison_table['Giá trị C'], comparison_table['Mean Macro F1 (CV)'], marker='o', color='purple', linewidth=2)\n",
    "plt.title(\"Đồ thị hiệu năng Macro F1 theo các giá trị C khác nhau\")\n",
    "plt.xlabel(\"Giá trị C\")\n",
    "plt.ylabel(\"Mean Macro F1 (5-Fold CV)\")\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Đánh giá Mô hình trên Tập Validation và Tập Test\n",
    "Sử dụng mô hình tốt nhất với $C$ tối ưu để đánh giá theo 3 thang đo:\n",
    "1. **Accuracy**\n",
    "2. **Macro F1-score (MF1)**\n",
    "3. **Weighted F1-score (WF1)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = grid_search.best_estimator_\n",
    "\n",
    "# Đánh giá trên Validation\n",
    "y_pred_val = best_model.predict(X_val)\n",
    "print(\"=== KẾT QUẢ TRÊN TẬP VALIDATION ===\")\n",
    "print(f\"Accuracy:    {accuracy_score(y_val, y_pred_val):.4f}\")\n",
    "print(f\"Macro F1:    {f1_score(y_val, y_pred_val, average='macro'):.4f}\")\n",
    "print(f\"Weighted F1: {f1_score(y_val, y_pred_val, average='weighted'):.4f}\")\n",
    "\n",
    "# Đánh giá trên Test\n",
    "y_pred_test = best_model.predict(X_test)\n",
    "print(\"\\n=== KẾT QUẢ TRÊN TẬP TEST ===\")\n",
    "print(f\"Accuracy:    {accuracy_score(y_test, y_pred_test):.4f}\")\n",
    "print(f\"Macro F1:    {f1_score(y_test, y_pred_test, average='macro'):.4f}\")\n",
    "print(f\"Weighted F1: {f1_score(y_test, y_pred_test, average='weighted'):.4f}\")\n",
    "\n",
    "print(\"\\n=== BÁO CÁO CHI TIẾT THEO TỪNG NHÃN (PER-CLASS F1) ===\")\n",
    "print(classification_report(y_test, y_pred_test, digits=4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7. Ma trận nhầm lẫn (Confusion Matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "labels_order = sorted(list(y_test.unique()))\n",
    "cm = confusion_matrix(y_test, y_pred_test, labels=labels_order)\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
    "            xticklabels=labels_order, yticklabels=labels_order)\n",
    "plt.title(f\"Confusion Matrix - Linear SVM (C={grid_search.best_params_['C']})\\nTest Macro F1 = {f1_score(y_test, y_pred_test, average='macro'):.4f}\")\n",
    "plt.xlabel(\"Nhãn dự đoán (Predicted Label)\")\n",
    "plt.ylabel(\"Nhãn thực tế (True Label)\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"confusion_matrix.png\", dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8. Xuất file kết quả dự đoán `predictions.csv`\n",
    "File bàn giao theo đúng định dạng yêu cầu của giảng viên:\n",
    "- `id`: Thứ tự bình luận\n",
    "- `text`: Câu bình luận gốc\n",
    "- `ground_truth`: Nhãn thực tế\n",
    "- `predicted_label`: Nhãn do mô hình dự đoán"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_df = pd.DataFrame({\n",
    "    'id': list(range(len(test_df))),\n",
    "    'text': test_df['text'],\n",
    "    'ground_truth': test_df['label'],\n",
    "    'predicted_label': y_pred_test\n",
    "})\n",
    "\n",
    "predictions_df.to_csv(\"predictions.csv\", index=False, encoding='utf-8-sig')\n",
    "print(f\"Đã xuất file dự đoán: predictions.csv với {len(predictions_df)} dòng!\")\n",
    "predictions_df.head(10)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("c:/Users/HP/Documents/Bai tap va 1 so file j do/nmxlnntn/btl/UIT_VSMEC_Nhom12.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, ensure_ascii=False, indent=2)

print("Notebook UIT_VSMEC_Nhom12.ipynb created successfully!")
