# TỔNG HỢP DATASET UIT-VSMEC (NHÓM 12 - NHẬP MÔN XLNNTN)

Thư mục này chứa đầy đủ 3 phiên bản dataset **UIT-VSMEC** (Vietnamese Social Media Emotion Corpus) được tải về từ Hugging Face và nguồn tác giả:

---

## 📁 Cấu trúc thư mục

```text
data/
├── 1_clapAI_UIT-VSMEC_csv/          <-- KHUYÊN DÙNG CHO BÀI TẬP LỚN
│   ├── train.csv                    (5,548 mẫu | Cột: text, label)
│   ├── val.csv                      (686 mẫu)
│   ├── test.csv                     (693 mẫu)
│   └── README.md
│
├── 2_tridm_UIT-VSMEC_json/
│   ├── train.json                   (5,548 mẫu | Khóa: Sentence, Emotion)
│   ├── valid.json                   (686 mẫu)
│   ├── test.json                    (693 mẫu)
│   └── README.md
│
└── 3_SEACrowd_uit_vsmec/
    ├── uit_vsmec.py                 (Mã nguồn nạp dataset chuẩn SEACrowd)
    ├── requirements.txt
    ├── LICENSE
    ├── README.md
    ├── __init__.py
    └── raw_data/                    (Dữ liệu CSV gốc từ Google Sheets của tác giả UIT)
        ├── train.csv                (5,548 dòng | Cột: Emotion, Sentence)
        ├── valid.csv                (686 dòng)
        └── test.csv                 (693 dòng)
```

---

## 📊 Thống kê 7 nhãn cảm xúc (Tập Train - 5,548 mẫu)

| STT | Nhãn cảm xúc | Số lượng mẫu (Train) | Tỷ lệ (%) |
|:---:|:-------------|:--------------------:|:---------:|
| 1   | **Enjoyment** (Thích thú/Vui vẻ) | 1,558 | ~28.1% |
| 2   | **Disgust** (Kinh tởm/Ghét bỏ)   | 1,071 | ~19.3% |
| 3   | **Other** (Khác / Trung tính)    | 1,021 | ~18.4% |
| 4   | **Sadness** (Buồn bã)            | 947   | ~17.1% |
| 5   | **Anger** (Tức giận)             | 391   | ~7.0%  |
| 6   | **Fear** (Sợ hãi)                | 318   | ~5.7%  |
| 7   | **Surprise** (Bất ngờ)           | 242   | ~4.4%  |

> **Tổng cộng:** Train: **5,548** mẫu | Val: **686** mẫu | Test: **693** mẫu | Toàn bộ: **6,927** mẫu.

---

## 🚀 Hướng dẫn nạp dữ liệu nhanh (cho bài TF-IDF + Linear SVM của Nhóm 12)

```python
import pandas as pd

# Nạp dữ liệu bản clapAI (khuyên dùng trực tiếp với pandas)
train_df = pd.read_csv("data/1_clapAI_UIT-VSMEC_csv/train.csv")
val_df = pd.read_csv("data/1_clapAI_UIT-VSMEC_csv/val.csv")
test_df = pd.read_csv("data/1_clapAI_UIT-VSMEC_csv/test.csv")

print("Train shape:", train_df.shape)
print("Val shape:  ", val_df.shape)
print("Test shape: ", test_df.shape)

# Xem 5 dòng đầu
print(train_df.head())
```
