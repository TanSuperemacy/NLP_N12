# GIẢI THÍCH CHI TIẾT ĐỀ TÀI NHÓM 12
## MÔN: NHẬP MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)

**Tên đề tài:**
> **Nhận diện cảm xúc văn bản mạng xã hội tiếng Việt (UIT-VSMEC) bằng TF-IDF + Linear SVM (GridSearchCV tinh chỉnh siêu tham số C)**

**Mục tiêu cốt lõi:**  
Xây dựng mô hình máy học tự động đọc hiểu một bình luận tiếng Việt trên mạng xã hội Facebook và phân loại chính xác vào 1 trong **7 trạng thái cảm xúc**: *Enjoyment (Thích thú), Disgust (Kinh tởm/Ghét bỏ), Other (Khác/Trung tính), Sadness (Buồn bã), Anger (Tức giận), Fear (Sợ hãi), Surprise (Bất ngờ)*.

---

## 1. THÀNH PHẦN 1: BỘ DỮ LIỆU UIT-VSMEC
* **Nó là gì?**  
  Bộ dữ liệu gồm **6,927 bình luận Facebook tiếng Việt**, được gán nhãn thủ công bởi các chuyên gia ngôn ngữ:
  * **Tập Train (Huấn luyện):** 5,548 mẫu
  * **Tập Validation (Phát triển):** 686 mẫu
  * **Tập Test (Kiểm thử độc lập):** 693 mẫu
* **Mục đích:**  
  * Cung cấp dữ liệu mẫu để thuật toán học các đặc trưng cảm xúc (tập Train).
  * Làm thước đo khách quan để tinh chỉnh tham số (tập Val) và kiểm tra khả năng dự đoán thực tế (tập Test).
* **Tại sao lại dùng nó?**  
  Đây là bộ dữ liệu chuẩn mực và uy tín hàng đầu tại Việt Nam cho bài toán này, do nhóm nghiên cứu trường ĐH Công nghệ Thông tin (UIT - ĐHQG TP.HCM) xây dựng và công bố tại hội nghị quốc tế PACLING 2019 / Springer 2020 (*Ho et al., 2020*).

---

## 2. THÀNH PHẦN 2: PIPELINE TIỀN XỬ LÝ VĂN BẢN (TEXT PREPROCESSING)
* **Nó là gì?**  
  Chuỗi các bước chuẩn hóa và "gọt giũa" văn bản thô từ Facebook trước khi đưa vào mô hình máy học.
* **Mục đích:**  
  Bình luận mạng xã hội có độ nhiễu rất cao (lỗi font, teencode, icon, viết kéo dài). Tiền xử lý giúp giảm thiểu nhiễu và bảo toàn trọn vẹn tín hiệu cảm xúc.
* **Tại sao phải dùng từng bước cụ thể:**
  1. **Chuẩn hóa Unicode NFC:**  
     * *Vấn đề:* Tiếng Việt có 2 chuẩn Unicode (Dựng sẵn NFC và Tổ hợp NFD). Bình luận mạng xã hội hay bị lỗi dấu rời (ví dụ chữ `thích` bị tách thành `thi ́ ch`).
     * *Giải pháp:* Đưa về chuẩn NFC giúp máy tính nhận diện đúng từ vựng thống nhất.
  2. **Ánh xạ Emoticon & Emoji thành Token ngữ nghĩa:**  
     * *Vấn đề:* Các công cụ NLP thông thường hay dùng regex xóa sạch ký tự đặc biệt, làm mất hoàn toàn emoji (`😭`, `😡`, `😂`) và icon (`:)))`, `:(((`).
     * *Giải pháp:* Emoji chính là **"linh hồn" biểu đạt cảm xúc**. Nhóm chuyển đổi chúng thành token chuyên dụng (`emoji_buon`, `emoji_tuc_gian`, `icon_vui`...) để mô hình bắt trọn vẹn đặc trưng này.
  3. **Bắt tiếng cười/khóc TRƯỚC khi rút gọn lặp & Bảo tồn dấu câu biểu cảm:**  
     * Ánh xạ `kkk+`, `haha+`, `hihi+` $\rightarrow$ `icon_vui`/`emoji_vui` trước khi rút gọn lặp để tránh lỗi `kkk` bị rút thành `k` rồi tra từ điển thành `không` (phủ định).
     * Ánh xạ `!+` $\rightarrow$ `token_than` và `?+` $\rightarrow$ `token_hoi` để giữ tín hiệu quan trọng cho 2 nhãn khó nhất là *Surprise* và *Anger*.
     * Thay dấu câu còn lại bằng khoảng trắng trước khi tra từ điển để các từ dính dấu câu như `"ko,"`, `"dc."` tách ra thành `"ko "`, `"dc "`.
  4. **Rút gọn ký tự lặp kéo dài & Chuẩn hóa Teencode khách quan:**  
     * Người dùng mạng xã hội hay gõ kéo dài để nhấn mạnh: `quáaaaa` $\rightarrow$ `quá`, `đẹpppp` $\rightarrow$ `đẹp`.
     * Chuyển các từ mạng xã hội phổ biến về dạng chuẩn (`ko` $\rightarrow$ `không`, `dc` $\rightarrow$ `được`, `wa` $\rightarrow$ `quá`, `j` $\rightarrow$ `gì`, `vl` $\rightarrow$ `vãi`...).
  5. **Tách từ tiếng Việt bằng PyVi (Word Segmentation):**  
     * Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không trùng với dấu cách. Ví dụ: `nước mắt` nếu không ghép sẽ bị hiểu thành `nước` (chất lỏng) và `mắt` (thị giác). Thư viện `pyvi` giúp ghép thành `thời_buổi`, `nước_mắt`, mang ý nghĩa cảm xúc trọn vẹn.

---

## 3. THÀNH PHẦN 3: TRÍCH XUẤT ĐẶC TRƯNG KÉP BẰNG FEATUREUNION (WORD + CHAR_WB TF-IDF)
* **Nó là gì?**  
  Cầu nối toán học chuyển đổi **Văn bản (Text) $\rightarrow$ Ma trận số học (Vectors)** bằng kiến trúc kết hợp 2 tầng đặc trưng (`FeatureUnion`).
* **Mục đích:**  
  Các thuật toán máy học không đọc được chữ, chỉ tính toán được trên ma trận số.
* **Tại sao lại dùng FeatureUnion kết hợp Word N-gram (1, 2), Char_wb (2, 5) và Sublinear TF:**
  * **Word TF-IDF (`ngram_range=(1, 2)`, `min_df=2`, `max_df=0.9`):** Bắt trọn các từ đơn và **cụm 2 từ đi liền nhau**, đặc biệt là **từ phủ định và tăng tiến** (`không_thích`, `quá_buồn`, `rất_vui`).
  * **Character n-grams trong từ (`analyzer='char_wb'`, `ngram_range=(2, 5)`, `min_df=2`):** Cắt từ thành các đoạn ký tự con 2–5 ký tự nằm trong ranh giới từ, giúp mô hình nhận diện các biến thể teencode, lỗi chính tả hoặc từ kéo dài mà từ điển chưa bao quát hết (tương tự ý tưởng kết hợp Word + Char n-grams tại Mục 5.2 bài báo gốc UIT).
  * **Sublinear TF (`1 + log(tf)`):** Tránh việc một từ lặp lại 10 lần thì có trọng số gấp 10 lần (hàm logarit phản ánh đúng mức độ nhấn mạnh thực tế hơn).
  * **Nguyên tắc chống rò rỉ dữ liệu tuyệt đối (Zero Data Leakage):** Đóng gói `FeatureUnion` và `LinearSVC` vào `sklearn.pipeline.Pipeline` để khi chạy 5-Fold Cross-Validation, TF-IDF chỉ `fit` trên 4 fold train và `transform` trên fold kiểm thử.

---

## 4. THÀNH PHẦN 4: THUẬT TOÁN LINEAR SVM (LINEAR SUPPORT VECTOR CLASSIFIER)
* **Nó là gì?**  
  Mô hình máy học tìm các **siêu phẳng (hyperplane)** tối ưu trong không gian đặc trưng chiều cao để phân tách 7 lớp cảm xúc với lề (margin) rộng nhất.
* **Mục đích:**  
  Dựa vào tọa độ vector số học của câu bình luận để đưa ra quyết định: câu này thuộc nhãn cảm xúc nào.
* **Tại sao lại dùng Linear SVM:**
  1. Sau bước TF-IDF kép, dữ liệu có số chiều rất lớn. Trong không gian nhiều chiều, dữ liệu hầu như đã phân tách tuyến tính được. Dùng **Linear SVM** chạy siêu nhanh (chỉ mất vài giây), hội tụ ổn định và tránh bị học vẹt (overfitting) so với nhân phi tuyến (RBF).
  2. Kích hoạt cơ chế **`class_weight='balanced'`**: Tập UIT-VSMEC bị lệch nhãn nặng (*Enjoyment* có 1,558 câu nhưng *Surprise* chỉ có 242 câu). Tham số này tự động phạt nặng hơn nếu mô hình đoán sai ở các nhãn ít mẫu, giúp cân bằng độ chính xác cho toàn bộ 7 nhãn.

---

## 5. THÀNH PHẦN 5: TINH CHỈNH SIÊU THAM SỐ $C$ BẰNG GRIDSEARCHCV
*(Kết quả lưu tại: `c_comparison_results.csv`)*

* **Nó là gì?**  
  Thực nghiệm tự động quét qua các giá trị $C \in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]$ bằng phương pháp **5-Fold Stratified Cross Validation** trên tập Train để tìm ra giá trị $C$ tối ưu nhất.
* **Ý nghĩa bản chất của tham số $C$:**
  * $C$ là **hệ số phạt khi mô hình đoán sai lề**. Nó quyết định sự đánh đổi giữa việc "cố gắng phân loại đúng tập Train" và "độ phẳng, rộng của đường biên".
  * **Nếu $C$ quá nhỏ:** Mô hình quá dễ dãi, chấp nhận nhiều sai sót $\rightarrow$ Ranh giới mờ nhạt $\rightarrow$ Bị Underfitting.
  * **Nếu $C$ quá lớn:** Mô hình phạt cực nặng mọi lỗi sai $\rightarrow$ Ranh giới bị uốn éo cục bộ theo từng điểm dữ liệu Train $\rightarrow$ Bị học vẹt (Overfitting), khi gặp dữ liệu thực tế thì dự đoán kém.
* **Tại sao lại chọn $C = 0.3$:**  
  Bảng kết quả thực nghiệm chứng minh rằng tại **$C = 0.3$**, mô hình đạt điểm 5-Fold CV Macro F1 cao nhất (**53.91%**, Rank 1).

---

## 6. THÀNH PHẦN 6: CÁC THANG ĐO ĐÁNH GIÁ (EVALUATION METRICS)
* **Tại sao không chỉ dùng Accuracy (Độ chính xác thông thường)?**  
  Vì dữ liệu bị mất cân bằng nhãn. Nếu một mô hình cứ đoán bừa vào nhãn chiếm số đông (*Enjoyment*) thì Accuracy vẫn có thể đạt ~30%, nhưng lại hoàn toàn "mù tịt" về các nhãn ít mẫu (*Surprise, Fear*).
* **Ý nghĩa các thang đo nhóm áp dụng:**
  * **Macro F1-score:** Tính điểm F1 riêng cho từng nhãn trong 7 nhãn rồi lấy trung bình cộng bình đẳng. Nhãn ít mẫu hay nhiều mẫu đều có tiếng nói như nhau. Đây là thước đo khắt khe và công bằng nhất.
  * **Weighted F1-score:** Tính F1 có nhân theo trọng số số lượng mẫu thực tế của từng nhãn.
  * **Per-class F1:** Báo cáo chi tiết điểm F1 riêng biệt của từng nhãn (giúp biết nhãn nào mô hình học tốt, nhãn nào còn yếu).
  * **Ma trận nhầm lẫn (Confusion Matrix):** Bảng trực quan hóa cho biết mô hình hay bị nhầm lẫn giữa cặp cảm xúc nào (ví dụ: *Anger* hay bị nhầm sang *Disgust*).

---

## 7. ĐỐI CHIẾU VỚI BÀI BÁO KHOA HỌC GỐC (HO ET AL., 2020)

Trong **Bảng 8 (Table 8)** của bài báo gốc UIT-VSMEC:
* **Mô hình SVM của nhóm tác giả trường UIT (`SVM + BoW`):**
  * Accuracy: **58.00%** | Weighted F1: **56.87%**
* **Mô hình Deep Learning tốt nhất toàn bài báo UIT (`CNN + word2Vec`):**
  * Accuracy: **59.74%** | Weighted F1: **59.74%**
* **Mô hình của Nhóm 12:**
  * **Giao thức 1 (Train-Only: 5,548 mẫu — So sánh chuẩn 100% với bài báo):**
    * Accuracy: **60.89%** | Weighted F1: **60.92%** | Macro F1: **57.75%**
    * Vượt xa `SVM + BoW` của UIT (**+4.05% WF1**) và vượt cả mạng nơ-ron `CNN + word2Vec` (**+1.18% WF1**).
  * **Giao thức 2 (Train + Val: 6,234 mẫu — Mở rộng tối ưu sau khi chọn $C=0.3$):**
    * Accuracy: **61.04%** | Weighted F1: **61.16%** | Macro F1: **58.24%**

**Lý do mô hình của nhóm bạn đạt kết quả cao hơn:**
1. Tiền xử lý chuyên sâu bảo tồn trọn vẹn tín hiệu cảm xúc mạng xã hội: Emoji/Emoticon, dấu câu biểu cảm (`token_than`, `token_hoi`), xử lý bẫy tiếng cười `kkk` trước khi rút gọn lặp, và tách dấu câu trước khi chuẩn hóa teencode.
2. Kiến trúc đặc trưng kép `FeatureUnion` kết hợp Word TF-IDF `(1, 2)` và Character n-grams trong ranh giới từ `char_wb (2, 5)` cùng hàm làm mượt logarit `sublinear_tf=True` và lọc nhiễu `min_df=2`.
3. Đặt toàn bộ vào `Pipeline` chống Data Leakage kết hợp cơ chế cân bằng lớp `class_weight='balanced'` và tinh chỉnh siêu tham số $C=0.3$ tối ưu theo `f1_macro`.
