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
  3. **Rút gọn ký tự lặp kéo dài:**  
     * Người dùng mạng xã hội hay gõ kéo dài để nhấn mạnh: `quáaaaa` $\rightarrow$ `quá`, `đẹpppp` $\rightarrow$ `đẹp`, `waaa` $\rightarrow$ `wa`. Rút gọn giúp máy không bị xem các từ này là từ mới.
  4. **Chuẩn hóa Teencode & Từ viết tắt:**  
     * Chuyển các từ mạng xã hội phổ biến về dạng chuẩn (`ko` $\rightarrow$ `không`, `dc` $\rightarrow$ `được`, `wa` $\rightarrow$ `quá`, `j` $\rightarrow$ `gì`...). Giúp gom tần số xuất hiện của từ về một mối, tránh phân mảnh từ vựng.
  5. **Tách từ tiếng Việt bằng PyVi (Word Segmentation):**  
     * Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không trùng với dấu cách. Ví dụ: `nước mắt` nếu không ghép sẽ bị hiểu thành `nước` (chất lỏng) và `mắt` (thị giác). Thư viện `pyvi` giúp ghép thành `thời_buổi`, `nước_mắt`, mang ý nghĩa cảm xúc trọn vẹn.

---

## 3. THÀNH PHẦN 3: TRÍCH XUẤT ĐẶC TRƯNG BẰNG TF-IDF
* **Nó là gì?**  
  Cầu nối toán học chuyển đổi **Văn bản (Text) $\rightarrow$ Ma trận số học (Vectors)**.
* **Mục đích:**  
  Các thuật toán máy học không đọc được chữ, chỉ tính toán được trên ma trận số.
* **Tại sao lại dùng TF-IDF kết hợp N-gram (1, 2) và Sublinear TF:**
  * **TF (Term Frequency):** Đo lường tần suất xuất hiện của từ trong một câu. Từ nào xuất hiện nhiều thì quan trọng với câu đó.
  * **IDF (Inverse Document Frequency):** Dìm điểm các từ phổ thông xuất hiện ở hầu hết mọi câu (như *thì, là, mà, của, bị*), giúp mô hình tập trung vào các từ mang tính phân loại cao.
  * **N-gram (1, 2):** Cực kỳ quan trọng để bắt **cụm 2 từ đi liền nhau**, đặc biệt là **từ phủ định** (bắt được trọn vẹn cụm `không_thích`, `quá_buồn` thay vì chỉ đọc từ đơn `thích` hay `buồn`).
  * **Sublinear TF (`1 + log(tf)`):** Tránh việc một từ lặp lại 10 lần thì có trọng số gấp 10 lần (hàm logarit phản ánh đúng mức độ nhấn mạnh thực tế hơn).
  * **Nguyên tắc chống rò rỉ dữ liệu (Data Leakage):** Chỉ `fit_transform` trên tập Train, tập Val và Test chỉ được `transform`.

---

## 4. THÀNH PHẦN 4: THUẬT TOÁN LINEAR SVM (LINEAR SUPPORT VECTOR CLASSIFIER)
* **Nó là gì?**  
  Mô hình máy học tìm các **siêu phẳng (hyperplane)** tối ưu trong không gian 10,000 chiều để phân tách 7 lớp cảm xúc với lề (margin) rộng nhất.
* **Mục đích:**  
  Dựa vào tọa độ vector số học của câu bình luận để đưa ra quyết định: câu này thuộc nhãn cảm xúc nào.
* **Tại sao lại dùng Linear SVM:**
  1. Sau bước TF-IDF, dữ liệu có số chiều rất lớn (10,000 chiều). Trong không gian nhiều chiều, dữ liệu hầu như đã phân tách tuyến tính được. Dùng **Linear SVM** chạy siêu nhanh (chỉ mất vài giây), hội tụ ổn định và tránh bị học vẹt (overfitting) so với nhân phi tuyến (RBF).
  2. Kích hoạt cơ chế **`class_weight='balanced'`**: Tập UIT-VSMEC bị lệch nhãn nặng (*Enjoyment* có 1,558 câu nhưng *Surprise* chỉ có 242 câu). Tham số này tự động phạt nặng hơn nếu mô hình đoán sai ở các nhãn ít mẫu, giúp cân bằng độ chính xác cho toàn bộ 7 nhãn.

---

## 5. THÀNH PHẦN 5: TINH CHỈNH SIÊU THAM SỐ $C$ BẰNG GRIDSEARCHCV
*(Kết quả lưu tại: `c_comparison_results.csv`)*

* **Nó là gì?**  
  Thực nghiệm tự động quét qua các giá trị $C \in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]$ bằng phương pháp **5-Fold Cross Validation** trên tập Train để tìm ra giá trị $C$ tối ưu nhất.
* **Ý nghĩa bản chất của tham số $C$:**
  * $C$ là **hệ số phạt khi mô hình đoán sai lề**. Nó quyết định sự đánh đổi giữa việc "cố gắng phân loại đúng tập Train" và "độ phẳng, rộng của đường biên".
  * **Nếu $C$ quá nhỏ:** Mô hình quá dễ dãi, chấp nhận nhiều sai sót $\rightarrow$ Ranh giới mờ nhạt $\rightarrow$ Bị Underfitting.
  * **Nếu $C$ quá lớn:** Mô hình phạt cực nặng mọi lỗi sai $\rightarrow$ Ranh giới bị uốn éo cục bộ theo từng điểm dữ liệu Train $\rightarrow$ Bị học vẹt (Overfitting), khi gặp dữ liệu thực tế thì dự đoán kém.
* **Tại sao lại chọn $C = 0.3$:**  
  Bảng kết quả thực nghiệm chứng minh rằng tại **$C = 0.3$**, mô hình đạt điểm F1 cao nhất (Rank 1). Khi tăng $C$ lên 1.0, 1.5, 2.0 thì điểm F1 bị tụt dần do hiện tượng Overfitting.

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
* **Mô hình SVM của nhóm tác giả trường UIT:**
  * Accuracy: **58.00%**
  * Weighted F1: **56.87%**
* **Mô hình của Nhóm 12:**
  * Accuracy: **~58.87%** (Vượt bài báo gốc)
  * Weighted F1: **~58.88%** (Vượt bài báo gốc 2.0%)
  * Thậm chí kết quả này còn cao hơn cả mạng nơ-ron sâu **LSTM** trong bài báo (LSTM chỉ đạt 53.30%).

**Lý do mô hình của nhóm bạn đạt kết quả cao hơn:**
1. Bài báo gốc giữ nguyên văn bản thô, không chuẩn hóa teencode và không bảo tồn emoji. Nhóm 12 có Pipeline tiền xử lý chuyên sâu cho tiếng Việt mạng xã hội.
2. Nhóm 12 áp dụng cơ chế `class_weight='balanced'` để xử lý bài toán mất cân bằng nhãn.
3. Không gian TF-IDF (1, 2) cô đọng 10,000 đặc trưng, sạch hơn và ít nhiễu hơn so với Bag-of-Characters 1–7 của bài báo gốc.
