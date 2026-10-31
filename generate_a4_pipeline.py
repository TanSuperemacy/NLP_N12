import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pymupdf

# Register TrueType Fonts with full Vietnamese Unicode support
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', 'C:/Windows/Fonts/ariali.ttf'))

def create_pipeline_pdf(filename="pipeline_A4_Nhom12.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4 # 595.28 x 841.89 pt
    
    # Clean professional light background
    c.setFillColor(colors.HexColor('#F8FAFC'))
    c.rect(0, 0, width, height, fill=True, stroke=False)
    
    margin_x = 26
    content_w = width - 2 * margin_x # 543.28 pt
    top_y = height - 20
    
    # -------------------------------------------------------------
    # 1. HEADER SECTION (Navy Gradient Style Card)
    # -------------------------------------------------------------
    header_h = 86
    header_y = top_y - header_h
    
    # Header card background
    c.setFillColor(colors.HexColor('#0F172A')) # Deep Slate / Navy
    c.roundRect(margin_x, header_y, content_w, header_h, 8, fill=True, stroke=False)
    
    # Top Institution / Course Banner
    c.setFillColor(colors.HexColor('#94A3B8'))
    c.setFont('Arial-Bold', 8)
    c.drawString(margin_x + 16, header_y + header_h - 18, "TRƯỜNG ĐH GIAO THÔNG VẬN TẢI - PHÂN HIỆU TP.HCM (UTC2)  •  KHOA CNTT  •  LỚP CQ.64.CN.CNTT")
    
    # Main Title
    c.setFillColor(colors.white)
    c.setFont('Arial-Bold', 14)
    c.drawString(margin_x + 16, header_y + header_h - 38, "SƠ ĐỒ PIPELINE: NHẬN DIỆN CẢM XÚC TIẾNG VIỆT (UIT-VSMEC)")
    
    # Subtitle / Model Info
    c.setFillColor(colors.HexColor('#E2E8F0'))
    c.setFont('Arial', 9.2)
    c.drawString(margin_x + 16, header_y + header_h - 55, "Học phần: Nhập môn Xử lý Ngôn ngữ Tự nhiên (NLP)  |  Mô hình: TF-IDF + Linear SVM (GridSearchCV C)")
    
    # Team Info Tag
    c.setFillColor(colors.HexColor('#38BDF8')) # Sky Blue Accent
    c.setFont('Arial-Bold', 8.8)
    c.drawString(margin_x + 16, header_y + header_h - 73, "Nhóm thực hiện: NHÓM 12  (Thành viên: Lê Anh Pha, Lê Nguyễn Huy Phát, Nguyễn Ngọc Quý, Âu Dương Tấn, Trần Minh Tấn)")
    
    # -------------------------------------------------------------
    # Helper to draw a workflow Stage Box
    # -------------------------------------------------------------
    def draw_stage(y_top, box_h, stage_badge, title, bg_color, border_color, items):
        y_bottom = y_top - box_h
        
        # Main card background
        c.setFillColor(colors.HexColor(bg_color))
        c.setStrokeColor(colors.HexColor(border_color))
        c.setLineWidth(1.3)
        c.roundRect(margin_x, y_bottom, content_w, box_h, 6, fill=True, stroke=True)
        
        # Header banner inside box
        header_bar_h = 24
        c.setFillColor(colors.HexColor(border_color))
        c.roundRect(margin_x, y_top - header_bar_h, content_w, header_bar_h, 6, fill=True, stroke=False)
        c.rect(margin_x, y_top - header_bar_h, content_w, 6, fill=True, stroke=False)
        
        # Badge + Title text
        c.setFillColor(colors.white)
        c.setFont('Arial-Bold', 10)
        c.drawString(margin_x + 14, y_top - 16, f"[{stage_badge}]  {title}")
        
        # Items / Bullet points inside
        c.setFillColor(colors.HexColor('#0F172A'))
        text_y = y_top - header_bar_h - 16
        
        for item in items:
            lead, detail = item
            c.setFont('Arial-Bold', 9)
            lead_str = f"• {lead}: "
            c.drawString(margin_x + 16, text_y, lead_str)
            lead_w = c.stringWidth(lead_str, 'Arial-Bold', 9)
            
            c.setFont('Arial', 9)
            c.drawString(margin_x + 16 + lead_w, text_y, detail)
            text_y -= 17.0
            
        return y_bottom

    def draw_arrow(y_pos):
        # Downward connecting arrow with styled head
        c.setFillColor(colors.HexColor('#64748B'))
        c.setStrokeColor(colors.HexColor('#64748B'))
        c.setLineWidth(1.8)
        arrow_w = 4.5
        # vertical line
        c.line(width / 2, y_pos, width / 2, y_pos - 12)
        # arrowhead
        path = c.beginPath()
        path.moveTo(width / 2 - arrow_w, y_pos - 11)
        path.lineTo(width / 2 + arrow_w, y_pos - 11)
        path.lineTo(width / 2, y_pos - 18)
        path.close()
        c.drawPath(path, fill=True, stroke=False)
        return y_pos - 20

    current_y = header_y - 14
    
    # -------------------------------------------------------------
    # STAGE 1: INPUT DATA
    # -------------------------------------------------------------
    items_1 = [
        ("Nguồn ngữ liệu", "Bộ dữ liệu UIT-VSMEC (Vietnamese Social Media Emotion Corpus) gồm 6,927 bình luận Facebook."),
        ("Phân chia chuẩn", "Train: 5,548 mẫu (80%)  |  Validation: 686 mẫu (10%)  |  Test: 693 mẫu (10%) (Giữ nguyên phân chia gốc)."),
        ("Không gian 7 nhãn", "Enjoyment (1,558), Disgust (1,071), Other (1,021), Sadness (947), Anger (391), Fear (318), Surprise (242).")
    ]
    current_y = draw_stage(current_y, 85, "GIAI ĐOẠN 1", "DỮ LIỆU ĐẦU VÀO (UIT-VSMEC CORPUS)", 
                           '#F0F9FF', '#0284C7', items_1)
    
    current_y = draw_arrow(current_y)
    
    # -------------------------------------------------------------
    # STAGE 2: PREPROCESSING (6 BƯỚC THEO ĐÚNG THỨ TỰ TỐI ƯU ĐÃ CHẠY)
    # -------------------------------------------------------------
    items_2 = [
        ("B1. Chuẩn hóa Unicode NFC", "Khắc phục triệt để lỗi phân mảnh dấu thanh tổ hợp tiếng Việt trên mạng xã hội (VD: thi ́ ch -> thích)."),
        ("B2. Làm sạch cơ bản", "Loại bỏ URL rác (http/www...), ký tự điều khiển lạ và chuyển toàn bộ văn bản về chữ thường (lowercase)."),
        ("B3. Bảo tồn Emoji & Emoticon", "Ánh xạ icon/emoji thành token cảm xúc (:)) -> icon_vui; 😭, :(( -> emoji_buon; 😡 -> emoji_tuc_gian...)."),
        ("B4. Rút gọn ký tự lặp (chạy trước)", "Thu gọn ký tự kéo dài nhấn mạnh về dạng chuẩn (quáaaaa -> quá, đẹpppp -> đẹp, waaa -> wa)."),
        ("B5. Chuẩn hóa Teencode (chạy sau)", "Ánh xạ từ viết tắt MXH về chuẩn (wa -> quá, ko -> không, dc -> được, j -> gì, cx -> cũng, bik -> biết...)."),
        ("B6. Tách từ ghép (PyVi)", "Phân định ranh giới từ tiếng Việt bằng ViTokenizer (thời buổi -> thời_buổi, nước mắt -> nước_mắt).")
    ]
    current_y = draw_stage(current_y, 136, "GIAI ĐOẠN 2", "TIỀN XỬ LÝ VĂN BẢN CHUYÊN SÂU (TEXT PREPROCESSING PIPELINE)", 
                           '#FFFBEB', '#D97706', items_2)
    
    current_y = draw_arrow(current_y)
    
    # -------------------------------------------------------------
    # STAGE 3: FEATURE EXTRACTION (TF-IDF)
    # -------------------------------------------------------------
    items_3 = [
        ("Mô hình biểu diễn", "TF-IDF Vectorizer (Term Frequency - Inverse Document Frequency) trích xuất trọng số thống kê ngữ nghĩa."),
        ("Cấu hình N-gram", "ngram_range = (1, 2) (bắt trọn cả từ đơn lẫn cặp 2 từ liền kề mang nghĩa phủ định: 'không thích', 'rất buồn')."),
        ("Tham số tối ưu hóa", "max_features = 10,000 đặc trưng nổi bật nhất  |  sublinear_tf = True (áp dụng 1 + log(tf) làm mượt tần số từ)."),
        ("Chống rò rỉ dữ liệu (No Leakage)", "fit_transform CHỈ áp dụng trên tập Train; tập Validation và Test CHỈ áp dụng transform.")
    ]
    current_y = draw_stage(current_y, 102, "GIAI ĐOẠN 3", "TRÍCH XUẤT ĐẶC TRƯNG TỪ VỰNG (TF-IDF VECTORIZATION)", 
                           '#FAF5FF', '#7C3AED', items_3)
    
    current_y = draw_arrow(current_y)
    
    # -------------------------------------------------------------
    # STAGE 4: MODEL TRAINING & TUNING (Linear SVM + GridSearchCV)
    # -------------------------------------------------------------
    items_4 = [
        ("Thuật toán phân loại", "Linear Support Vector Classification (LinearSVC) - Tìm siêu phẳng tối ưu lồi trong không gian 10,000 chiều."),
        ("Xử lý lệch nhãn", "class_weight = 'balanced' (tự động cân chỉnh trọng số phạt nghịch đảo tần suất lớp để hỗ trợ các nhãn ít mẫu)."),
        ("Tinh chỉnh tham số (GridSearch)", "Thực nghiệm 5-Fold Stratified Cross-Validation trên tập Train với dải tham số C ∈ [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]."),
        ("Tiêu chí tối ưu & Kết quả", "Mục tiêu tối ưu: scoring = 'f1_macro'. Kết quả tìm ra siêu tham số tối ưu nhất: C = 0.3 (Rank 1, CV Macro F1: 52.43%).")
    ]
    current_y = draw_stage(current_y, 102, "GIAI ĐOẠN 4", "HUẤN LUYỆN & TINH CHỈNH MÔ HÌNH (LINEAR SVM & GRIDSEARCHCV)", 
                           '#F0FDF4', '#16A34A', items_4)
    
    current_y = draw_arrow(current_y)
    
    # -------------------------------------------------------------
    # STAGE 5: EVALUATION & INFERENCE
    # -------------------------------------------------------------
    items_5 = [
        ("Hiệu năng tập Kiểm thử (Test Set)", "Accuracy: 58.87% (408/693 mẫu)  |  Macro F1-Score: 55.97%  |  Weighted F1-Score: 58.88%."),
        ("Hiệu năng tập Phát triển (Val Set)", "Accuracy: 55.39%  |  Macro F1-Score: 51.81%  |  Weighted F1-Score: 55.54%."),
        ("Hiệu năng từng nhãn (Test F1)", "Enjoyment: 68.21% | Sadness: 66.09% | Fear: 62.26% | Other: 52.94% | Surprise: 52.05% | Disgust: 51.56% | Anger: 38.71%."),
        ("Sản phẩm bàn giao hoàn chỉnh", "predictions.csv (693 dòng kết quả), confusion_matrix.png, c_comparison_results.csv, UIT_VSMEC_Nhom12.ipynb.")
    ]
    current_y = draw_stage(current_y, 102, "GIAI ĐOẠN 5", "ĐÁNH GIÁ MÔ HÌNH & XUẤT KẾT QUẢ DỰ ĐOÁN (EVALUATION & INFERENCE)", 
                           '#FEF2F2', '#DC2626', items_5)
    
    # -------------------------------------------------------------
    # FOOTER COMPARISON SUMMARY BOX
    # -------------------------------------------------------------
    footer_box_h = 48
    footer_box_y = current_y - 12 - footer_box_h
    c.setFillColor(colors.HexColor('#EFF6FF'))
    c.setStrokeColor(colors.HexColor('#3B82F6'))
    c.setLineWidth(1.2)
    c.roundRect(margin_x, footer_box_y, content_w, footer_box_h, 5, fill=True, stroke=True)
    
    c.setFillColor(colors.HexColor('#1E40AF'))
    c.setFont('Arial-Bold', 9)
    c.drawString(margin_x + 12, footer_box_y + footer_box_h - 15, "★ ĐỐI SÁNH KHOA HỌC VỚI BÀI BÁO GỐC (HO ET AL., PACLING 2019 / SPRINGER 2020):")
    
    c.setFont('Arial', 8.5)
    c.setFillColor(colors.HexColor('#1E3A8A'))
    c.drawString(margin_x + 12, footer_box_y + footer_box_h - 29, "• SVM Bài báo gốc (BoW): Accuracy 58.00%, Weighted F1 56.87%  ->  Pipeline Nhóm 12: Accuracy 58.87% (+0.87%), Weighted F1 58.88% (+2.01%).")
    c.drawString(margin_x + 12, footer_box_y + footer_box_h - 42, "• Mô hình nhóm đạt hiệu năng xấp xỉ mạng nơ-ron sâu CNN (59.74%) và vượt trội mạng LSTM (53.30%) trong khi thời gian huấn luyện chỉ mất 2-3 giây.")
    
    # Bottom metadata line
    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont('Arial-Italic', 7.5)
    c.drawString(margin_x, 12, "Báo cáo tiến độ đồ án môn Nhập môn Xử lý Ngôn ngữ Tự nhiên • Nhóm 12 • Định dạng: 1 trang A4 tiêu chuẩn (210 x 297 mm)")
    c.drawRightString(width - margin_x, 12, "Hệ thống phân loại cảm xúc UIT-VSMEC • Học kỳ I (2026)")
    
    c.save()
    print(f"Created {filename} successfully! Final Y bottom = {footer_box_y}")

if __name__ == "__main__":
    create_pipeline_pdf()
