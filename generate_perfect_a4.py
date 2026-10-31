import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pymupdf

# Register TrueType Fonts with full Vietnamese Unicode support
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', 'C:/Windows/Fonts/ariali.ttf'))

def build_pdf(filename="pipeline.pdf", include_title=True):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4 # 595.28 x 841.89 pt
    
    # Pure White Background
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=True, stroke=False)
    
    margin_x = 30
    content_w = width - 2 * margin_x # 535.28 pt
    text_w = content_w - 24 # 511.28 pt
    
    item_style = ParagraphStyle(
        'ItemStyle',
        fontName='Arial',
        fontSize=8.8,
        leading=13.0,
        textColor=colors.black
    )
    
    stages_data = [
        {
            "badge": "BƯỚC 1",
            "title": "TÌM HIỂU & CHUẨN BỊ BỘ DỮ LIỆU (DATA EXPLORATION)",
            "items": [
                "<b>• Nguồn ngữ liệu gốc:</b> Bộ dữ liệu UIT-VSMEC (Vietnamese Social Media Emotion Corpus) gồm 6,927 bình luận Facebook.",
                "<b>• Phân chia phân tầng (Stratified Split):</b> Tỷ lệ chuẩn 80 - 10 - 10 đảm bảo đủ 7 nhãn ở cả 3 tập (Train: 5,548 | Val: 686 | Test: 693).",
                "<b>• Không gian nhãn & Phân bố:</b> 7 nhãn lệch mẫu nặng: Enjoyment (1,558), Disgust (1,071), Other (1,021), Sadness (947), Anger (391), Fear (318), Surprise (242)."
            ]
        },
        {
            "badge": "BƯỚC 2",
            "title": "TIỀN XỬ LÝ VĂN BẢN CHUYÊN SÂU (TEXT PREPROCESSING PIPELINE)",
            "items": [
                "<b>• B1. Chuẩn hóa Unicode NFC:</b> Khắc phục lỗi phân mảnh dấu thanh tổ hợp tiếng Việt trên MXH (VD: thi ́ ch -> thích).",
                "<b>• B2. Làm sạch cơ bản:</b> Loại bỏ URL rác (http/www...), ký tự điều khiển lạ và chuyển toàn bộ văn bản về chữ thường (lowercase).",
                "<b>• B3. Bảo tồn Emoji & Emoticon:</b> Ánh xạ icon/emoji thành token cảm xúc (:)) -> icon_vui; 😭, :(( -> emoji_buon; 😡 -> emoji_tuc_gian...).",
                "<b>• B4. Rút gọn ký tự lặp (chạy trước):</b> Thu gọn ký tự kéo dài nhấn mạnh về dạng chuẩn (quáaaaa -> quá, đẹpppp -> đẹp, waaa -> wa).",
                "<b>• B5. Chuẩn hóa Teencode (chạy sau):</b> Ánh xạ từ viết tắt MXH về chuẩn (wa -> quá, ko -> không, dc -> được, j -> gì, cx -> cũng, bik -> biết...).",
                "<b>• B6. Tách từ ghép tiếng Việt (PyVi):</b> Phân định ranh giới từ bằng ViTokenizer (thời buổi -> thời_buổi, nước mắt -> nước_mắt)."
            ]
        },
        {
            "badge": "BƯỚC 3",
            "title": "TRÍCH XUẤT ĐẶC TRƯNG TỪ VỰNG (TF-IDF VECTORIZATION)",
            "items": [
                "<b>• Mô hình biểu diễn:</b> TF-IDF Vectorizer (Term Frequency - Inverse Document Frequency) trích xuất đặc trưng thống kê từ ngữ.",
                "<b>• Cấu hình N-gram:</b> ngram_range = (1, 2) (bắt trọn cả từ đơn lẫn cặp 2 từ liền kề mang nghĩa phủ định: 'không thích', 'rất buồn').",
                "<b>• Tham số tối ưu hóa:</b> max_features = 10,000 đặc trưng quan trọng nhất  |  sublinear_tf = True (làm mượt tần số xuất hiện của từ).",
                "<b>• Chống rò rỉ dữ liệu (No Data Leakage):</b> fit_transform CHỈ áp dụng trên tập Train; tập Validation và Test CHỈ áp dụng transform."
            ]
        },
        {
            "badge": "BƯỚC 4",
            "title": "HUẤN LUYỆN & TINH CHỈNH MÔ HÌNH (MODEL TRAINING & GRIDSEARCHCV)",
            "items": [
                "<b>• Thuật toán phân loại:</b> Linear Support Vector Classification (LinearSVC) - Tìm siêu phẳng phân tách tối ưu lồi trong không gian 10,000 chiều.",
                "<b>• Cân bằng dữ liệu lệch nhãn:</b> class_weight = 'balanced' (tự động cân chỉnh trọng số phạt nghịch đảo tần suất lớp để hỗ trợ các nhãn ít mẫu).",
                "<b>• Xác định độ đo mục tiêu & Tinh chỉnh:</b> Thiết lập scoring = 'f1_macro' trong 5-Fold Stratified CV với dải C ∈ [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0].",
                "<b>• Kết quả tối ưu:</b> Tìm ra siêu tham số tối ưu nhất C = 0.3 (Rank 1, CV Macro F1: 52.43%) đạt độ cân bằng tốt nhất giữa bias và variance."
            ]
        },
        {
            "badge": "BƯỚC 5",
            "title": "ĐÁNH GIÁ MÔ HÌNH & HIỂN THỊ KẾT QUẢ (EVALUATION & VISUALIZATION)",
            "items": [
                "<b>• Kết quả độ đo (Test Set):</b> Accuracy: 58.87% (408/693 mẫu)  |  Macro F1-Score: 55.97%  |  Weighted F1-Score: 58.88%.",
                "<b>• Kết quả độ đo (Validation Set):</b> Accuracy: 55.39%  |  Macro F1-Score: 51.81%  |  Weighted F1-Score: 55.54%.",
                "<b>• F1-Score từng nhãn (Test Set):</b> Enjoyment: 68.21% | Sadness: 66.09% | Fear: 62.26% | Other: 52.94% | Surprise: 52.05% | Disgust: 51.56% | Anger: 38.71%.",
                "<b>• Trực quan hóa & Sản phẩm bàn giao:</b> Ma trận nhầm lẫn (confusion_matrix.png phân tích các cặp dễ nhầm lẫn như Anger-Disgust), predictions.csv."
            ]
        }
    ]
    
    measured_stages = []
    total_stages_h = 0
    header_bar_h = 22
    padding_v = 14
    item_gap = 4.5
    
    for s in stages_data:
        para_objs = []
        box_text_h = 0
        for text in s["items"]:
            p = Paragraph(text, item_style)
            w, h = p.wrap(text_w, 200)
            para_objs.append((p, w, h))
            box_text_h += h
        
        box_text_h += item_gap * (len(s["items"]) - 1)
        total_box_h = header_bar_h + padding_v + box_text_h
        measured_stages.append({
            "badge": s["badge"],
            "title": s["title"],
            "paras": para_objs,
            "total_h": total_box_h
        })
        total_stages_h += total_box_h
        
    num_arrows = len(stages_data) - 1
    arrow_h = 16
    total_flow_h = total_stages_h + num_arrows * arrow_h
    
    top_y = height - 26
    
    if include_title:
        # SINGLE CENTERED HEADER LINE (Guaranteed NO collision!)
        title_text = "PIPELINE: NHẬN DIỆN CẢM XÚC TIẾNG VIỆT (UIT-VSMEC) • TF-IDF + LINEAR SVM • NHÓM 12"
        c.setFillColor(colors.black)
        c.setFont('Arial-Bold', 10.5)
        c.drawCentredString(width / 2, top_y, title_text)
        
        # Subtle horizontal divider line
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.8)
        c.line(margin_x, top_y - 6, width - margin_x, top_y - 6)
        
        current_y = top_y - 20
    else:
        # Center the flow vertically on the page
        current_y = height - (height - total_flow_h) / 2
    
    for idx, stage in enumerate(measured_stages):
        box_h = stage["total_h"]
        y_top = current_y
        y_bottom = y_top - box_h
        
        # 1. Main Box Outline (Pure black border, pure white background)
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.black)
        c.setLineWidth(1.2)
        c.roundRect(margin_x, y_bottom, content_w, box_h, 4, fill=True, stroke=True)
        
        # 2. Header Bar (Solid Black with Crisp White Text)
        c.setFillColor(colors.black)
        c.roundRect(margin_x, y_top - header_bar_h, content_w, header_bar_h, 4, fill=True, stroke=False)
        c.rect(margin_x, y_top - header_bar_h, content_w, 4, fill=True, stroke=False)
        
        # Header text
        c.setFillColor(colors.white)
        c.setFont('Arial-Bold', 9.2)
        c.drawString(margin_x + 12, y_top - 15, f"{stage['badge']} • {stage['title']}")
        
        # 3. Draw Paragraphs inside (Auto-wrapped, zero overflow)
        cur_para_y = y_top - header_bar_h - 10
        for p, pw, ph in stage["paras"]:
            cur_para_y -= ph
            p.drawOn(c, margin_x + 12, cur_para_y)
            cur_para_y -= item_gap
            
        current_y = y_bottom
        
        # 4. Draw Connecting Arrow
        if idx < num_arrows:
            arrow_start_y = current_y
            arrow_end_y = current_y - arrow_h
            
            c.setStrokeColor(colors.black)
            c.setFillColor(colors.black)
            c.setLineWidth(1.5)
            
            # line
            c.line(width / 2, arrow_start_y, width / 2, arrow_end_y + 5)
            
            # arrowhead
            path = c.beginPath()
            path.moveTo(width / 2 - 4, arrow_end_y + 5)
            path.lineTo(width / 2 + 4, arrow_end_y + 5)
            path.lineTo(width / 2, arrow_end_y)
            path.close()
            c.drawPath(path, fill=True, stroke=False)
            
            current_y = arrow_end_y
            
    c.save()
    print(f"Generated {filename} successfully! Final Y bottom = {current_y:.1f}")

if __name__ == "__main__":
    build_pdf("pipeline.pdf", include_title=True)
    build_pdf("pipeline_A4_Nhom12.pdf", include_title=True)
