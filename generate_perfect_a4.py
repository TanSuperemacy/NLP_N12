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
        fontSize=8.4,
        leading=11.8,
        textColor=colors.black
    )
    
    stages_data = [
        {
            "badge": "BƯỚC 1",
            "title": "TÌM HIỂU & CHUẨN BỊ BỘ DỮ LIỆU (DATA EXPLORATION)",
            "items": [
                "<b>• Nguồn ngữ liệu gốc:</b> Bộ dữ liệu UIT-VSMEC (Vietnamese Social Media Emotion Corpus) gồm 6,927 bình luận Facebook (Null = 0, Duplicate = 0).",
                "<b>• Phân chia phân tầng (Stratified Split):</b> Tỷ lệ chuẩn 80 - 10 - 10 đảm bảo đủ 7 nhãn ở cả 3 tập (Train: 5,548 | Val: 686 | Test: 693).",
                "<b>• Không gian nhãn & Phân bố:</b> 7 nhãn lệch mẫu nặng: Enjoyment (1,558), Disgust (1,071), Other (1,021), Sadness (947), Anger (391), Fear (318), Surprise (242)."
            ]
        },
        {
            "badge": "BƯỚC 2",
            "title": "TIỀN XỬ LÝ VĂN BẢN CHUYÊN SÂU (TEXT PREPROCESSING PIPELINE)",
            "items": [
                "<b>• B1. Chuẩn hóa Unicode NFC & Làm sạch:</b> Khắc phục lỗi phân mảnh dấu thanh tổ hợp tiếng Việt (thi ́ ch -> thích) và xóa URL rác (http/www...).",
                "<b>• B2. Bảo tồn Emoji & Emoticon:</b> Ánh xạ icon/emoji thành token ngữ nghĩa (:)) -> icon_vui; 😭, :(( -> emoji_buon; 😡 -> emoji_tuc_gian...).",
                "<b>• B3. Bắt tiếng cười/khóc TRƯỚC khi rút gọn lặp:</b> Ánh xạ kkk+, haha+, hihi+ -> icon_vui/emoji_vui trước để tránh lỗi kkk -> k -> không (phủ định).",
                "<b>• B4. Bảo tồn dấu câu cảm xúc & Tách dấu câu:</b> Ánh xạ !+ -> token_than, ?+ -> token_hoi (hỗ trợ Surprise/Anger); tách dấu câu trước khi tra từ điển.",
                "<b>• B5. Rút gọn ký tự lặp & Chuẩn hóa Teencode:</b> Thu gọn từ kéo dài (đẹpppp -> đẹp) và chuẩn hóa chính tả khách quan (ko, -> không, dc. -> được, vl -> vãi).",
                "<b>• B6. Tách từ ghép tiếng Việt (PyVi):</b> Phân định ranh giới từ bằng ViTokenizer (thời buổi -> thời_buổi, nước mắt -> nước_mắt)."
            ]
        },
        {
            "badge": "BƯỚC 3",
            "title": "TRÍCH XUẤT ĐẶC TRƯNG KÉP (FEATUREUNION: WORD + CHAR_WB TF-IDF)",
            "items": [
                "<b>• Kiến trúc FeatureUnion (tương tự Mục 5.2 bài báo gốc UIT):</b> Kết hợp song song đặc trưng từ vựng (Word TF-IDF) và hình thái ký tự (Char_wb TF-IDF).",
                "<b>• Nhánh 1 - Word TF-IDF:</b> ngram_range = (1, 2) bắt từ đơn và cụm phủ định ('không_thích', 'quá_buồn')  |  min_df = 2, max_df = 0.9  |  sublinear_tf = True.",
                "<b>• Nhánh 2 - Char_wb TF-IDF:</b> analyzer = 'char_wb', ngram_range = (2, 5) bắt biến thể teencode/lỗi chính tả trong từ  |  min_df = 2  |  sublinear_tf = True.",
                "<b>• Chống rò rỉ dữ liệu (Zero Data Leakage):</b> Đóng gói FeatureUnion và LinearSVC vào sklearn.pipeline.Pipeline (chỉ fit trên fold Train khi CV)."
            ]
        },
        {
            "badge": "BƯỚC 4",
            "title": "HUẤN LUYỆN & TINH CHỈNH MÔ HÌNH (MODEL TRAINING & GRIDSEARCHCV)",
            "items": [
                "<b>• Thuật toán phân loại:</b> Linear Support Vector Classification (LinearSVC, random_state = 42, max_iter = 5000).",
                "<b>• Cân bằng dữ liệu lệch nhãn:</b> class_weight = 'balanced' (tự động cân chỉnh trọng số phạt nghịch đảo tần suất lớp giúp tăng vọt F1 ở các nhãn thiểu số).",
                "<b>• Xác định độ đo mục tiêu & Tinh chỉnh:</b> Thiết lập scoring = 'f1_macro' trong 5-Fold StratifiedKFold với dải C ∈ [0.05, 0.1, 0.2, 0.3, 0.5, 1.0].",
                "<b>• Kết quả tối ưu:</b> Tìm ra siêu tham số tối ưu C = 0.3 (Rank 1, Best 5-Fold CV Macro F1 = 53.91%) vừa chống học vẹt (overfit) vừa chống underfit."
            ]
        },
        {
            "badge": "BƯỚC 5",
            "title": "ĐÁNH GIÁ MÔ HÌNH & HIỂN THỊ KẾT QUẢ (EVALUATION & VISUALIZATION)",
            "items": [
                "<b>• Giao thức 1 (Train-Only 5,548 mẫu - Chuẩn UIT):</b> Test Acc: 60.89% | Macro F1: 57.75% | Weighted F1: 60.92% (Vượt SVM UIT +4.05% & CNN UIT +1.18%).",
                "<b>• Giao thức 2 (Train + Val 6,234 mẫu - Mở rộng):</b> Test Acc: 61.04% | Macro F1: 58.24% | Weighted F1: 61.16% (Bứt phá +1.42% WF1 so với CNN UIT).",
                "<b>• Per-Class F1 (Test - Train-Only):</b> Enjoyment: 69.61% | Sadness: 67.83% | Fear: 64.00% | Disgust: 56.83% | Surprise: 56.76% | Other: 53.94% | Anger: 35.29%.",
                "<b>• Trực quan hóa & Bàn giao:</b> Bảng Ablation Study bóc tách đóng góp, Confusion Matrix (confusion_matrix.png), file dự đoán predictions.csv (693 dòng)."
            ]
        }
    ]
    
    measured_stages = []
    total_stages_h = 0
    header_bar_h = 20
    padding_v = 11
    item_gap = 3.5
    
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
    arrow_h = 12
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
