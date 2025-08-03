# KL-1: Vision Transformer for Fake/Real Shoe Classification

## Mô tả dự án
KL-1 là hệ thống nhận diện giày thật/giả sử dụng mô hình Vision Transformer (ViT) tự xây dựng, tối ưu cho dữ liệu giày Việt Nam. Dự án gồm backend FastAPI, frontend web đơn giản, và các công cụ phân tích giải thích bằng attention heatmap.

## Cấu hình mô hình
- Ảnh đầu vào: 224x224 RGB
- Patch: 16x16, 196 patch
- Số lớp Transformer: 12
- Số đầu attention: 8
- Kích thước nhúng: 512
- MLP ratio: 4.0
- Dropout: 0.15
- DropPath: 0.1
- Số lớp phân loại: 2 (Real/Fake)

## Tham số huấn luyện
- Batch size: 32
- Epoch: 50
- Optimizer: AdamW, weight decay 0.05
- Learning rate: 0.0003 (CosineAnnealingLR)
- Early stopping: 7 epoch không cải thiện
- Loss: CrossEntropyLoss
- Class balancing: WeightedRandomSampler
- Mixed Precision: torch.amp.autocast, GradScaler

## Sử dụng
1. **Cài đặt Python >=3.10 và các thư viện:**
   - pip install -r config/requirements.txt
2. **Khởi động API:**
   - python src/api.py
3. **Truy cập giao diện web:**
   - Mở trình duyệt tại http://127.0.0.1:8000
4. **Upload ảnh để phân loại và xem heatmap:**
   - Ảnh kết quả và heatmap sẽ lưu tại thư mục uploads/

## Cấu trúc thư mục
- src/: Mã nguồn backend, mô hình, phân tích
- web/: Giao diện web
- uploads/: Ảnh kết quả và heatmap
- results/: Model checkpoint, log, biểu đồ huấn luyện
- config/: Cấu hình, requirements

## Kết quả & lưu ý
- Accuracy trên tập test: ~70%
- Attention heatmap trực quan hóa vùng ảnh quan trọng
- Yêu cầu GPU để huấn luyện, có thể chạy inference trên CPU

## Tác giả & liên hệ
- Vũ Trọng Phú
- Email: troqphu13@gmail.com

---
**Lưu ý:** Dự án này chỉ dùng cho mục đích nghiên cứu, không khuyến nghị sử dụng cho các hệ thống kiểm định hàng hóa thực tế nếu chưa kiểm thử kỹ lưỡng.
