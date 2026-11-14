# Day 06 · Data Ingestion Engineering Lab

**Học viên:** (Họ và tên học viên)  
**MSSV:** 2A202603743 · **Lớp:** L3B  
**Corpus:** *AI Engineering: Building Applications with Foundation Models* (Chip Huyen, O'Reilly Media 2025) — Chapters 1 & 2 (112 content pages, 224,576 characters).

---

## 1. Giới thiệu tổng quan

Kho lưu trữ này triển khai pipeline kỹ thuật tiền xử lý dữ liệu (**Data Ingestion Pipeline**) cho AI Agent, thực hiện so sánh có kiểm soát **3 chiến lược (A, B, C)** trên cùng tập dữ liệu, mô hình nhúng và bộ câu hỏi đánh giá:
* **Strategy A:** Fixed-size chunking (500 ký tự, overlap 50), không áp dụng filter khi tìm kiếm.
* **Strategy B:** Structure-aware / Paragraph chunking ($\le 800$ ký tự), bảo toàn ngữ cảnh đoạn văn, không filter.
* **Strategy C:** Chunking giống hệt Strategy B (349 chunks), áp dụng **Metadata Filtering** (`chapter` và `section`) trên Qdrant Cloud cho các câu hỏi có ngữ cảnh lọc.

Hạ tầng sử dụng:
* **Embedding Model:** OpenAI API `text-embedding-3-small` (1536 chiều).
* **Vector Store:** **Qdrant Cloud** (Cluster quản lý đám mây, metric Cosine, payload index cho metadata).

---

## 2. Cấu trúc thư mục

```text
├── STUDENT.md                         # Thông tin học viên và commit
├── README.md                          # Hướng dẫn setup, cấu hình và lệnh chạy
├── requirements.txt                   # Danh sách package python
├── .env.example                       # Mẫu cấu hình API key
├── data/
│   ├── manifest.json                  # Mô tả nguồn, quyền, số trang (112 trang), ký tự
│   └── Chip Huyen - AI Engineering... # File PDF gốc
├── benchmark/
│   ├── questions.json                 # 20 câu hỏi (15 dev / 5 test) kèm gold evidence
│   └── selection_criteria.md          # Tiêu chí chọn cấu hình chốt trước khi xem test
├── starter/src/                       # Pipeline hoàn chỉnh (ingestion, chunking, embeddings, store, evaluation, run)
├── src/                               # Bản đồng bộ pipeline tại root
├── artifacts/
│   └── experiment_log.csv             # Bảng log chi tiết kết quả thực nghiệm A/B/C
├── results/
│   ├── traces_strategy_A.json         # Top-3 trace và nhãn hit cho Strategy A
│   ├── traces_strategy_B.json         # Top-3 trace và nhãn hit cho Strategy B
│   ├── traces_strategy_C.json         # Top-3 trace và nhãn hit cho Strategy C
│   ├── summary_metrics.json           # Tóm tắt số liệu Hit@1 và Evidence Hit@3
│   ├── manual_audit.json              # Dữ liệu chấm thủ công 5 khía cạnh
│   └── manual_audit.md                # Bảng chấm thủ công Coherence, Metadata, Grounding
└── report/individual/
    └── 2A202603743_HoTen.md           # Báo cáo cá nhân chi tiết theo mẫu chuẩn
```

---

## 3. Hướng dẫn thiết lập và chạy tái lập kết quả

### Bước 1: Kích hoạt môi trường ảo
```powershell
# Kích hoạt venv (PowerShell)
.venv\Scripts\Activate.ps1

# Hoặc cài đặt dependencies nếu tạo môi trường mới:
pip install -r requirements.txt
```

### Bước 2: Cấu hình biến môi trường (`.env`)
Tạo file `.env` từ `.env.example`:
```powershell
Copy-Item .env.example .env
```
Điền các giá trị vào file `.env`:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
```

### Bước 3: Kiểm tra kết nối hạ tầng
```powershell
python scripts/check_setup.py
```
*Kỳ vọng:* Hiện `[SUCCESS] All systems verified!`.

### Bước 4: Chạy toàn bộ Pipeline 1-Click
```powershell
python -m starter.src.run
```
Lệnh trên sẽ tự động:
1. Đọc và chuẩn hóa 112 trang nội dung kỹ thuật từ sách của Chip Huyen.
2. Sinh chunk cho cả 3 chiến lược A (539 chunks), B (349 chunks), C (349 chunks).
3. Tạo vector embedding qua OpenAI API và nạp vào 3 collection trên Qdrant Cloud.
4. Chạy truy xuất Top-3 cho 20 câu hỏi benchmark (15 Dev, 5 Test).
5. Tính toán Hit@1, Evidence Hit@3, tỷ lệ tuân thủ filter và xuất toàn bộ traces ra thư mục `results/`.
6. Cập nhật số liệu vào file `artifacts/experiment_log.csv`.

---

## 4. Tóm tắt kết quả thực nghiệm

| Strategy | Dev Hit@1 | Dev Hit@3 | Test Hit@1 | Test Hit@3 | Chunk Count | Coherence TB | Filter Compliance | Grounding TB |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A (Fixed)** | 9/15 (60.0%) | 10/15 (66.67%) | 5/5 (100.0%) | 5/5 (100.0%) | 539 | 0.80 / 2.0 | 1/2 (50.0%) | 1.80 / 2.0 |
| **B (Paragraph)** | 9/15 (60.0%) | **12/15 (80.0%)** | 4/5 (80.0%) | 5/5 (100.0%) | **349** | **2.00 / 2.0** | 1/2 (50.0%) | **2.00 / 2.0** |
| **C (Para + Filter)** | 9/15 (60.0%) | **12/15 (80.0%)** | 4/5 (80.0%) | 5/5 (100.0%) | **349** | **2.00 / 2.0** | **2/2 (100.0%)** | **2.00 / 2.0** |

*Chiến lược tối ưu được chọn:* **Strategy C** nhờ duy trì Hit@3 cao nhất (80.0%), tính mạch lạc trọn vẹn (2.0/2.0), tuân thủ 100% filter trên các câu hỏi có ngữ cảnh và tiết kiệm 35.2% dung lượng lưu trữ so với baseline.
