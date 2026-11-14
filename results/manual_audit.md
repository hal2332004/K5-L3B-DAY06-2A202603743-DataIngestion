# Bảng Chấm Thủ Công 5 Khía Cạnh (Manual Audit Report)
*Thời gian chạy: 2026-11-14 | Người chấm: Tác giả bài làm (tự đánh giá theo GUIDE.md)*

---

## 1. Tính mạch lạc của Chunk (Chunk Coherence - 10 vị trí nguồn cố định)
*Thang điểm: 0 = bị cắt giữa ý/thiếu chủ thể; 1 = hiểu một phần/thiếu ngữ cảnh; 2 = tự hiểu được và giữ trọn ý.*
*Mẫu kiểm tra: 10 trang đầu tiên (Trang 25 đến Trang 34).*

| Vị trí | Trang | Nội dung / Chủ đề | Điểm A | Lý do chấm Strategy A (Fixed-size 500/50) | Điểm B/C | Lý do chấm Strategy B & C (Paragraph <=800) |
|:---:|:---:|---|:---:|---|:---:|---|
| **1** | 25 | Chapter 1 mở đầu & định vị | 1 | Cắt ngang câu mở đầu ở ký tự thứ 500, ngắt giữa câu ghép. | 2 | Giữ trọn tiêu đề và toàn bộ đoạn văn mở đầu chương 1. |
| **2** | 26 | Khái niệm Traditional ML vs Foundation Models | 1 | Cắt ngang footnote và tách định nghĩa khỏi ngữ cảnh so sánh. | 2 | Giữ nguyên vẹn đoạn văn định nghĩa và bước chuyển giao công nghệ. |
| **3** | 27 | Autoregressive generation & công thức chuỗi | 0 | Cắt ngay giữa công thức toán học và chỉ số chuỗi, mất ý nghĩa. | 2 | Giữ trọn công thức toán và câu văn giải thích đi kèm. |
| **4** | 28 | 3 lý do LM sử dụng Token thay vì từ/ký tự | 1 | Cắt ngang giữa điều 1 và điều 2 của danh sách liệt kê. | 2 | Gom trọn vẹn 3 ý liệt kê vào một semantic unit thống nhất. |
| **5** | 29 | Masked Language Models vs Autoregressive | 1 | Cắt đứt câu giải thích tại sao BERT ít phổ biến cho text generation. | 2 | Giữ trọn đoạn văn phân tích đối chiếu BERT và causal LM. |
| **6** | 30 | Đóng khung nhiều tác vụ thành Completion | 1 | Cắt ví dụ prompt "How are you in French is..." sang chunk kế tiếp. | 2 | Giữ trọn luận điểm, danh sách tác vụ và ví dụ dịch thuật. |
| **7** | 31 | Phân phối xác suất dự đoán token kế tiếp | 1 | Tách rời khái niệm Softmax khỏi phần giải thích từ vựng. | 2 | Toàn bộ đoạn văn cung cấp ngữ cảnh đầy đủ về phân phối xác suất. |
| **8** | 32 | Chuyển dịch sang Multimodal Foundation Models | 1 | Cắt ngang footnote 7 về trọng số mô hình (model weights). | 2 | Ranh giới đoạn văn bảo toàn trọn vẹn quá trình chuyển dịch multimodal. |
| **9** | 33 | Thuộc tính scaling dữ liệu của Self-supervision | 0 | Cắt lơ lửng ngay giữa từ kỹ thuật ("self-"), gây khó hiểu hoàn toàn. | 2 | Giữ nguyên vẹn luận điểm về self-supervision không cần nhãn thủ công. |
| **10** | 34 | Natural language supervision của OpenAI CLIP | 1 | Cắt đôi câu mô tả cách huấn luyện CLIP, citation nằm ở chunk sau. | 2 | Giữ trọn vẹn ngữ cảnh giải thích cặp ảnh-chữ không cần gán nhãn tay. |

* **Điểm Coherence trung bình:**
  * **Strategy A:** **0.80 / 2.0** (Nhiều lần cắt ngang câu, công thức và danh sách).
  * **Strategy B, C, D, E:** **2.00 / 2.0** (Các đoạn văn kỹ thuật được bảo toàn cấu trúc trọn vẹn; D bổ sung tiền tố ngữ cảnh; E bảo toàn bidirectional attention toàn trang).

---

## 2. Tính hữu dụng của Metadata (Metadata Usefulness)
*Đánh giá trên 5 câu Test × Top-3 chunks = 15 bản ghi.*

| Chỉ số | Strategy A (Fixed) | Strategy B (Paragraph) | Strategy C (Para + Filter) | Strategy D (Contextual) | Strategy E (Late Chunking) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Số bản ghi kiểm tra** | 15 / 15 | 15 / 15 | 15 / 15 | 15 / 15 | 15 / 15 |
| **Số bản ghi metadata ĐỦ và ĐÚNG** | **15 / 15 (100%)** | **15 / 15 (100%)** | **15 / 15 (100%)** | **15 / 15 (100%)** | **15 / 15 (100%)** |
| *• `source` và `page`* | Đầy đủ, truy vết đúng trang gốc | Đầy đủ, truy vết đúng trang gốc | Đầy đủ, truy vết đúng trang gốc | Đầy đủ, truy vết đúng trang gốc | Đầy đủ, truy vết đúng trang gốc |
| *• `chapter` (1 hoặc 2)* | Đúng 100% | Đúng 100% | Đúng 100% | Đúng 100% | Đúng 100% |
| *• `section` chuyên môn* | Đúng 100% | Đúng 100% | Đúng 100% | Đúng 100% | Đúng 100% |
| **Tỷ lệ tuân thủ Filter (2 câu Test có filter: q18, q20)** | **1 / 2 (50%)** | **1 / 2 (50%)** | **2 / 2 (100%)** | **2 / 2 (100%)** | **2 / 2 (100%)** |
| **Tỷ lệ tuân thủ trên TOÀN BỘ 5 câu có filter** | **2 / 5 (40%)** | **2 / 5 (40%)** | **5 / 5 (100%)** | **4 / 5 (80%)** | **3 / 5 (60%)** |

* **Nhận xét chuyên sâu:**
  * *Lợi ích:* Trường metadata `chapter` và `section` cho phép thu hẹp không gian tìm kiếm trên Qdrant Cloud. Ở Strategy C, câu `q20` không bị nhiễu bởi các đoạn nhắc đến "volunteers" hoặc "conversations" ở chương khác.
  * *Strategy D (Contextual):* Ngay cả khi không áp dụng filter cứng ở tầng cơ sở dữ liệu, việc bổ sung hierarchical prefix vào chunk text giúp tỷ lệ tuân thủ phân vùng nhảy vọt từ 40% (ở B) lên 80% (ở D)!
  * *Strategy E (Late Chunking):* Nhờ attention toàn trang, tỷ lệ tuân thủ filter tự nhiên đạt 60% mà không cần filter, vector chỉ 1024 chiều.

---

## 3. Chất lượng thông tin nền (Grounding Quality)
*Đánh giá trên 5 câu Test. Thang điểm: 0 = không có bằng chứng; 1 = có nhưng thiếu điều kiện/ngữ cảnh; 2 = đủ để trả lời và truy nguồn.*

| Câu Test | Nội dung câu hỏi | Điểm A | Lý do A | Điểm B | Lý do B | Điểm C | Lý do C |
|:---:|---|:---:|---|:---:|---|:---:|---|
| **q16** | Phương pháp giám sát huấn luyện CLIP? | 2 | Chunk chứa trọn vẹn cụm 'natural language supervision'. | 2 | Đủ ngữ cảnh giải thích biến thể của self-supervision. | 2 | Đủ bằng chứng hoàn hảo để trả lời. |
| **q17** | Use case GenAI phổ biến nhất trong khảo sát? | 2 | Chunk chứa rõ ràng 'coding is hands down the most popular'. | 2 | Đủ bằng chứng và ngữ cảnh khảo sát công nghiệp. | 2 | Đủ bằng chứng và ngữ cảnh. |
| **q18** | Tỷ lệ tiếng Anh trong Common Crawl? | 1 | Chunk cắt cụt bảng Table 2-1, mất tiêu đề cột (%) dẫn đến phải suy đoán. | 2 | Giữ trọn vẹn cả bảng Table 2-1 và tiêu đề cột %, rõ ràng 45.8786%. | 2 | Giữ trọn bảng và lọc đúng Chapter 2 Training Data. |
| **q19** | Test time compute nhắm vào mức sinh nào? | 2 | Chứa rõ câu đối chiếu 'sample the whole output' vs 'next token'. | 2 | Ngữ cảnh đoạn văn phân tích test-time compute đầy đủ. | 2 | Đầy đủ dữ kiện lập luận. |
| **q20** | Số tình nguyện viên LAION huy động? | 2 | Chứa số liệu '13,500 volunteers' và '10,000 conversations'. | 2 | Đoạn văn đầy đủ thông tin tổ chức phi lợi nhuận LAION. | 2 | Đầy đủ thông tin và lọc chuẩn xác trong post-training. |

* **Điểm Grounding trung bình:**
  * **Strategy A:** **1.80 / 2.0** (0 mức 0, 1 mức 1, 4 mức 2).
  * **Strategy B, C, D, E:** **2.00 / 2.0** (0 mức 0, 0 mức 1, 5 mức 2). Đều cung cấp đầy đủ bằng chứng và ngữ cảnh hoàn chỉnh để LLM tổng hợp câu trả lời chính xác.

---

## 4. Phân tích 2 trường hợp thất bại (Failure Cases)

### Trường hợp 1: Câu `q02` (Tập Dev - Trang 28)
* **Câu hỏi:** *"Why do language models use tokens as their unit instead of words or characters?"*
* **Gold Page:** Trang 28.
* **Gold Evidence:** *"1. Compared to characters, tokens allow the model to break words into meaningful components."*
* **Kết quả thực tế:**
  * Strategy A: Trượt Top-3 (Trả về các trang 31, 30, 27).
  * Strategy B & C: Trượt Top-3 (Trả về các trang 31, 30, 27).
* **Nguyên nhân cốt lõi (Root Cause):**
  * Các trang 30 và 31 thảo luận rất nhiều về *"language models"*, *"next-token prediction"*, và *"vocabulary distribution"*. Mật độ từ khóa chung này cao hơn hẳn trang 28 (trang 28 tập trung vào hình thái học từ: *"cook"* + *"ing"*).
  * Do đó, mô hình dense embedding thuần túy bị thiên lệch theo ngữ nghĩa chung của "token prediction" thay vì "nguyên nhân chọn token làm đơn vị". Đây là điểm yếu điển hình của Dense Retrieval, cần bổ sung Hybrid Search (BM25) để bắt chính xác cụm *"unit instead of word or character"*.

### Trường hợp 2: Câu `q08` (Tập Dev - Trang 83)
* **Câu hỏi:** *"Why are recurrent neural networks particularly vulnerable to vanishing and exploding gradients?"*
* **Gold Page:** Trang 83.
* **Gold Evidence:** *"RNNs are especially prone to vanishing and exploding gradients due to their recursive structure."*
* **Kết quả thực tế:**
  * Strategy A: Trượt Top-3 (Trả về các trang 84, 82, 85).
  * Strategy B & C: Trượt Top-3 (Trả về các trang 84, 82, 85).
* **Nguyên nhân cốt lõi (Root Cause):**
  * Trong Section *Model Architecture*, trang 84 và 85 phân tích sâu về Transformer và cơ chế Self-Attention nhằm giải quyết triệt để vấn đề gradient của RNN.
  * Các trang 84–85 lặp lại các thuật ngữ *"vanishing gradients"* và *"attention weights"* với dung lượng text lớn hơn nhiều so với ghi chú ngắn về RNN ở trang 83, dẫn đến độ tương đồng cosine của trang 84–85 cao hơn và đẩy trang 83 ra khỏi Top-3.

