# Tiêu chí chọn cấu hình chiến lược (Selection Criteria)
*Ghi nhận và chốt trước khi chạy thử nghiệm trên tập Test theo quy định của GUIDE.md và RUBRIC.md.*

---

## 1. Thứ tự ưu tiên quyết định lựa chọn cấu hình tối ưu (Frozen Criteria)
Khi so sánh kết quả giữa các chiến lược A, B, và C trên tập **Dev (15 câu hỏi)**, chiến lược được chọn làm cấu hình tối ưu (Best Strategy) phải thỏa mãn theo thứ tự ưu tiên sau:

1. **Ưu tiên 1 (Chỉ số cốt lõi): Evidence Hit@3 trên tập Dev cao nhất**
   * Trong hệ thống RAG cho AI Agent, mục tiêu quan trọng nhất của Data Ingestion là đảm bảo thông tin nền (grounding evidence) lọt vào Top-3 để LLM có đầy đủ dữ kiện trả lời đúng, tránh hiện tượng ảo giác (hallucination).
2. **Ưu tiên 2 (Độ chính xác tức thì): Hit@1 trên tập Dev cao hơn**
   * Nếu hai cấu hình có Hit@3 tương đương nhau, cấu hình nào đưa đúng chunk chứa evidence lên ngay vị trí Top-1 sẽ được ưu tiên vì giúp giảm độ trễ và chi phí prompt cho agent.
3. **Ưu tiên 3 (Chất lượng ngữ cảnh): Điểm Chunk Coherence trung bình $\ge 1.5 / 2.0$**
   * Đo lường trên 10 vị trí nguồn cố định. Chunk không được bị cắt cụt giữa câu hoặc mất chủ ngữ/vị ngữ quan trọng.
4. **Ưu tiên 4 (Tài nguyên và độ trễ): Số lượng chunk và chi phí nhúng/lưu trữ**
   * Ưu tiên chiến lược có số lượng chunk vừa phải, tiết kiệm dung lượng lưu trữ trên Qdrant Cloud và thời gian tìm kiếm.

---

## 2. Giả thuyết ban đầu (Hypotheses) cho 3 chiến lược

* **Chiến lược A (Fixed-size: 500 chars, overlap: 50, No filter):**
  * *Giả thuyết:* Cắt cơ học theo độ dài cố định sẽ tạo ra nhiều chunk nhất. Khả năng cao sẽ làm đứt gãy các định nghĩa kỹ thuật dài hoặc bảng biểu ở giữa câu $\rightarrow$ Điểm Coherence sẽ thấp nhất (khoảng 0.8 - 1.2 / 2.0). Hit@1 có thể bị nhiễu do chunk thiếu ngữ cảnh trọn vẹn.
* **Chiến lược B (Structure-aware / By-paragraph: max 800 chars, No filter):**
  * *Giả thuyết:* Gom các đoạn văn hoàn chỉnh theo cấu trúc tự nhiên của sách. Điểm Coherence sẽ cao vượt trội ($\ge 1.7 / 2.0$). Số lượng chunk ít hơn đáng kể so với A, giúp tìm kiếm nhanh hơn. Hit@3 và Grounding dự kiến cao hơn A.
* **Chiến lược C (Structure-aware + Metadata Filtering):**
  * *Giả thuyết:* Dùng cùng phương pháp chunking với B để cô lập hoàn toàn biến số chunking. Đối với các câu hỏi có điều kiện lọc (ví dụ: `chapter` hoặc `section`), áp dụng payload filter tại Qdrant Cloud. Chiến lược C sẽ loại bỏ 100% các ứng viên gây nhiễu ngoài phạm vi, giúp độ chính xác của các câu có filter đạt tối đa.

