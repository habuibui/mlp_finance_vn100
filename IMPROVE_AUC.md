# Đề xuất cải thiện AUC (từ ~0.5 lên mức cao hơn, hướng tới ~0.8)

Khi AUC xấp xỉ 0.5, mô hình gần như đoán ngẫu nhiên. Dưới đây là các hướng đi có thể thử, ưu tiên từ **kiểm tra dữ liệu** rồi mới chỉnh target/feature/model.

---

## 1. Kiểm tra dữ liệu (ưu tiên đầu tiên)

### 1.1 Dữ liệu có “sai” không?

- **Lỗi thường gặp:** nhầm ngày (date), nhầm mã (mack), merge sai → feature và target không khớp theo thời gian/cổ phiếu.
- **Cần kiểm:**
  - Cột `date` có đúng định dạng và thứ tự thời gian không?
  - Merge OHLC + fundamental + sentiment theo `(mack, date)` có bị trùng hoặc mất dòng không?
  - Giá `close` có bất thường (âm, nhảy vọt, lệch đơn vị) không?

### 1.2 Có tín hiệu trong dữ liệu không?

- Trong code đã thêm **Phase 7d**: in **tương quan |r| giữa từng feature và target**.
- Nếu **mọi |r| đều rất nhỏ (ví dụ < 0.05)** → tín hiệu tuyến tính rất yếu, dễ dẫn tới AUC ~0.5.
- Khi đó cần xem lại: cách tạo target, độ trễ dữ liệu (sentiment/price), hoặc thêm feature khác (xem mục 3).

### 1.3 Rò rỉ thông tin (data leakage)

- Target đang dùng là **lợi suất 5 ngày tới** (`pct_change(5).shift(-5)`) → đúng, không dùng tương lai.
- Cần đảm bảo: **mọi feature** đều chỉ dùng thông tin **tại hoặc trước** thời điểm dự đoán (không dùng giá/return của các ngày sau).
- Sentiment: nếu tin “ngày D” được gán cho “ngày D”, mà bạn dự đoán return từ D đến D+5 thì ổn; nếu tin D lại được gán nhầm cho D+1 thì dễ lệch.

---

## 2. Thay đổi cách định nghĩa target

Target quá “nhiễu” sẽ khó học, dù feature tốt.

- **Tăng horizon:** thử **10 ngày hoặc 20 ngày** thay vì 5 ngày (return ổn định hơn, ít nhiễu).
- **Ngưỡng rõ hơn:** thử quantile **20/80** thay vì 30/70 (bỏ nhiều vùng “giữa” → hai lớp tách bệt hơn).
- **Quantile theo từng mã hoặc theo sector:** thay vì quantile toàn bộ, tính quantile **theo mack** (hoặc theo sector) rồi gán 0/1 → target phù hợp hơn với từng nhóm cổ phiếu.
- **Tránh overlap:** đảm bảo các “cửa sổ 5 ngày” không chồng lấn quá nhiều (có thể lấy mẫu theo ngày hoặc theo chu kỳ).

---

## 3. Cải thiện feature

- **Thêm độ trễ (lag):** sentiment/technical có thể có tác dụng sau vài ngày; thử thêm **sent_score lag 1, 2, 3 ngày**, hoặc return lag 2, 3.
- **Feature tương tác:** ví dụ RSI × volatility, hoặc sentiment × sector (one-hot sector).
- **Theo ngành:** thêm **sector dummies** hoặc feature trung bình theo sector.
- **Giảm nhiễu:** bỏ bớt feature tương quan rất cao với nhau (đã làm với ngưỡng 0.8), có thể thử ngưỡng chặt hơn (ví dụ 0.7) nếu vẫn quá nhiều cột tương quan.
- **Chọn feature theo importance:** dùng permutation importance / SHAP đã có trong code, chỉ giữ lại top feature thay vì dùng “All”.

---

## 4. Mô hình và huấn luyện

- **Thử mô hình đơn giản trước:** chạy **Logistic Regression** (hoặc Ridge classifier) trên cùng feature và target.  
  - Nếu LR cũng ~0.5 → vấn đề chủ yếu ở **dữ liệu/target**, không phải do MLP phức tạp.  
  - Nếu LR > 0.5 rõ rệt → có tín hiệu, lúc đó mới tối ưu MLP (số layer, regularization, learning rate).
- **MLP:** thử giảm độ phức tạp (ít layer/unit hơn), tăng `alpha` (regularization), hoặc giảm `max_iter` + early stopping mạnh hơn để tránh overfit.
- **SMOTE:** với dữ liệu rất nhiễu, SMOTE đôi khi làm tệ hơn; thử tắt SMOTE một lần để so sánh AUC.

---

## 5. Đánh giá và thử nghiệm

- **Nhiều split thời gian:** thử 2–3 mốc split (train 60%/70%/80%) xem AUC test có ổn định không.
- **AUC theo sector:** xem sector nào AUC cao hơn; có thể tập trung cải thiện feature cho nhóm đó trước.
- **Kỳ vọng thực tế:** dự đoán giá cổ phiếu rất khó; AUC **0.55–0.65** đã có thể coi là có tín hiệu; **0.8** rất khó đạt và dễ overfit nếu không cẩn thận.

---

## Thứ tự gợi ý

1. Chạy pipeline, xem **Phase 7d** (tương quan feature–target). Nếu |r| đều rất nhỏ → ưu tiên kiểm tra dữ liệu và định nghĩa target (mục 1, 2).
2. Kiểm tra merge, ngày, và một vài mã cụ thể (vẽ vài chuỗi close + target) để chắc không sai dữ liệu.
3. Thử target 10 ngày, quantile 20/80, hoặc quantile theo mack (mục 2).
4. Thử Logistic Regression trên cùng data (mục 4); nếu LR > 0.5 thì tiếp tục cải thiện feature và MLP (mục 3, 4).
5. Chỉ kỳ vọng AUC rất cao (0.8) khi đã có bằng chứng tín hiệu rõ (LR hoặc correlation không quá thấp) và đã thử nhiều hướng trên.

Nếu bạn gửi thêm (ví dụ: vài dòng đầu của `ohlc`/`news`, hoặc kết quả in ra Phase 7d), có thể đề xuất cụ thể hơn cho dataset của bạn.
