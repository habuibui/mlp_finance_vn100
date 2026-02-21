# Đề xuất cải thiện pipeline MLP & Sentiment

## 1. Vấn đề: Chỉ dùng ~40% dữ liệu

**Nguyên nhân:** Target được tạo bằng quantile 0.2 / 0.8 → chỉ giữ **bottom 20%** (target=0) và **top 20%** (target=1) của `future_return_5d`. 60% dữ liệu ở giữa bị loại.

**Đã áp dụng trong code:**
- Thêm cấu hình `TARGET_MODE` và quantile trong `mlp_complete.py`:
  - **quantile:** `TARGET_QUANTILE_LOW` / `TARGET_QUANTILE_HIGH` (mặc định 0.3/0.7 → dùng **60%** dữ liệu).
  - **binary:** target = 1 nếu return > ngưỡng, 0 nếu không → dùng **~100%** dữ liệu (chỉ bỏ dòng thiếu return).
  - **fixed_pct:** ngưỡng % cố định (ví dụ >1% = 1, <-1% = 0), phần giữa = nan (có thể bỏ).
- In ra số dòng và % dữ liệu dùng sau khi tạo target.

**Gợi ý thêm:** Nếu cần dùng tối đa dữ liệu mà vẫn 2 lớp, đặt `TARGET_MODE = "binary"` và `TARGET_FIXED_THRESHOLD_PCT = 0` (hoặc 0.005 cho 0.5%).

---

## 2. Vấn đề: Kết quả chỉ đạt ~50% so với return target (AUC ~ 0.5)

**Nguyên nhân có thể:** Tín hiệu yếu, nhiễu, target quá khó (extreme quantile), hoặc model/feature chưa đủ.

**Đề xuất đã áp dụng:**
- **Dùng thêm dữ liệu:** quantile 0.3/0.7 hoặc binary → nhiều mẫu hơn, học ổn định hơn.
- **class_weight="balanced"** cho MLP → giảm lệch lớp khi dùng binary hoặc quantile rộng.

**Đề xuất có thể thử thêm:**

| Hạng mục | Mô tả |
|----------|--------|
| **Target** | Thử `TARGET_MODE = "binary"` với threshold nhỏ (0, 0.005). Hoặc regression (dự báo return liên tục) thay vì classification. |
| **Horizon** | Thử `future_return_1d` (1 ngày) thay vì 5 ngày: tín hiệu gần hơn, dễ học hơn (nhưng nhiều noise). |
| **Feature** | Thêm lag sentiment (sent_score lag 1–5 ngày), rolling mean sentiment theo tuần, hoặc tương tác sector × sentiment. |
| **Train/test** | Giữ time-based split; có thể thử **expanding window** (train luôn mở rộng) thay vì cố định 2023-01-01. |
| **Model** | Grid search `hidden_layer_sizes`, `alpha`, `learning_rate_init`; thử thêm dropout (nếu chuyển sang PyTorch/Keras). |
| **Sentiment** | Trong `sentiment_analysis.py` đã xử lý toàn bộ tin; có thể thêm **aggregation theo cửa sổ** (3 ngày, 5 ngày) trong `mlp_complete.py` khi merge. |
| **Đánh giá** | So sánh AUC theo sector (đã có); thêm **backtest đơn giản**: nếu model dự báo 1 thì kỳ vọng return trung bình cao hơn 0. |

---

## 3. File đã chỉnh

- **mlp_complete.py:** Cấu hình target (quantile/binary/fixed_pct), in % dữ liệu dùng, `class_weight="balanced"` cho MLP.
- **sentiment_analysis.py:** Không thay đổi; logic sentiment đã dùng toàn bộ tin (không phải nguồn 40%).

Chạy lại pipeline với `TARGET_MODE = "quantile"`, `TARGET_QUANTILE_LOW = 0.3`, `TARGET_QUANTILE_HIGH = 0.7` để dùng 60% dữ liệu và so sánh AUC với thiết lập cũ (0.2/0.8).
