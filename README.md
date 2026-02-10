## Mô tả pipeline tổng quan

Pipeline gồm hai bước chính:
- `sentiment_analysis.py`: phân tích cảm xúc tin tức tiếng Việt bằng PhoBERT và tạo thêm các biến sentiment ở cấp độ từng tin/ngày/cổ phiếu.
- `mlp_complete.py`: trộn dữ liệu giá, cơ bản và sentiment; xây dựng các đặc trưng, tạo biến mục tiêu và huấn luyện mô hình MLP để dự đoán xác suất giá cổ phiếu tăng mạnh trong 5 ngày tới.

---

## Dữ liệu đầu vào

- **`data/ohlc.csv`**  
  - Thông tin giá và khối lượng theo ngày cho từng mã (`mack`, `date`, `open`, `high`, `low`, `close`, `volume`, ...).  
  - Được dùng để:
    - Tính **các chỉ báo kỹ thuật**: SMA/EMA nhiều kỳ, RSI, MACD, Bollinger Bands, momentum (các tỷ suất sinh lợi trễ `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`), volatility 20 ngày, volume ratio,…

- **`data/fundamental.csv`**  
  - Dữ liệu cơ bản theo quý/năm cho từng mã: `eps`, `roe`, `roa`, `pb`, `pe`, `lnst_yoy`, `nophaitra_vcsh`, `vonhoa_tts`,…  
  - Được chuyển về **tần suất ngày** bằng cách resample theo ngày và **ffill** trong từng mã, sau đó ghép với dữ liệu giá.

- **`data/news.csv`**  
  - Tin tức thô theo ngày cho từng mã: `mack`, `date`, `title`,…  
  - Dùng làm đầu vào cho module sentiment (`sentiment_analysis.py`) để tính điểm cảm xúc cho từng tiêu đề.

- **`data/news_with_sentiment.csv`**  
  - Nếu chưa tồn tại, sẽ được tạo bởi `sentiment_analysis.py`.  
  - Bổ sung các cột sentiment ở cấp độ tin:  
    - `sent_pos`, `sent_neu`, `sent_neg`: xác suất tích cực/trung lập/tiêu cực.  
    - `sent_score`: điểm liên tục \(= \text{pos} - \text{neg}\).  
    - `sentiment_label`: nhãn \(-1, 0, 1\) (negative, neutral, positive).  
  - Trong `mlp_complete.py`, các cột này được **gom theo ngày–mã** để tạo các đặc trưng tổng hợp: mean/std/max của pos/neg, mean/std/min/max của score, `news_count` (số tin trong ngày).

---

## Các mô hình / phương pháp luận sử dụng

- **Phân tích cảm xúc (Sentiment Analysis – `sentiment_analysis.py`)**
  - Mô hình **PhoBERT**:
    - Tokenizer: `vinai/phobert-base`.
    - Mô hình phân loại sentiment: `wonrax/phobert-base-vietnamese-sentiment`.
  - Xử lý theo **mini-batch** trên GPU (nếu có), sử dụng softmax để lấy xác suất cho 3 lớp: positive, neutral, negative.
  - Tính các biến:
    - Xác suất từng lớp: `sent_pos`, `sent_neu`, `sent_neg`.
    - Điểm sentiment liên tục: `sent_score = sent_pos - sent_neg`.
    - Nhãn sentiment rời rạc: {-1, 0, 1} dựa trên argmax xác suất.
  - Đồng thời sinh **biểu đồ phân phối sentiment** và lưu vào thư mục `visualizations/`.

- **Xây dựng đặc trưng kỹ thuật, cơ bản, sentiment (`mlp_complete.py`)**
  - **Technical features**: SMA, EMA, RSI, MACD (+ signal, histogram), Bollinger Bands, volume ratio, các tỷ suất sinh lợi trễ, volatility 20 ngày,…
  - **Fundamental features**: bộ chỉ số EPS, ROE, ROA, P/E, P/B, tăng trưởng lợi nhuận, cấu trúc vốn,… được nội suy theo ngày và ghép với OHLC.
  - **Sentiment features**: thống kê theo ngày–mã của `sent_pos`, `sent_neu`, `sent_neg`, `sent_score`, `news_count`.
  - Xử lý missing:
    - Ffill theo từng mã cho fundamental & sentiment.
    - Điền giá trị trung tính cho một số biến sentiment (ví dụ score = 0, pos/neg ≈ 0.33 nếu thiếu).

- **Mô hình chính: Multi-Layer Perceptron (MLP)**
  - Kiến trúc mặc định: **4 tầng ẩn** `(256, 128, 64, 32)`, activation ReLU, optimizer Adam, learning rate adaptive, early stopping.  
  - Input là các tập feature khác nhau (ablation study): Fundamental, Technical, Sentiment, Fund+Tech, Fund+Sent, Tech+Sent, All.
  - Chuẩn hóa input bằng **RobustScaler** chỉ trên tập train.
  - Với dữ liệu mất cân bằng, áp dụng **SMOTE** (nếu `imbalanced-learn` có sẵn) trên tập train.

- **Phân tích chuyên sâu & kiểm định học thuật**
  - **TimeSeriesSplit**: đánh giá AUC theo từng fold thời gian (rolling).  
  - **SHAP (KernelExplainer)**: phân tích độ quan trọng đặc trưng cho mô hình tốt nhất.  
  - **Permutation Importance (AUC-based)**: đo mức giảm AUC khi xáo trộn từng đặc trưng.  
  - **Phân tích theo ngành**: huấn luyện lại mô hình tốt nhất riêng cho từng sector (dựa trên mapping `mack → sector`).  
  - **Bootstrap CI & Wilcoxon signed-rank test**:  
    - Bootstrap ước lượng khoảng tin cậy 95% cho AUC.  
    - Wilcoxon so sánh có ghép cặp AUC giữa mô hình tốt nhất và các cấu hình feature khác.

---

## Biến mục tiêu

- Biến mục tiêu được định nghĩa trong `mlp_complete.py` như sau:
  - Tính **tỷ suất sinh lợi tương lai 5 ngày** cho mỗi dòng dữ liệu:
    - `future_return_5d = pct_change(5).shift(-5)` trên cột `close` theo từng mã.
  - Lấy **phân vị 20% và 80%** của phân phối `future_return_5d` trên toàn bộ tập dữ liệu:
    - `q_low = quantile(0.2)` – ngưỡng “giảm mạnh”.
    - `q_high = quantile(0.8)` – ngưỡng “tăng mạnh”.
  - Ánh xạ thành **bài toán phân loại nhị phân**:
    - `target = 1` nếu `future_return_5d ≥ q_high` (có khả năng tăng mạnh trong 5 ngày tới).  
    - `target = 0` nếu `future_return_5d ≤ q_low` (có khả năng giảm mạnh).  
    - Các quan sát ở giữa bị loại bỏ (`NaN` → drop).
  - Sau xử lý, `target` là biến nhị phân (`int`) dùng để huấn luyện MLP.

---

## Cách đánh giá mô hình

- **Chia tập train/test theo thời gian**  
  - Cắt theo mốc `split_date = "2023-01-01"`: dữ liệu trước ngày này để train, sau/ngày này để test.  
  - Đảm bảo không rò rỉ thông tin tương lai (time leakage).

- **Các thước đo chính trên tập test**
  - **Accuracy**: tỷ lệ dự đoán đúng nhị phân.  
  - **AUC (ROC AUC)**: diện tích dưới đường cong ROC, đo quality của xác suất dự đoán.  
  - **Precision, Recall, F1**: đánh giá cân bằng giữa việc bắt được các trường hợp tăng mạnh và tránh báo động giả.  
  - **Average Precision (AP)**: diện tích dưới đường cong Precision–Recall.

- **Thí nghiệm ablation (so sánh tổ hợp đặc trưng)**
  - Chạy `train_mlp_time_split` cho từng nhóm feature: Fundamental, Technical, Sentiment, Fund+Tech, Fund+Sent, Tech+Sent, All.  
  - Lưu kết quả vào `model_results.csv` và vẽ biểu đồ `ablation_auc_bar.png`, `radar_metrics.png`, `feature_combination_comparison.png`.

- **Đánh giá ổn định theo thời gian và thống kê**
  - **Rolling TimeSeriesSplit AUC** (`rolling_auc.png`): AUC theo từng fold thời gian để xem tính ổn định.  
  - **Bootstrap CI** (`auc_confidence_intervals.png`): khoảng tin cậy 95% cho AUC từng mô hình.  
  - **Wilcoxon signed-rank test** (`statistical_comparison.png`, `statistical_tests.csv`): xác định chênh lệch giữa mô hình tốt nhất và các mô hình khác có **ý nghĩa thống kê** hay không (p-value < 0.05).

- **Phân tích diễn giải mô hình**
  - **Confusion Matrix** (`confusion_matrix.png`): phân tích lỗi dự đoán.  
  - **ROC Curve, Precision–Recall Curve** (`roc_curve_best_model.png`, `precision_recall_curve.png`).  
  - **SHAP & Permutation Importance** (`shap_summary.png`, `permutation_importance.png`): xác định đặc trưng nào đóng góp nhiều nhất vào quyết định của mô hình.

---

## Kết quả trả ra

- **File dữ liệu và báo cáo**
  - `data/news_with_sentiment.csv`: tin tức đã được gán sentiment cho từng tiêu đề.  
  - `model_results.csv`: bảng tổng hợp Accuracy, AUC, Precision, Recall, F1, AP cho từng tổ hợp feature.  
  - `sector_results.csv`: hiệu suất (AUC, Accuracy, F1) của mô hình tốt nhất trên từng ngành.  
  - `statistical_tests.csv`: kết quả kiểm định Wilcoxon giữa mô hình tốt nhất và các mô hình khác.  
  - `feature_combination_analysis.csv`: bảng xếp hạng và phân tích chi tiết ablation study.

- **Biểu đồ / hình ảnh trong thư mục `visualizations/`** (được sinh khi chạy code)
  - `sentiment_distribution.png`: phân phối nhãn và điểm sentiment từ PhoBERT.  
  - `ablation_auc_bar.png`, `feature_combination_comparison.png`, `radar_metrics.png`: so sánh hiệu suất các nhóm feature.  
  - `rolling_auc.png`: AUC theo từng fold thời gian.  
  - `precision_recall_curve.png`, `roc_curve_best_model.png`, `confusion_matrix.png`: đánh giá chi tiết mô hình tốt nhất.  
  - `permutation_importance.png`, `shap_summary.png`: độ quan trọng đặc trưng.  
  - `feature_correlation.png`: ma trận tương quan giữa các đặc trưng.  
  - `sector_performance.png`: hiệu suất theo ngành.  
  - `auc_confidence_intervals.png`, `statistical_comparison.png`: kết quả bootstrap CI và kiểm định thống kê.  
  - `learning_curve.png`: đường cong hội tụ loss của MLP.

---

## Kết quả kỳ vọng

- **Về mặt định lượng**
  - Mô hình tốt nhất (thường là nhóm feature kết hợp, ví dụ `All` hoặc `Tech+Sent`) được kỳ vọng đạt **AUC cao hơn đáng kể mức ngẫu nhiên 0.5**, nằm trong khoảng **0.55–0.65** tùy chất lượng dữ liệu.  
  - Accuracy và F1 ở mức trên 0.55–0.60, thể hiện khả năng nhận diện các phiên “tăng mạnh” tốt hơn đoán ngẫu nhiên.  
  - Bootstrap CI cho AUC **không cắt qua 0.5** đối với mô hình tốt nhất, cho thấy tín hiệu dự báo có ý nghĩa.  
  - Các kiểm định Wilcoxon cho thấy mô hình tốt nhất có **p-value < 0.05** khi so với một số cấu hình kém hơn, chứng minh cải thiện có ý nghĩa thống kê.

- **Về mặt định tính / diễn giải**
  - Các đặc trưng **momentum, volatility, một số chỉ tiêu cơ bản và sentiment** được kỳ vọng nằm trong nhóm feature quan trọng nhất theo SHAP và Permutation Importance.  
  - Một số ngành có cấu trúc dòng tiền và tin tức rõ ràng (ví dụ tài chính, ngân hàng, hàng tiêu dùng) được kỳ vọng có **AUC cao hơn mặt bằng chung** trong `sector_results.csv`.  
  - Kết quả cho thấy việc **kết hợp thông tin kỹ thuật + cơ bản + cảm xúc thị trường** giúp mô hình ổn định hơn so với từng nhóm đặc trưng đơn lẻ, dù mức cải thiện có thể vừa phải (synergy nhẹ thay vì nhảy vọt).

---

## Cách chạy nhanh

- Cài đặt thư viện (môi trường Python 3.x):
```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap imbalanced-learn
pip install torch transformers tqdm
```

- Chạy lần lượt hai bước chính:
```bash
# 1. Tạo cache sentiment (chỉ cần chạy lại khi thay đổi news.csv)
python sentiment_analysis.py

# 2. Huấn luyện và đánh giá mô hình MLP
python mlp_complete.py
```

*(Bạn cũng có thể sử dụng `run_all.bat` nếu đã cấu hình sẵn trên Windows.)*
