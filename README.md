# Path Planning Optimization: Multi-Objective Reinforcement Learning (MO-RL)

Dự án này ứng dụng các thuật toán **Học Tăng Cường Sâu Đa Mục Tiêu (MO-RL)** và **Thuật toán Tiến hóa (MOEA/D)** để giải quyết bài toán quy hoạch quỹ đạo (Path Planning) cho robot trong môi trường có cảm biến (sensors) và vật cản (obstacles).

Hệ thống được thiết kế để tìm ra tập các đường đi tối ưu (Pareto Front) cân bằng đồng thời 3 mục tiêu:
1. **Exposure (Độ phơi nhiễm):** Tối thiểu hóa rủi ro bị phát hiện bởi mạng lưới cảm biến.
2. **Length (Quãng đường):** Tối thiểu hóa độ dài di chuyển.
3. **Feasibility (Tính khả thi):** Đảm bảo không va chạm vật cản và di chuyển trơn tru.

---

## Các tính năng nổi bật

* **State-of-the-Art MO-RL:** Cài đặt các thuật toán tiên tiến nhất bao gồm **MO-PPO**, **MO-TD3**, và **MO-SAC**.
* **Kiến trúc mạng Neural tối ưu:** 
  * Sử dụng **Triple-Head Critic** để dự đoán phần thưởng (Value/Q-value) độc lập cho từng mục tiêu.
  * **Conditioned State (16-dim):** Tích hợp trực tiếp vector trọng số mục tiêu `[w_exp, w_len, w_feas]` vào state, giúp Agent tự động điều chỉnh chiến thuật linh hoạt.
* **Đánh giá tự động chuẩn khoa học:**
  * Benchmark tự động chạy qua nhiều hạt giống ngẫu nhiên (Seeds: 42, 100, 2024).
  * Tự động tính toán các siêu dộ đo (Metrics) đa mục tiêu: **Hypervolume (HV)** và **IGD+**.
  * Vẽ đồ thị với dải mờ (EMA Smoothed Mean/Std) sẵn sàng để đưa vào báo cáo/bài báo.

---

## Cấu trúc dự án

```text
├── algorithm/
│   ├── moead.py           # Thuật toán Multi-Objective Evolutionary (MOEA/D)
│   ├── mo_ppo.py          # Multi-Objective Proximal Policy Optimization
│   ├── mo_td3.py          # Multi-Objective Twin Delayed DDPG
│   └── mo_sac.py          # Multi-Objective Soft Actor-Critic
├── utils/
│   ├── config_loader.py   # Load file cấu hình YAML
│   ├── draw.py            # Hỗ trợ vẽ bản đồ, đồ thị
│   └── generator.py       # Khởi tạo môi trường giả lập
├── data/                  # Thư mục lưu môi trường (phân loại theo số lượng sensor)
├── result_benchmark/      # Thư mục lưu toàn bộ kết quả sau khi chạy
├── config.yaml            # File cấu hình trung tâm (Môi trường, Thuật toán)
├── generate_env.py        # Script tạo bản đồ/môi trường mới
└── run_benchmark.py       # Script chính để chạy huấn luyện và đánh giá
```

---

## Cài đặt

Dự án yêu cầu **Python 3.8+**. Cài đặt các thư viện phụ thuộc bằng lệnh sau:

```bash
pip install numpy matplotlib pyyaml shapely torch pymoo
```
> **Lưu ý:** Khuyến nghị cài đặt `torch` phiên bản hỗ trợ **CUDA** nếu máy bạn có GPU để quá trình huấn luyện RL diễn ra nhanh hơn.

---

## Hướng dẫn sử dụng

### Bước 1: Khởi tạo môi trường (Map)
Trước khi chạy thuật toán, bạn cần tạo một bản đồ giả lập. Mọi thông số (số cảm biến, số vật cản, kích thước bản đồ) được chỉnh sửa trong file `config.yaml`.

```bash
python generate_env.py
```
*Đầu ra:* Một file JSON sẽ được lưu tự động, ví dụ: `data/50 sensors/env_20260124_120000.json`.

### Bước 2: Chạy Benchmark & Huấn luyện
Sử dụng script `run_benchmark.py` để bắt đầu quá trình huấn luyện. Bạn cần truyền đường dẫn của file môi trường vừa tạo ở Bước 1.

**Chạy so sánh TẤT CẢ thuật toán (Benchmark toàn diện):**
```bash
python run_benchmark.py "data/50 sensors/env_20260124_120000.json"
```

**Chạy thử nghiệm một thuật toán cụ thể (Dùng cờ `--algo`):**
Nếu bạn chỉ muốn kiểm tra độc lập một thuật toán để tiết kiệm thời gian:
```bash
python run_benchmark.py "data/50 sensors/env_20260124_120000.json" --algo MO-SAC
```
*(Các tên hợp lệ cho `--algo`: `MOEAD`, `MO-PPO`, `MO-TD3`, `MO-SAC`).*

---

## Hướng dẫn đọc kết quả

Sau khi chạy xong, kết quả sẽ được lưu tại: `result_benchmark/Bench_{n}sensors_{timestamp}/`. Thư mục này chia làm 3 phần:

### 1. Thư mục `comparative_plots/` (Đồ thị so sánh)
* **`01_Hypervolume_Compare.png`**: Thể hiện chỉ số Hypervolume. Đường nào càng **CA0** thì thuật toán đó tìm được tập Pareto càng phân bố rộng và chất lượng.
* **`02_IGD_Plus_Compare.png`**: Thể hiện chỉ số IGD+. Đường nào càng **THẤP** thì các nghiệm tìm được càng gần với tập nghiệm tối ưu thực sự (True Pareto Front).
* **`04_SuccessRate_Compare.png`**: Tỉ lệ phần trăm Agent tìm được đường đi không đâm vào vật cản trong suốt quá trình học.
* **`05A` & `05B_ParetoFront.png`**: Trực quan hóa hình dáng tập Pareto (đường Trade-off giữa Exposure và Length).

### 2. Thư mục `individual_algorithms/` (Kết quả chi tiết)
Bên trong thư mục của từng thuật toán (VD: `MO-PPO/`):
* **`{Algo}_Map_Seed_{X}.png`**: Bản đồ hiển thị trực quan quỹ đạo tốt nhất mà Agent tìm được.
* **`{Algo}_Loss_Reward.png`**: (Chỉ dành cho MO-RL) Biểu đồ hội tụ của *Actor Loss, Critic Loss* và sự gia tăng của 3 mảng *Reward* (Exposure, Length, Feas). Rất hữu ích để debug quá trình học.

### 3. Thư mục `data/` (Dữ liệu thô JSON)
* **`00_benchmark_summary.json`**: Bảng tổng kết dạng số (Mean & Std) của Thời gian chạy, HV, IGD+ và Tỉ lệ thành công.
* **Các file `data_*.json`**: Chứa mảng dữ liệu thô (đã tính trung bình và độ lệch chuẩn). Sử dụng dữ liệu này nếu bạn muốn tự vẽ đồ thị bằng Excel, OriginLab, hoặc MATLAB.

---
*Mọi siêu tham số của RL (Learning Rate, Batch Size, Gamma, v.v.) và thông số môi trường có thể dễ dàng được điều chỉnh tại file `config.yaml`.*