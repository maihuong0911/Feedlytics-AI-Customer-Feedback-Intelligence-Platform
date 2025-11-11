# Feedlytics: AI Customer Feedback Intelligence Platform 🇻🇳

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Model: PhoBERT & Gemini AI](https://img.shields.io/badge/Models-PhoBERT%2FGemini%20AI-red.svg)]()

Dự án Feedlytics là một hệ thống phân tích phản hồi khách hàng tự động (Customer Feedback Analysis System) sử dụng các mô hình Ngôn ngữ Lớn (LLM) và Học Sâu (Deep Learning) để xử lý văn bản tiếng Việt từ các nền tảng Thương mại Điện tử. Mục tiêu là chuyển đổi dữ liệu phản hồi thô thành các **Actionable Insights** (Thông tin hành động được) về Sentiment (Cảm xúc) và Topic (Chủ đề).

## ✨ Tính năng nổi bật

* **Sentiment Classification (PhoBERT):** Phân loại cảm xúc (Positive/Negative/Neutral) với độ chính xác **93.4\%** bằng cách Fine-tuning mô hình PhoBERT trên tập dữ liệu E-commerce 20.000 mẫu.
* **Topic Modeling (Gemini AI):** Tự động phân loại 6 chủ đề chính (ví dụ: Chất lượng sản phẩm, Giao hàng/Logistics, Dịch vụ khách hàng) và sinh ra **3 đề xuất cải tiến cụ thể**.
* **Web Application:** Giao diện người dùng thân thiện (Flask + Bootstrap 5), hỗ trợ xử lý hàng loạt file **CSV/Excel/DOCX**.
* **Performance:** Xử lý 1000 mẫu phản hồi trong khoảng **12.5 giây** (trên GPU RTX 3060).

## 🚀 Cấu trúc dự án

| File/Thư mục | Mô tả |
| :--- | :--- |
| `app.py` | Core Flask application: định tuyến (routing), logic nghiệp vụ, quản lý phiên và xử lý request. |
| `train_phobert.py` | Script dùng để **Fine-tune** mô hình PhoBERT trên tập dữ liệu tiếng Việt. |
| `train_script.py` | Script tiền xử lý dữ liệu và chuẩn bị môi trường cho việc đào tạo/chạy mô hình. |
| `test_phobert.py` | Script đánh giá hiệu suất (Accuracy, F1-score) của mô hình PhoBERT đã huấn luyện. |
| `train.csv` | Tập dữ liệu mẫu (hoặc tập huấn luyện 20.000 samples) được sử dụng trong nghiên cứu. |
| `feedback_reports.db` | Cơ sở dữ liệu SQLite (Lưu trữ tài khoản người dùng, báo cáo đã lưu). |
| `README.md` | File mô tả dự án. |

## ⚙️ Hướng dẫn cài đặt và triển khai

### 1. Yêu cầu môi trường

* Python 3.10+
* GPU NVIDIA (khuyến nghị cho PhoBERT inference/training)
* CUDA Toolkit 11.8+

### 2. Cài đặt Dependencies

Tạo và kích hoạt môi trường ảo, sau đó cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt 
# (requirements.txt chứa các gói như: torch, transformers, flask, pandas, google-genai, openpyxl, python-docx, ... )
