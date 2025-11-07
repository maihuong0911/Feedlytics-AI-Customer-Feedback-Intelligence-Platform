# train_script.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import os

# Đường dẫn file và mô hình
TRAIN_FILE_PATH = 'data/clean_data.csv' 
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_model():
    """Tải dữ liệu đã làm sạch, huấn luyện mô hình phân tích cảm xúc và lưu lại."""
    try:
        print(f"1. Tải dữ liệu đã làm sạch từ {TRAIN_FILE_PATH}...")
        df_train = pd.read_csv(TRAIN_FILE_PATH)
        
        # KIỂM TRA CỘT BẮT BUỘC
        if 'tokenized_text' not in df_train.columns or 'label' not in df_train.columns:
            print("LỖI: File clean_data.csv phải có cột 'tokenized_text' và 'label'.")
            return

        # Đặt X là cột tokenized_text, y là cột label
        X = df_train['tokenized_text'].astype(str).fillna('')
        y = df_train['label'] 

        # Xây dựng Pipeline
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            ('clf', LinearSVC(C=1.0, max_iter=1000))
        ])

        print(f"2. Bắt đầu huấn luyện mô hình trên {len(df_train)} mẫu...")
        model_pipeline.fit(X, y)
        print("Huấn luyện hoàn tất.")

        # Lưu Mô hình
        joblib.dump(model_pipeline, MODEL_PATH)
        print(f"3. Mô hình đã được lưu tại {MODEL_PATH}")
        print("Hoàn tất! Giờ bạn có thể chạy app.py.")

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu tại {TRAIN_FILE_PATH}. Vui lòng kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")

if __name__ == '__main__':
    train_model()