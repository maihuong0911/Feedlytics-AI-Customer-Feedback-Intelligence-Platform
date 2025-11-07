import sqlite3
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, g 
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
import pandas as pd
import re # <<< THÊM: Thư viện regex để tách Text và Rating
import os
import io
import docx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- START: PHẦN THÊM MỚI CHO GEMINI AI ---
from google import genai
# --- END: PHẦN THÊM MỚI CHO GEMINI AI ---

# ----------------------------------------------------------------
# --- CẤU HÌNH CƠ BẢN & BẢO MẬT ---
# ----------------------------------------------------------------
DATABASE = 'feedback_reports.db'
app = Flask(__name__) 

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_dev_secret_key_change_me')

# --- CẤU HÌNH FLASK-MAIL ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com' 
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your_email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your_app_password')
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME'] 

mail = Mail(app)

# --- CẤU HÌNH FLASK-LOGIN ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 
login_manager.login_message = "Vui lòng đăng nhập để truy cập trang này."

s = URLSafeTimedSerializer(app.config['SECRET_KEY'], salt='password-reset-salt')

# --- CẤU HÌNH MÔ HÌNH VÀ DATA ---
MODEL_PATH = 'model/phobert_finetuned' 
REPORT_DB_TABLE = 'reports'
DEFAULT_TEXT_COLUMN = 'Văn bản phản hồi'
MAX_ROWS_TO_ANALYZE = 1000

# ✅ CACHE MODEL AI
_MODEL_CACHE = None
# ✅ XÁC ĐỊNH DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Thiết bị AI đang dùng: {DEVICE}")


# --- START: KHỞI TẠO GEMINI CLIENT ---
GEMINI_CLIENT = None
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
    if GEMINI_API_KEY:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Đã khởi tạo Gemini Client thành công.")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy GEMINI_API_KEY. Chức năng AI sẽ không hoạt động.")
except Exception as e:
    print(f"❌ Lỗi khởi tạo Gemini Client: {e}")
# --- END: KHỞI TẠO GEMINI CLIENT ---


# ----------------------------------------------------------------
# --- USER MODEL VÀ KHỞI TẠO DB ---
# ----------------------------------------------------------------
class User(UserMixin):
    def __init__(self, id, username, email, password):
        self.id = id
        self.username = username
        self.email = email
        self.password = password
    
@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close() 
    if user_data:
        return User(*user_data)
    return None

def get_db_connection():
    """Mở và trả về kết nối DB."""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE) 
    return g.db

@app.teardown_appcontext
def close_db_connection(exception):
    """Đảm bảo kết nối DB được đóng khi app context kết thúc."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {REPORT_DB_TABLE} (
            id INTEGER PRIMARY KEY,
            report_name TEXT NOT NULL,
            analysis_data TEXT, 
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()


# ----------------------------------------------------------------
# --- HÀM TIỆN ÍCH CHO PHÂN TÍCH (BAO GỒM DOCX) ---
# ----------------------------------------------------------------
def sentiment_to_rating(sentiment):
    """Chuyển đổi sentiment sang rating 1-5."""
    if sentiment == 'positive': return 5.0
    if sentiment == 'negative': return 1.0
    return 3.0

# <<< THÊM: HÀM TRÍCH XUẤT TEXT TỪ DOCX
def extract_text_from_docx(file):
    """Trích xuất tất cả văn bản, tách theo dòng từ file docx."""
    try:
        file.seek(0)
        document = docx.Document(io.BytesIO(file.read()))
        # Trích xuất từng đoạn (paragraph), coi mỗi đoạn là một phản hồi
        lines = []
        for para in document.paragraphs:
            if para.text.strip():
                lines.append(para.text.strip())
        return lines
    except Exception as e:
        app.logger.error(f"Lỗi trích xuất DOCX: {e}")
        return None
# >>> KẾT THÚC HÀM TRÍCH XUẤT TEXT TỪ DOCX


# ✅ THAY THẾ HÀM LOAD MODEL
def load_model():
    """Tải PhoBERT fine-tuned với cache."""
    global _MODEL_CACHE
    
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy PhoBERT fine-tuned tại {MODEL_PATH}")

    # Tải Tokenizer và Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Chuyển model sang device (CPU/GPU) và chế độ inference
    model.to(DEVICE)
    model.eval() 

    _MODEL_CACHE = (tokenizer, model)
    app.logger.info(f"✅ Đã load PhoBERT fine-tuned vào cache. Device: {DEVICE}")
    return _MODEL_CACHE

def send_password_reset_email(user):
    """Gửi email đặt lại mật khẩu."""
    token = s.dumps(user.email, salt='password-reset-salt')
    reset_url = url_for('reset_password', token=token, _external=True)
    
    msg = Message('Đặt lại Mật khẩu - AI Feedback Analyzer Pro', recipients=[user.email])
    msg.body = f"""
    Xin chào {user.username},

    Vui lòng nhấp vào liên kết sau để đặt lại mật khẩu của bạn:
    {reset_url}
    
    Liên kết này sẽ hết hạn sau 1 giờ.
    """
    try:
        mail.send(msg)
        return True
    except Exception as e:
        app.logger.error(f"LỖI GỬI EMAIL: {e}")
        return False


# ----------------------------------------------------------------
# --- 1. ENDPOINTS XÁC THỰC (Không thay đổi) ---
# ----------------------------------------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, password FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()

        if user_data and check_password_hash(user_data[3], password):
            user = User(*user_data)
            login_user(user)
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Tên người dùng hoặc mật khẩu không đúng.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                           (username, email, hashed_password))
            conn.commit()
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Tên người dùng hoặc Email đã tồn tại.', 'danger')
        except Exception as e:
            flash(f'Lỗi đăng ký: {e}', 'danger')

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Bạn đã đăng xuất.', 'info')
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
        user_data = cursor.fetchone()
        
        if user_data:
            user = User(*user_data)
            if send_password_reset_email(user):
                flash(f'Một liên kết đặt lại mật khẩu đã được gửi đến {email}.', 'info')
            else:
                 flash('Lỗi khi gửi email. Vui lòng kiểm tra cấu hình SMTP.', 'danger')
        else:
            flash('Không tìm thấy tài khoản nào với địa chỉ email này.', 'danger')
            
        return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)
    except Exception:
        flash('Liên kết đặt lại mật khẩu không hợp lệ hoặc đã hết hạn.', 'danger')
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
    user_data = cursor.fetchone()
    
    if not user_data:
        flash('Người dùng không tồn tại.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if new_password != confirm_password:
            flash('Mật khẩu nhập lại không khớp.', 'danger')
            return render_template('reset_password.html', token=token, email=email)
        
        hashed_password = generate_password_hash(new_password)
        
        cursor.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, email))
        conn.commit()
        
        flash('Mật khẩu của bạn đã được cập nhật thành công! Vui lòng đăng nhập.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token, email=email)


# ----------------------------------------------------------------
# --- 2. ENDPOINTS CHÍNH CỦA ỨNG DỤNG (Không thay đổi) ---
# ----------------------------------------------------------------

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=current_user.username)


# ----------------------------------------------------------------
# --- API: Phân tích Dữ liệu (ĐÃ CẬP NHẬT CHO PHOBERT VÀ DOCX) ---
# ----------------------------------------------------------------

@app.route('/api/analyze_file', methods=['POST'])
@login_required
def analyze_file():
    """✅ Phân tích file CSV, Excel, DOCX và lưu báo cáo (Sử dụng PhoBERT và tạo Rating 1-5)."""
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'Không tìm thấy file tải lên.'}), 400

    file = request.files['file']
    text_column = request.form.get('text_column', DEFAULT_TEXT_COLUMN) 
    
    df = None
    filename = file.filename
    
    try:
        
        if filename.lower().endswith('.csv'):
            # Logic đọc CSV (có thử nhiều encoding)
            file.seek(0)
            file_content = file.read()
            encodings_to_try = ['utf-8', 'latin-1', 'cp1258']
            for encoding in encodings_to_try:
                try:
                    file_stream = io.StringIO(file_content.decode(encoding))
                    try:
                        df = pd.read_csv(file_stream)
                    except pd.errors.ParserError:
                         file_stream.seek(0)
                         df = pd.read_csv(file_stream, sep=';')
                    if df is not None:
                        break 
                except Exception:
                    continue 

        elif filename.lower().endswith(('.xlsx', '.xls')):
            # Logic đọc file Excel
            file.seek(0)
            df = pd.read_excel(file)
            
        elif filename.lower().endswith('.docx'):
            # <<< LOGIC MỚI: Đọc file DOCX, phân tách từng dòng/đoạn
            doc_lines = extract_text_from_docx(file)
            
            if doc_lines:
                feedback_texts = []
                ratings_list_from_docx = []
                
                # Biểu thức chính quy: tìm (,) theo sau bởi 1-2 chữ số (rating) ở cuối dòng
                rating_pattern = re.compile(r',\s*(\d{1,2}(?:\.\d{1})?)\s*$') 

                for line in doc_lines:
                    match = rating_pattern.search(line)
                    if match:
                        rating_str = match.group(1)
                        # Tách văn bản phản hồi khỏi rating
                        text = line[:match.start()].strip() 
                        feedback_texts.append(text)
                        try:
                            ratings_list_from_docx.append(float(rating_str))
                        except ValueError:
                            ratings_list_from_docx.append(None) # Không thể parse rating
                    else:
                        # Nếu không tìm thấy rating, coi toàn bộ dòng là text
                        feedback_texts.append(line)
                        ratings_list_from_docx.append(None)
                        
                df = pd.DataFrame({
                    DEFAULT_TEXT_COLUMN: feedback_texts,
                    'Rating_Docx': ratings_list_from_docx # Tạo cột tạm để sau này gán vào cột 'rating' chính
                })
                # Đảm bảo text_column là tên cột chính xác nếu file là DOCX
                text_column = DEFAULT_TEXT_COLUMN 
            else:
                 return jsonify({'status': 'error', 'error': 'Lỗi: Không thể trích xuất văn bản từ file DOCX hoặc file trống.'}), 500

        else:
            return jsonify({'status': 'error', 'error': 'Định dạng file không được hỗ trợ. Vui lòng dùng CSV, Excel (.xlsx, .xls) hoặc DOCX.'}), 400

        
        if df is None or df.empty:
             return jsonify({'status': 'error', 'error': 'Lỗi Encoding hoặc Định dạng: Không thể đọc dữ liệu từ file.'}), 500

        # ✅ Tải mô hình PhoBERT (đã có cache)
        tokenizer, model = load_model()

        # Xác định cột phản hồi
        if text_column not in df.columns:
             if len(df.columns) > 0 and not filename.lower().endswith('.docx'):
                text_column = df.columns[0]
                app.logger.warning(f"Chuyển sang cột đầu tiên: '{text_column}'.")
             elif filename.lower().endswith('.docx'):
                 pass
             else:
                return jsonify({'status': 'error', 'error': f'Không tìm thấy cột phản hồi "{text_column}".'}), 400
        
        if df[text_column].empty:
             return jsonify({'status': 'error', 'error': f'Cột "{text_column}" không có dữ liệu.'}), 400
        
    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    except Exception as e:
        app.logger.error(f"Lỗi xử lý file: {e}")
        return jsonify({'status': 'error', 'error': f'Lỗi xử lý file: {e}'}), 500
    
    # --- BẮT ĐẦU PHÂN TÍCH ---
    
    df = df.reset_index(drop=True)
    
    # ✅ GIỚI HẠN SỐ DÒNG PHÂN TÍCH
    if len(df) > MAX_ROWS_TO_ANALYZE:
        app.logger.warning(f"File có {len(df)} dòng. Chỉ phân tích {MAX_ROWS_TO_ANALYZE} dòng đầu.")
        df = df.head(MAX_ROWS_TO_ANALYZE)
    
    df['id'] = range(1, len(df) + 1)
    X_predict = df[text_column].astype(str).fillna('') 
    
    # 1. Phân tích Sentiment & Rating 1-5 
    sentiments = []
    ratings_list = []
    
    texts = X_predict.tolist()
    
    for i in range(0, len(texts), 32): 
        batch_texts = texts[i:i + 32]
        
        inputs = tokenizer(batch_texts, 
                           return_tensors="pt", 
                           truncation=True, 
                           padding=True, 
                           max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().tolist() # Lấy xác suất
            predicted_classes = torch.argmax(logits, dim=1).cpu().tolist()
        
        for predicted_class, prob_list in zip(predicted_classes, probs):
            max_prob = max(prob_list)
            
            if predicted_class == 0: # Negative
                sentiments.append('negative')
                ratings_list.append(1.0 if max_prob >= 0.8 else 2.0)
            elif predicted_class == 2: # Positive
                sentiments.append('positive')
                ratings_list.append(5.0 if max_prob >= 0.8 else 4.0)
            else: # Neutral
                sentiments.append('neutral')
                ratings_list.append(3.0)

    # Gán kết quả dự đoán vào DataFrame
    df['sentiment'] = sentiments
    df['rating'] = ratings_list

    # 2. Xử lý RATING (Ưu tiên rating gốc từ file CSV/Excel/DOCX nếu có)
    rating_column_name = None
    potential_rating_cols = ['Rating', 'rating', 'Đánh giá'] 
    
    for col in potential_rating_cols:
        if col in df.columns:
            rating_column_name = col
            break
            
    if filename.lower().endswith('.docx') and 'Rating_Docx' in df.columns:
        # Xử lý đặc biệt cho DOCX: Rating gốc nằm ở cột tạm 'Rating_Docx'
        df['rating'] = df['Rating_Docx'].fillna(df['rating'])
        app.logger.info("Đã sử dụng Rating gốc từ DOCX (tách từ cuối dòng).")
    elif rating_column_name:
        try:
            # Ưu tiên rating gốc từ CSV/Excel nếu có
            original_ratings = pd.to_numeric(df[rating_column_name], errors='coerce').astype(float).round(2)
            df['rating'] = original_ratings.fillna(df['rating']) 
            app.logger.info("Đã sử dụng và bổ sung Rating gốc từ CSV/Excel.")
        except Exception as e:
            app.logger.warning(f"Lỗi chuyển đổi Rating gốc: {e}. Đã dùng rating dự đoán 1-5.")
    
    ratings_list = df['rating'].tolist() 

    # --- TÍNH TOÁN METRICS (Giữ nguyên) ---
    total_count = len(df)
    positive_count = sum(1 for s in sentiments if s == 'positive')
    negative_count = sum(1 for s in sentiments if s == 'negative')
    neutral_count = total_count - positive_count - negative_count
    
    metrics = {
        'total_count': total_count,
        'positive_rate': round((positive_count / total_count) * 100, 2) if total_count > 0 else 0,
        'negative_rate': round((negative_count / total_count) * 100, 2) if total_count > 0 else 0,
        'neutral_rate': round((neutral_count / total_count) * 100, 2) if total_count > 0 else 0,
        'avg_rating': round(df['rating'].mean(), 2) if total_count > 0 else 0, 
    }
    
    # Đổi tên cột (Giữ nguyên)
    df = df.rename(columns={text_column: 'feedback_text'})
    df['topic'] = 'Chưa phân loại'
    
    # Loại bỏ cột tạm 'Rating_Docx' nếu tồn tại
    if 'Rating_Docx' in df.columns:
        df = df.drop(columns=['Rating_Docx'])

    table_data_df = df[['id', 'feedback_text', 'sentiment', 'rating', 'topic']]
    table_data_json_string = table_data_df.to_json(orient='records')
    table_data = json.loads(table_data_json_string)

    sentiments_list = df['sentiment'].tolist()
    
    # ... (Giữ nguyên phần lưu báo cáo tự động và trả về JSON) ...
    report_name = f"Phân tích File: {file.filename}"
    report_data = {
        'metrics': metrics,
        'table_data': table_data,
        'sentiments': sentiments_list,
        'ratings': ratings_list,
    }
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"INSERT INTO {REPORT_DB_TABLE} (report_name, analysis_data, user_id) VALUES (?, ?, ?)", 
                       (report_name, json.dumps(report_data, ensure_ascii=False), current_user.id))
        report_id = cursor.lastrowid
        conn.commit()
    except Exception as e:
        app.logger.error(f"Lỗi lưu báo cáo: {e}")
    
    return jsonify({
        'status': 'success',
        'report_name': report_name,
        'metrics': metrics,
        'table_data': table_data,
        'sentiments': sentiments_list,
        'ratings': ratings_list,
    })


# ----------------------------------------------------------------
# --- API: Xem trước Dữ liệu (ĐÃ CẬP NHẬT CHO DOCX) ---
# ----------------------------------------------------------------

@app.route('/api/preview_file', methods=['POST'])
@login_required
def preview_file():
    """Đọc và xem trước 10 dòng đầu tiên của file CSV, Excel, DOCX."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'Không tìm thấy file tải lên.'}), 400

    file = request.files['file']

    if file.filename == '' or not file.filename.lower().endswith(('.csv', '.xlsx', '.xls', '.docx')):
        return jsonify({'status': 'error', 'error': 'Tên file không hợp lệ hoặc định dạng không được hỗ trợ.'}), 400
        
    df = None
    filename = file.filename
    
    try:
        
        if filename.lower().endswith('.csv'):
            # Logic đọc CSV (có thử nhiều encoding)
            file.seek(0)
            file_content = file.read()
            encodings_to_try = ['utf-8', 'latin-1', 'cp1258']
            for encoding in encodings_to_try:
                try:
                    file_stream = io.StringIO(file_content.decode(encoding))
                    
                    try:
                        df = pd.read_csv(file_stream)
                    except pd.errors.ParserError:
                         file_stream.seek(0)
                         df = pd.read_csv(file_stream, sep=';') 
                    
                    if df is not None:
                        break 
                except Exception:
                    continue 

        elif filename.lower().endswith(('.xlsx', '.xls')):
            # Logic đọc file Excel
            file.seek(0)
            df = pd.read_excel(file)
            
        elif filename.lower().endswith('.docx'):
            # <<< LOGIC MỚI: Đọc file DOCX, phân tách từng dòng/đoạn
            doc_lines = extract_text_from_docx(file)
            
            if doc_lines:
                preview_data = []
                for i, line in enumerate(doc_lines):
                    if i >= 10: break # Chỉ lấy 10 dòng đầu
                    
                    # Cố gắng tách text và rating để preview trông đẹp hơn
                    match = re.search(r',\s*(\d{1,2}(?:\.\d{1})?)\s*$', line) 
                    text_to_show = line
                    rating_to_show = 'N/A'
                    
                    if match:
                         text_to_show = line[:match.start()].strip()
                         rating_to_show = match.group(1)
                         
                    preview_data.append({
                        "ID": i + 1,
                        "Văn bản Phản hồi": text_to_show[:100] + '...' if len(text_to_show) > 100 else text_to_show,
                        "Rating Gốc": rating_to_show
                    })
                
                df = pd.DataFrame(preview_data)
            else:
                 return jsonify({'status': 'error', 'error': 'Lỗi: Không thể trích xuất văn bản từ file DOCX hoặc file trống.'}), 500
        
        if df is None or df.empty:
             return jsonify({'status': 'error', 'error': 'Không thể đọc dữ liệu từ file.'}), 500

    except Exception as e:
        app.logger.error(f"Lỗi xử lý file xem trước: {e}")
        return jsonify({'status': 'error', 'error': f'Lỗi xử lý file: {e}'}), 500

    try:
        # Nếu là CSV/Excel, lấy 10 dòng đầu
        if filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            data_head = df.head(10)
        else:
            # Nếu là DOCX, df đã chứa dữ liệu preview 10 dòng
            data_head = df
            
        preview_data_json_string = data_head.to_json(orient='records')
        preview_data = json.loads(preview_data_json_string) 

        columns = list(df.columns)
        total_rows = len(df)
        
        # Nếu là DOCX, total_rows là số dòng, columns là các cột tự tạo
        if filename.lower().endswith('.docx'):
            columns = ['ID', 'Văn bản Phản hồi', 'Rating Gốc']
            
        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'total_rows': total_rows,
            'columns': columns,
            'data': preview_data
        })
        
    except Exception as e:
        app.logger.error(f"Lỗi chuyển đổi dữ liệu xem trước: {e}")
        return jsonify({'status': 'error', 'error': f'Lỗi tạo preview: {e}'}), 500


# ----------------------------------------------------------------
# --- API: Phân tích Topic & Đề xuất (GEMINI AI - Không thay đổi) ---
# ----------------------------------------------------------------

@app.route('/api/gemini_full_analysis', methods=['POST'])
@login_required
def gemini_full_analysis():
    """✅ Phân tích Topic với Gemini AI - ĐÃ TỐI ƯU (100 feedback, model Flash)."""
    
    if not GEMINI_CLIENT:
        return jsonify({
            "status": "error", 
            "message": "API Key cho Gemini chưa được thiết lập. Vui lòng cấu hình GEMINI_API_KEY."
        }), 503

    data = request.get_json()
    feedbacks_data = data.get('feedbacks', [])
    
    if not feedbacks_data:
        return jsonify({"status": "error", "message": "Không có phản hồi nào được cung cấp."}), 400

    # ✅ LỌC FEEDBACK THÔNG MINH
    valid_feedbacks = [
        f for f in feedbacks_data 
        if f.get('text') and len(f['text'].strip()) > 10
    ]
    
    # Sắp xếp theo độ dài (feedback dài hơn thường chứa nhiều thông tin hơn)
    valid_feedbacks.sort(key=lambda x: len(x['text']), reverse=True)
    
    # ✅ GIỚI HẠN LẠI THÀNH 100 FEEDBACK
    valid_feedbacks = valid_feedbacks[:100]
    
    if not valid_feedbacks:
        return jsonify({
            "status": "error", 
            "message": "Không có phản hồi hợp lệ để phân tích."
        }), 400

    # ✅ PROMPT TỐI ƯU (ngắn gọn hơn)
    prompt_feedbacks = "\n".join([
        f"{item['id']}|{item['text'][:150]}"  # Giới hạn 150 ký tự/feedback
        for item in valid_feedbacks
    ])
    
    prompt = f"""Phân tích {len(valid_feedbacks)} phản hồi khách hàng:

1. Phân loại từng feedback vào 1 trong 6 topic: ['Sản phẩm/Chất lượng', 'Dịch vụ Khách hàng', 'Giá cả/Khuyến mãi', 'Giao hàng/Logistics', 'Website/Ứng dụng', 'Khác']
2. Đưa ra 3 đề xuất hành động cụ thể

Feedback (ID|Text):
---
{prompt_feedbacks}
---

Trả lời đúng JSON format:
{{
  "classified_feedbacks": [{{"id": 1, "topic": "Tên Topic"}}, ...],
  "topics": ["Topic A", "Topic B"],
  "suggestions": ["Đề xuất 1", "Đề xuất 2", "Đề xuất 3"]
}}"""

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.0-flash-exp',  # ✅ Dùng model NHANH hơn
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,  # ✅ Giảm temperature để kết quả ổn định hơn
            )
        )
        
        result = json.loads(response.text)
        
        # ✅ VALIDATE kết quả
        required_keys = ['classified_feedbacks', 'topics', 'suggestions']
        if not all(key in result for key in required_keys):
            raise ValueError("Gemini trả về thiếu trường bắt buộc")
        
        app.logger.info(f"✅ Gemini phân tích thành công {len(valid_feedbacks)} feedback")
        return jsonify({"status": "success", "result": result})

    except json.JSONDecodeError as e:
        app.logger.error(f"LỖI PARSE JSON từ Gemini: {e}")
        return jsonify({
            "status": "error", 
            "message": "Gemini trả về định dạng không hợp lệ. Vui lòng thử lại."
        }), 500
        
    except Exception as e:
        app.logger.error(f"LỖI GỌI GEMINI API: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Lỗi xử lý AI: {str(e)}"
        }), 500


# ----------------------------------------------------------------
# --- CÁC ENDPOINTS REPORT (Không thay đổi) ---
# ----------------------------------------------------------------

@app.route('/api/reports', methods=['GET'])
@login_required
def get_reports():
    """Lấy danh sách các báo cáo đã lưu."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, report_name, analysis_data, created_at FROM {REPORT_DB_TABLE} WHERE user_id = ? ORDER BY created_at DESC", (current_user.id,))
        reports_data = cursor.fetchall()
        
        reports = []
        for r in reports_data:
            report_id, report_name, analysis_data_json, created_at = r
            
            try:
                analysis_data = json.loads(analysis_data_json)
                metrics = analysis_data.get('metrics', {'total_count': 0})
            except (json.JSONDecodeError, TypeError):
                metrics = {'total_count': 0}
            
            reports.append({
                'id': report_id, 
                'name': report_name, 
                'timestamp': created_at, 
                'metrics': {
                    'count': metrics.get('total_count', 0)
                }
            })
        
        return jsonify({'status': 'success', 'reports': reports})
        
    except Exception as e:
        app.logger.error(f"LỖI TẢI DANH SÁCH BÁO CÁO: {e}")
        return jsonify({'status': 'error', 'error': 'Không thể tải danh sách báo cáo.'}), 500 

@app.route('/api/report/<int:report_id>', methods=['GET'])
@login_required
def load_report(report_id):
    """Tải chi tiết một báo cáo."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT report_name, analysis_data FROM {REPORT_DB_TABLE} WHERE id = ? AND user_id = ?", 
                   (report_id, current_user.id))
    report_data = cursor.fetchone()

    if report_data:
        report_name = report_data[0]
        try:
            data = json.loads(report_data[1]) 
            
            return jsonify({
                'status': 'success',
                'report_name': report_name,
                'metrics': data.get('metrics', {}),
                'table_data': data.get('table_data', []),
                'sentiments': data.get('sentiments', []),
                'ratings': data.get('ratings', []),
                'gemini_result': data.get('gemini_result', None) 
            })
        except (json.JSONDecodeError, TypeError) as e:
            app.logger.error(f"Lỗi giải mã JSON báo cáo {report_id}: {e}")
            return jsonify({'status': 'error', 'error': 'Dữ liệu báo cáo bị hỏng.'}), 500
    
    return jsonify({'status': 'error', 'error': 'Không tìm thấy báo cáo.'}), 404

@app.route('/api/report/<int:report_id>', methods=['DELETE'])
@login_required
def delete_report(report_id):
    """Xóa một báo cáo đã lưu."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(f"DELETE FROM {REPORT_DB_TABLE} WHERE id = ? AND user_id = ?", 
                   (report_id, current_user.id))
    rows_affected = cursor.rowcount
    conn.commit()

    if rows_affected > 0:
        return jsonify({'status': 'success', 'message': f'Báo cáo ID {report_id} đã được xóa.'})
    
    return jsonify({'status': 'error', 'error': 'Không tìm thấy báo cáo hoặc bạn không có quyền xóa.'}), 404


@app.route('/api/save_report', methods=['POST'])
@login_required
def save_report():
    """Lưu dữ liệu Dashboard hiện tại thành một báo cáo mới."""
    data = request.json
    
    report_name = data.get('name', f"Báo cáo thủ công của {current_user.username}") 
    
    report_data = {
        'metrics': data.get('metrics', {}),
        'table_data': data.get('table_data', []),
        'sentiments': data.get('sentiments', []),
        'ratings': data.get('ratings', []),
        'gemini_result': data.get('gemini_result', None) 
    }
    
    if not report_data['sentiments']:
         return jsonify({'status': 'error', 'error': 'Không có dữ liệu để lưu báo cáo.'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"INSERT INTO {REPORT_DB_TABLE} (report_name, analysis_data, user_id) VALUES (?, ?, ?)", 
                       (report_name, json.dumps(report_data, ensure_ascii=False), current_user.id))
        report_id = cursor.lastrowid
        conn.commit()
        
        return jsonify({
            'status': 'success',
            'report_id': report_id,
            'report_name': report_name,
            'message': f'Báo cáo "{report_name}" đã được lưu thành công.'
        })
    except Exception as e:
        app.logger.error(f"Lỗi lưu báo cáo thủ công: {e}")
        return jsonify({'status': 'error', 'error': f'Lỗi lưu báo cáo: {e}'}), 500


# ----------------------------------------------------------------
# --- CÁC ENDPOINTS KHÁC (ĐÃ CẬP NHẬT CHO PHOBERT) ---
# ----------------------------------------------------------------

@app.route('/api/analyze_text', methods=['POST'])
@login_required
def analyze_single_text():
    """Phân tích sentiment cho một đoạn văn bản đơn lẻ (Sử dụng PhoBERT)."""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'status': 'error', 'error': 'Vui lòng cung cấp văn bản để phân tích.'}), 400

    try:
        tokenizer, model = load_model()
    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    
    # ✅ CODE CẬP NHẬT CHO PHOBERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # Ánh xạ class ID sang Sentiment
    if predicted_class == 0:
        sentiment = 'negative'
    elif predicted_class == 2:
        sentiment = 'positive'
    else:
        sentiment = 'neutral'
        
    rating = sentiment_to_rating(sentiment)

    return jsonify({
        'status': 'success',
        'sentiment': sentiment,
        'rating': rating,
    })

@app.route('/api/check_model_status', methods=['GET'])
@login_required
def check_model_status():
    """Kiểm tra trạng thái của thư mục mô hình PhoBERT AI."""
    if os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH):
        try:
            tokenizer, model = load_model()
            model_info = str(model.config)
            return jsonify({
                'status': 'success',
                'message': f'Mô hình PhoBERT fine-tuned đã được tải thành công. Device: {DEVICE}.',
                'model_path': MODEL_PATH,
            })
        except Exception as e:
             return jsonify({
                'status': 'info',
                'message': f'Tìm thấy thư mục mô hình tại {MODEL_PATH} nhưng không thể tải: {e}',
            })
    else:
        return jsonify({
            'status': 'info',
            'message': f'Chưa tìm thấy thư mục mô hình PhoBERT tại {MODEL_PATH}. Vui lòng đặt mô hình vào đây.',
        })


# ----------------------------------------------------------------
# --- KHỞI CHẠY APP ---
# ----------------------------------------------------------------

if __name__ == '__main__':
    with app.app_context():
        init_db() 
    if not os.path.exists('model'):
        os.makedirs('model')
        print("Đã tạo thư mục 'model/'. Vui lòng đặt thư mục 'phobert_finetuned' vào đây.")
        
    app.run(debug=True)
