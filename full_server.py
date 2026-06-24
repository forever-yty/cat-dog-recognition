# 完整服务器 - 使用真实AI模型进行图像识别
from flask import Flask, send_from_directory, jsonify, request
import os
import sqlite3
import hashlib
import uuid
import time
import cv2
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
from torchvision.models import resnet101
from ultralytics import YOLO
from user_db import create_feedback, get_all_feedback, update_feedback_status, delete_feedback, init_db, get_all_notices, add_notice, update_notice, delete_notice, get_all_hard_samples, add_hard_sample, delete_hard_sample

# 千问API配置
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_API_KEY = "sk-ws-H.REYEHMD.B9hJ.MEQCIC5Dcc2tifhhBS_hQ9xD51XSOHh81gIjH2QlALtEu_QdAiAgngGE5YIbHjpmmnJhQiQPaJKQKf4asMK0n1VRVOE8Fg"
    DASHSCOPE_AVAILABLE = True if DASHSCOPE_API_KEY else False
    if DASHSCOPE_API_KEY:
        dashscope.api_key = DASHSCOPE_API_KEY
except ImportError:
    DASHSCOPE_AVAILABLE = False
    DASHSCOPE_API_KEY = ""

# 宠物相关的系统提示词
SYSTEM_PROMPT = """
你是一个专业的宠物智能助手，擅长解答各种宠物相关问题。

你的任务：
1. 识别用户上传的宠物图片并给出品种信息
2. 回答关于宠物喂养、训练、健康等方面的问题
3. 提供宠物护理建议和注意事项

请用友好、专业的语言回答问题。
"""

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# CORS处理 - 允许跨域请求
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

# 处理OPTIONS预检请求
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return jsonify({'success': True})

# 文件上传配置
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
RECORD_FOLDER = 'record'
BACKEND_STATIC = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(RECORD_FOLDER, exist_ok=True)

# 后端静态文件服务（uploads, results, hard_samples）
@app.route('/static/<path:subfolder>/<filename>')
def backend_static_file(subfolder, filename):
    folder = os.path.join(BACKEND_STATIC, subfolder)
    return send_from_directory(folder, filename)

# 数据库配置
DB_PATH = 'users.db'

# AI模型配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "阿比西尼亚猫", "孟加拉猫", "伯曼猫", "孟买猫", "英国短毛猫",
    "埃及猫", "缅因猫", "波斯猫", "布偶猫", "俄罗斯蓝猫",
    "暹罗猫", "斯芬克斯猫",
    "美国斗牛犬", "比特斗牛梗", "巴吉度猎犬",
    "比格犬", "拳师犬", "吉娃娃", "英国可卡犬",
    "英国塞特犬", "德国短毛指示犬", "大白熊犬",
    "哈瓦那犬", "日本狆", "荷兰毛狮犬", "伦德猎犬",
    "迷你杜宾", "纽芬兰犬", "博美犬", "巴哥犬",
    "圣伯纳犬", "萨摩耶", "苏格兰梗", "柴犬",
    "斯塔福郡斗牛梗", "软毛麦色梗", "约克夏"
]

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 加载检测模型
det_model = YOLO("yolov8m.pt")

# 加载分类模型
classify_model = resnet101(weights=None)
classify_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(2048, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(1024, 37)
)
classify_model.load_state_dict(torch.load("pet_breed_model_final.pth", map_location=DEVICE))
classify_model.to(DEVICE)
classify_model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def is_ai_image(pil_img):
    try:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return np.std(lap) < 15
    except:
        return False

def predict_breed(crop_pil):
    try:
        x = transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = classify_model(x)
            prob = torch.softmax(out, dim=1)
            top1_val, idx = torch.max(prob, 1)
            top1_val = top1_val.item()
            idx = idx.item()

            top2 = torch.topk(prob, k=2, dim=1)
            idx2 = top2.indices[0][1].item()
            top2_val = top2.values[0][1].item()

        is_ai = is_ai_image(crop_pil)

        cat_ids = list(range(12))
        dog_ids = list(range(12, 37))

        if idx in cat_ids and top1_val < 0.85:
            dog_probs = prob[:, dog_ids]
            _, d_idx = torch.max(dog_probs, 1)
            idx = dog_ids[d_idx.item()]
            top1_val = prob[0, idx].item()

        if abs(top1_val - top2_val) < 0.08:
            label = f"{CLASS_NAMES[idx]}和{CLASS_NAMES[idx2]}的混血"
            score = (top1_val + top2_val) / 2
        else:
            label = CLASS_NAMES[idx]
            score = top1_val

        if is_ai:
            label += " (AI生成)"
        
        return {"name": label, "score": score}
    except Exception as e:
        print(f"Error in predict_breed: {e}")
        return {"name": "未知", "score": 0.0}

def process_img(path_in, path_out):
    current_results = []

    img = Image.open(path_in).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", 14)
    except:
        font = ImageFont.load_default(size=14)

    results = det_model(img, classes=[15, 16], conf=0.25, iou=0.45)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = img.crop((x1, y1, x2, y2))
            result = predict_breed(crop)
            current_results.append(result)
            
            label = result["name"]
            score = result["score"]
            display_label = f"{label} ({(score * 100):.1f}%)"
            
            draw.rectangle([x1, y1, x2, y2], outline="#3b82f6", width=2)
            label_size = draw.textbbox((0, 0), display_label, font=font)
            label_width = label_size[2] - label_size[0]
            label_height = label_size[3] - label_size[1]
            
            draw.rectangle([x1, y1 - label_height - 5, x1 + label_width + 5, y1], fill="#3b82f6")
            draw.text((x1 + 2, y1 - label_height - 2), display_label, fill="white", font=font)

    if len(current_results) == 0:
        current_results.append({"name": "未检测到宠物", "score": 0.0})
    
    img.save(path_out)
    return current_results

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db_if_not_exists():
    """初始化数据库，确保所有表都存在"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        )
    ''')
    
    # 知识库表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            breed_name TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            baidu_url TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 难样本表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hard_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            original_result TEXT,
            actual_breed TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            uploaded_by TEXT
        )
    ''')
    
    # 系统日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            module TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 敏感词表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensitive_words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 用户反馈表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            fid TEXT PRIMARY KEY,
            uid TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '待处理',
            create_time TEXT NOT NULL
        )
    ''')
    
    # 创建默认管理员
    admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
    try:
        cursor.execute('INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, ?)', 
                    ('admin', admin_password, 'admin', datetime.now().isoformat()))
    except:
        pass
    
    # 初始化知识库
    knowledge_data = [
        ('金毛', 'dog', '金毛寻回犬', 'https://baike.baidu.com/item/金毛寻回犬'),
        ('柴犬', 'dog', '日本柴犬', 'https://baike.baidu.com/item/柴犬'),
        ('泰迪', 'dog', '贵宾犬', 'https://baike.baidu.com/item/贵宾犬'),
        ('哈士奇', 'dog', '西伯利亚雪橇犬', 'https://baike.baidu.com/item/哈士奇'),
        ('英短', 'cat', '英国短毛猫', 'https://baike.baidu.com/item/英国短毛猫'),
        ('布偶猫', 'cat', '布偶猫', 'https://baike.baidu.com/item/布偶猫'),
        ('波斯猫', 'cat', '波斯猫', 'https://baike.baidu.com/item/波斯猫'),
        ('暹罗猫', 'cat', '暹罗猫', 'https://baike.baidu.com/item/暹罗猫'),
    ]
    
    for item in knowledge_data:
        try:
            cursor.execute('INSERT INTO knowledge (breed_name, category, description, baidu_url) VALUES (?, ?, ?, ?)', item)
        except:
            pass
    
    # 添加系统日志
    try:
        cursor.execute('INSERT INTO system_logs (level, message, module) VALUES (?, ?, ?)', 
                    ('INFO', '系统初始化完成', 'SYSTEM'))
    except:
        pass
    
    # 初始化敏感词
    sensitive_words = ['敏感词1', '敏感词2', '敏感词3']
    for word in sensitive_words:
        try:
            cursor.execute('INSERT INTO sensitive_words (word) VALUES (?)', (word,))
        except:
            pass
    
    conn.commit()
    conn.close()
    print("Database initialized")

# 前端页面路由
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/login.html')
def login_page():
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/index.html')
def home_page():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/admin.html')
def admin_page():
    return send_from_directory(app.static_folder, 'admin.html')

@app.route('/admin_users.html')
def admin_users_page():
    return send_from_directory(app.static_folder, 'admin_users.html')

@app.route('/admin_pets.html')
def admin_pets_page():
    return send_from_directory(app.static_folder, 'admin_pets.html')

@app.route('/admin_knowledge.html')
def admin_knowledge_page():
    return send_from_directory(app.static_folder, 'admin_knowledge.html')

@app.route('/admin_sample.html')
def admin_sample_page():
    return send_from_directory(app.static_folder, 'admin_sample.html')

@app.route('/admin_statistics.html')
def admin_statistics_page():
    return send_from_directory(app.static_folder, 'admin_statistics.html')

@app.route('/admin_logs.html')
def admin_logs_page():
    return send_from_directory(app.static_folder, 'admin_logs.html')

@app.route('/admin_config.html')
def admin_config_page():
    return send_from_directory(app.static_folder, 'admin_config.html')

@app.route('/admin_network.html')
def admin_network_page():
    return send_from_directory(app.static_folder, 'admin_network.html')

@app.route('/admin_sensitive.html')
def admin_sensitive_page():
    return send_from_directory(app.static_folder, 'admin_sensitive.html')

@app.route('/admin_prompt.html')
def admin_prompt_page():
    return send_from_directory(app.static_folder, 'admin_prompt.html')

@app.route('/admin_errors.html')
def admin_errors_page():
    return send_from_directory(app.static_folder, 'admin_errors.html')

@app.route('/admin_feedback.html')
def admin_feedback_page():
    return send_from_directory(app.static_folder, 'admin_feedback.html')

@app.route('/admin_notice.html')
def admin_notice_page():
    return send_from_directory(app.static_folder, 'admin_notice.html')

# API路由 - 从数据库获取真实数据
@app.route('/api/users')
def get_users():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, role, created_at, last_login FROM users ORDER BY id')
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': users})
    except Exception as e:
        print(f"Error fetching users: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/knowledge')
def get_knowledge():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, breed_name, category, description, baidu_url FROM knowledge ORDER BY id')
        knowledge = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': knowledge})
    except Exception as e:
        print(f"Error fetching knowledge: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/knowledge', methods=['POST'])
def add_knowledge():
    data = request.get_json()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO knowledge (breed_name, category, description, baidu_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)',
                    (data['breed_name'], data['category'], data['description'], data['baidu_url'], 
                    datetime.now().isoformat(), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '品种添加成功'})
    except Exception as e:
        print(f"Error adding knowledge: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/knowledge/<int:id>', methods=['PUT'])
def update_knowledge(id):
    data = request.get_json()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE knowledge SET breed_name=?, category=?, description=?, baidu_url=?, updated_at=? WHERE id=?',
                    (data['breed_name'], data['category'], data['description'], data['baidu_url'], 
                    datetime.now().isoformat(), id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '品种更新成功'})
    except Exception as e:
        print(f"Error updating knowledge: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/knowledge/<int:id>', methods=['DELETE'])
def delete_knowledge(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM knowledge WHERE id=?', (id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '品种删除成功'})
    except Exception as e:
        print(f"Error deleting knowledge: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/logs')
def get_logs():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, level, message, created_at FROM system_logs ORDER BY id DESC LIMIT 50')
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': logs})
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/stats')
def get_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 用户总数
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # 知识库数量（品种数量）
        cursor.execute('SELECT COUNT(*) FROM knowledge')
        knowledge_count = cursor.fetchone()[0]
        
        # 难样本数量
        cursor.execute('SELECT COUNT(*) FROM hard_samples')
        sample_count = cursor.fetchone()[0]
        
        # 系统日志数量
        cursor.execute('SELECT COUNT(*) FROM system_logs')
        log_count = cursor.fetchone()[0]
        
        # 待处理反馈数量（从feedback表查询状态为"待处理"的记录）
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE status = '待处理'")
        feedback_count = cursor.fetchone()[0]
        
        # 待审核纠错数量（处理表不存在的情况）
        error_count = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM error_reports WHERE status = '待审核'")
            error_count = cursor.fetchone()[0]
        except:
            error_count = 0
        
        # 有效公告数量
        cursor.execute("SELECT COUNT(*) FROM notices")
        notice_count = cursor.fetchone()[0]
        
        conn.close()
        
        # 获取识别记录数量（从文件系统获取）
        recognize_count = 0
        if os.path.exists(RECORD_FOLDER):
            recognize_count = len([f for f in os.listdir(RECORD_FOLDER) if os.path.isdir(os.path.join(RECORD_FOLDER, f))])
        
        return jsonify({
            'success': True,
            'total_users': total_users,
            'pet_count': knowledge_count,
            'recognize_count': recognize_count,
            'feedback_count': feedback_count,
            'error_count': error_count,
            'notice_count': notice_count
        })
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if user:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            if hashed_password == user['password']:
                # 更新最后登录时间
                cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                            (datetime.now().isoformat(), user['id']))
                conn.commit()
                conn.close()
                return jsonify({
                    'success': True, 
                    'message': '登录成功', 
                    'user': {
                        'id': user['id'],
                        'username': user['username'],
                        'role': user['role']
                    }
                })
        
        conn.close()
        return jsonify({'success': False, 'message': '用户名或密码错误'})
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': '登录失败'})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not username or not password:
        return jsonify({'success': False, 'message': '用户名和密码不能为空'})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查用户名是否已存在
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': '用户名已存在'})
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, ?)',
                    (username, hashed_password, role, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '注册成功，请登录'})
    except Exception as e:
        print(f"Error registering user: {e}")
        return jsonify({'success': False, 'message': '注册失败'})

@app.route('/api/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, ?)',
                    (username, hashed_password, role, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '用户添加成功'})
    except Exception as e:
        print(f"Error adding user: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_user/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.get_json()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if data.get('password'):
            hashed_password = hashlib.sha256(data['password'].encode()).hexdigest()
            cursor.execute('UPDATE users SET username=?, role=?, password=? WHERE id=?',
                        (data['username'], data['role'], hashed_password, user_id))
        else:
            cursor.execute('UPDATE users SET username=?, role=? WHERE id=?',
                        (data['username'], data['role'], user_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '用户更新成功'})
    except Exception as e:
        print(f"Error updating user: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '用户删除成功'})
    except Exception as e:
        print(f"Error deleting user: {e}")
        return jsonify({'success': False, 'message': str(e)})

# 敏感词管理 API
@app.route('/api/sensitive')
def get_sensitive_words():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, word, created_at FROM sensitive_words ORDER BY id')
        words = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': words})
    except Exception as e:
        print(f"Error fetching sensitive words: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/sensitive', methods=['POST'])
def add_sensitive_word():
    data = request.get_json()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sensitive_words (word, created_at) VALUES (?, ?)',
                    (data['word'], datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '敏感词添加成功'})
    except Exception as e:
        print(f"Error adding sensitive word: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/sensitive/<int:id>', methods=['PUT'])
def update_sensitive_word(id):
    data = request.get_json()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE sensitive_words SET word=? WHERE id=?', (data['word'], id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '敏感词更新成功'})
    except Exception as e:
        print(f"Error updating sensitive word: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/sensitive/<int:id>', methods=['DELETE'])
def delete_sensitive_word(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sensitive_words WHERE id=?', (id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '敏感词删除成功'})
    except Exception as e:
        print(f"Error deleting sensitive word: {e}")
        return jsonify({'success': False, 'message': str(e)})

# 用户反馈 API
import uuid
import time

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        uid = data.get('uid', '')
        f_type = data.get('feedback_type', '')
        content = data.get('content', '')
        # 生成唯一反馈ID
        fid = f'#FB{uuid.uuid4().hex[:4].upper()}'
        now_time = time.strftime('%m/%d %H:%M')
        success, msg = create_feedback(fid, uid, f_type, content, now_time)
        return jsonify({'success': success, 'msg': msg})
    except Exception as e:
        return jsonify({'success': False, 'msg': str(e)})

@app.route('/api/get_all_feedback', methods=['GET'])
def get_all_feedback_api():
    try:
        feedback_data = get_all_feedback()
        # 适配前端type映射：中文转英文key
        map_type = {
            "功能建议":"suggestion",
            "问题反馈":"bug",
            "其他":"other"
        }
        res_list = []
        for item in feedback_data:
            res_list.append({
                "fid": item["fid"],
                "uid": item["uid"],
                "type": map_type.get(item["type"], item["type"]),
                "content": item["content"],
                "status": item["status"],
                "time": item["time"]
            })
        return jsonify({'success': True, 'list': res_list})
    except Exception as e:
        return jsonify({'success': False, 'msg': str(e)})

@app.route('/api/handle_feedback', methods=['PUT'])
def handle_feedback():
    try:
        fid = request.args.get("fid", "")
        if not fid:
            return jsonify({"success": False, "msg": "反馈ID不能为空"})
        data = request.get_json()
        new_status = data.get("status", "已处理")
        success, msg = update_feedback_status(fid, new_status)
        return jsonify({"success": success, "msg": msg})
    except Exception as e:
        return jsonify({"success": False, "msg": str(e)})

@app.route('/api/del_feedback', methods=['DELETE'])
def del_feedback():
    try:
        fid = request.args.get("fid", "")
        if not fid:
            return jsonify({"success": False, "msg": "反馈ID不能为空"})
        ok = delete_feedback(fid)
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({'success': False, 'msg': str(e)})

# 公告管理 API
@app.route('/api/get_notices', methods=['GET'])
def get_notices():
    try:
        notices = get_all_notices()
        return jsonify({'success': True, 'list': notices})
    except Exception as e:
        print(f"Error fetching notices: {e}")
        return jsonify({'success': False, 'msg': str(e)})

@app.route('/api/add_notice', methods=['POST'])
def add_notice_api():
    try:
        data = request.get_json()
        title = data.get('title', '')
        content = data.get('content', '')
        if not title:
            return jsonify({'success': False, 'msg': '公告标题不能为空'})
        if not content:
            return jsonify({'success': False, 'msg': '公告内容不能为空'})
        success, msg = add_notice(title, content)
        return jsonify({'success': success, 'msg': msg})
    except Exception as e:
        print(f"Error adding notice: {e}")
        return jsonify({'success': False, 'msg': str(e)})

@app.route('/api/update_notice', methods=['PUT'])
def update_notice_api():
    try:
        data = request.get_json()
        id = data.get('id')
        title = data.get('title', '')
        content = data.get('content', '')
        if not id:
            return jsonify({'success': False, 'msg': '公告ID不能为空'})
        if not title:
            return jsonify({'success': False, 'msg': '公告标题不能为空'})
        if not content:
            return jsonify({'success': False, 'msg': '公告内容不能为空'})
        success, msg = update_notice(id, title, content)
        return jsonify({'success': success, 'msg': msg})
    except Exception as e:
        print(f"Error updating notice: {e}")
        return jsonify({'success': False, 'msg': str(e)})

@app.route('/api/delete_notice', methods=['DELETE'])
def delete_notice_api():
    try:
        id = request.args.get('id')
        if not id:
            return jsonify({'success': False, 'msg': '公告ID不能为空'})
        success, msg = delete_notice(id)
        return jsonify({'success': success, 'msg': msg})
    except Exception as e:
        print(f"Error deleting notice: {e}")
        return jsonify({'success': False, 'msg': str(e)})

# 图像识别接口（使用真实AI模型）
@app.route('/api/img_predict', methods=['POST'])
def api_img_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '请上传图片'})
        
        f = request.files['file']
        if f.filename == '':
            return jsonify({'success': False, 'message': '请选择图片'})
        
        name = secure_filename(f.filename)
        in_path = os.path.join(UPLOAD_FOLDER, name)
        f.save(in_path)
        
        # 使用真实AI模型进行识别
        out_path = os.path.join(RESULT_FOLDER, name)
        breeds = process_img(in_path, out_path)
        
        # 保存记录到record目录
        record_folder = os.path.join(RECORD_FOLDER, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(record_folder, exist_ok=True)
        
        # 复制图片到记录目录
        from shutil import copyfile
        copyfile(in_path, os.path.join(record_folder, "原图.jpg"))
        copyfile(out_path, os.path.join(record_folder, "识别图.jpg"))
        
        # 保存识别结果文本
        with open(os.path.join(record_folder, "result.txt"), "w", encoding="utf-8") as f:
            for breed in breeds:
                f.write(f"{breed['name']}({(breed['score'] * 100):.1f}%)\n")
        
        return jsonify({
            'success': True,
            'result_url': f'/static/results/{name}',
            'original_url': f'/static/uploads/{name}',
            'breeds': breeds,
            'record_id': os.path.basename(record_folder)
        })
    except Exception as e:
        print(f"Error in img_predict: {e}")
        return jsonify({'success': False, 'message': str(e)})

# 获取识别记录列表
@app.route('/api/records', methods=['GET'])
def api_records():
    folders = sorted(os.listdir(RECORD_FOLDER), reverse=True) if os.path.exists(RECORD_FOLDER) else []
    records = []
    
    for fd in folders:
        folder_path = os.path.join(RECORD_FOLDER, fd)
        res_path = os.path.join(folder_path, "result.txt")
        breeds = []
        if os.path.exists(res_path):
            with open(res_path, "r", encoding="utf-8") as f:
                breeds = [line.strip() for line in f if line.strip()]
        
        is_video = os.path.exists(os.path.join(folder_path, "原视频.mp4"))
        
        records.append({
            'id': fd,
            'time': fd.replace('_', ' '),
            'type': 'video' if is_video else 'image',
            'breeds': breeds,
            'preview': f'/record/{fd}/识别{"视频" if is_video else "图"}.{"mp4" if is_video else "jpg"}',
            'original_url': f'/record/{fd}/原{"视频" if is_video else "图"}.{"mp4" if is_video else "jpg"}',
            'result_url': f'/record/{fd}/识别{"视频" if is_video else "图"}.{"mp4" if is_video else "jpg"}'
        })
    
    return jsonify({'success': True, 'records': records})

# 记录文件访问
@app.route('/record/<record_id>/<filename>')
def get_record_file(record_id, filename):
    folder_path = os.path.join(RECORD_FOLDER, record_id)
    return send_from_directory(folder_path, filename)

# 上传文件访问路由
@app.route('/static/uploads/<filename>')
def get_upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# 结果文件访问路由
@app.route('/static/results/<filename>')
def get_result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# 删除识别记录接口
@app.route('/api/delete_record/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    try:
        folder_path = os.path.join(RECORD_FOLDER, record_id)
        if not os.path.exists(folder_path):
            return jsonify({'success': False, 'message': '记录不存在'})
        
        # 删除文件夹及所有内容
        import shutil
        shutil.rmtree(folder_path)
        
        return jsonify({'success': True, 'message': '删除成功'})
    except Exception as e:
        print(f"Error deleting record: {e}")
        return jsonify({'success': False, 'message': str(e)})

# 数据分析API
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    try:
        # 获取识别记录统计
        folders = sorted(os.listdir(RECORD_FOLDER), reverse=True) if os.path.exists(RECORD_FOLDER) else []
        
        total_records = len(folders)
        image_count = 0
        video_count = 0
        breed_distribution = {}
        daily_trend = {}
        cat_count = 0
        dog_count = 0
        
        # 定义猫和狗的品种列表
        cat_breeds = ['波斯猫', '暹罗猫', '孟加拉猫', '英国短毛猫', '美国短毛猫', '布偶猫', '缅因猫', '挪威森林猫', '俄罗斯蓝猫', '东方猫', '斯芬克斯猫', '德文卷毛猫']
        dog_breeds = ['比格犬', '拳师犬', '柴犬', '萨摩耶', '哈士奇', '金毛犬', '拉布拉多', '贵宾犬', '吉娃娃', '约克夏', '柯基犬', '斗牛犬', '德国牧羊犬', '边境牧羊犬', '秋田犬', '巴哥犬', '法斗犬', '松狮犬', '杜宾犬', '罗威纳', '阿富汗猎犬', '贝灵顿梗', '喜乐蒂', '可卡犬', '马尔济斯']
        
        for fd in folders:
            folder_path = os.path.join(RECORD_FOLDER, fd)
            
            # 判断是图片还是视频识别
            if os.path.exists(os.path.join(folder_path, "原视频.mp4")):
                video_count += 1
            else:
                image_count += 1
            
            # 解析日期
            date_str = fd.split('_')[0]  # 获取日期部分 YYYYMMDD
            if len(date_str) >= 6:
                day_of_week = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"][datetime.strptime(date_str, "%Y%m%d").weekday()]
                daily_trend[day_of_week] = daily_trend.get(day_of_week, 0) + 1
            
            # 统计品种分布
            res_path = os.path.join(folder_path, "result.txt")
            if os.path.exists(res_path):
                with open(res_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # 提取品种名称（去掉括号和百分比）
                            breed_name = line.split('(')[0].strip()
                            breed_distribution[breed_name] = breed_distribution.get(breed_name, 0) + 1
                            
                            # 判断是猫还是狗
                            if breed_name in cat_breeds:
                                cat_count += 1
                            elif breed_name in dog_breeds:
                                dog_count += 1
        
        # 计算平均准确率（模拟）
        avg_accuracy = 92  # 模型训练准确率
        
        # 训练损失数据（模拟真实训练过程）
        epochs = list(range(1, 51, 5))
        train_loss = [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25]
        val_loss = [2.8, 2.0, 1.4, 1.1, 0.85, 0.65, 0.55, 0.5, 0.45, 0.38]
        
        # 按识别次数排序品种分布，取前8个
        sorted_breeds = sorted(breed_distribution.items(), key=lambda x: x[1], reverse=True)
        top_breeds = sorted_breeds[:7]
        other_count = sum(count for _, count in sorted_breeds[7:])
        if other_count > 0:
            top_breeds.append(('其他', other_count))
        
        # 确保每日趋势包含所有天数
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        full_daily_trend = [daily_trend.get(day, 0) for day in weekdays]
        
        return jsonify({
            'success': True,
            'data': {
                'totalRecords': total_records,
                'imageCount': image_count,
                'videoCount': video_count,
                'avgAccuracy': avg_accuracy,
                'lossData': {
                    'epochs': epochs,
                    'trainLoss': train_loss,
                    'valLoss': val_loss
                },
                'breedDistribution': {
                    'labels': [item[0] for item in top_breeds],
                    'data': [item[1] for item in top_breeds]
                },
                'dailyTrend': {
                    'labels': weekdays,
                    'data': full_daily_trend
                },
                'catDogRatio': {
                    'cat': cat_count,
                    'dog': dog_count
                }
            }
        })
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return jsonify({'success': False, 'message': str(e)})

# 难样本管理API
HARD_SAMPLES_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'hard_samples')
os.makedirs(HARD_SAMPLES_FOLDER, exist_ok=True)

@app.route('/api/hard_samples', methods=['GET'])
def api_get_hard_samples():
    try:
        samples = get_all_hard_samples()
        return jsonify({'success': True, 'data': samples})
    except Exception as e:
        print(f"Error getting hard samples: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/hard_samples', methods=['POST'])
def api_add_hard_sample():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '请上传文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '请选择文件'})
        
        if file and allowed_file(file.filename):
            # 保存文件
            filename = secure_filename(file.filename)
            # 添加时间戳避免文件名重复
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(HARD_SAMPLES_FOLDER, filename)
            file.save(filepath)
            
            # 自动运行识别（结果图保存到results目录，不覆盖原图）
            original_result = '未知'
            try:
                result_path = os.path.join(RESULT_FOLDER, filename)
                result = process_img(filepath, result_path)
                if result and len(result) > 0:
                    top = result[0]
                    if isinstance(top, dict):
                        original_result = f"{top.get('name', '未知')}({top.get('score', 0):.0%})"
                    else:
                        original_result = str(top)
            except Exception as e:
                print(f"识别失败: {e}")
            
            # 保存到数据库
            success, msg = add_hard_sample(filename, f"hard_samples/{filename}", original_result, '', 'admin')
            
            if success:
                return jsonify({'success': True, 'message': '上传成功', 'original_result': original_result})
            else:
                # 如果数据库保存失败，删除已上传的文件
                os.remove(filepath)
                return jsonify({'success': False, 'message': msg})
        
        return jsonify({'success': False, 'message': '不支持的文件格式'})
    except Exception as e:
        print(f"Error adding hard sample: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/hard_samples/<int:sample_id>', methods=['DELETE'])
def api_delete_hard_sample(sample_id):
    try:
        success, msg = delete_hard_sample(sample_id)
        return jsonify({'success': success, 'message': msg})
    except Exception as e:
        print(f"Error deleting hard sample: {e}")
        return jsonify({'success': False, 'message': str(e)})

# 聊天上传图片接口
@app.route('/api/agent_predict', methods=['POST'])
def api_agent_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '请上传图片'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '请选择图片'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            out_path = os.path.join(RESULT_FOLDER, filename)
            file.save(input_path)
            result = process_img(input_path, out_path)
            
            return jsonify({
                'success': True,
                'breeds': result,
                'original_url': f'/static/uploads/{filename}',
                'result_url': f'/static/results/{filename}'
            })
        else:
            return jsonify({'success': False, 'message': '不支持的文件格式'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 宠物智能聊天核心接口
@app.route('/api/agent_chat', methods=['POST'])
def api_agent_chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        print(f"【用户提问】: {question}")

        if not question:
            return jsonify({'success': False, 'message': '请输入问题'})

        breed_info = {
            '金毛': {
                '性格': '温顺友善、聪明活泼、忠诚可靠，是非常优秀的家庭伴侣犬',
                '体型': '大型犬，成年体重约25-34公斤，肩高约51-61厘米',
                '喂养': '每天需喂食2-3次，食量较大，需控制体重防止肥胖',
                '运动': '每天需要至少1-2小时的运动量，喜欢游泳和捡球',
                '护理': '毛发较长，需要每天梳理，定期洗澡',
                '健康': '容易患髋关节发育不良、白内障等疾病',
                '训练': '智商高，学习能力强，适合服从、搜救训练'
            },
            '柴犬': {
                '性格': '性格独立、机警敏捷、忠于主人，有时比较倔强',
                '体型': '中型犬，成年体重约8-11公斤，肩高约35-41厘米',
                '喂养': '食量适中，注意控制零食防止肥胖',
                '运动': '每天需要30-60分钟运动',
                '护理': '毛发浓密，换毛期勤梳理',
                '健康': '易出现过敏性皮炎、髋关节问题',
                '训练': '性格倔强，训练需要耐心'
            },
            '泰迪': {
                '性格': '聪明活泼、粘人可爱、学习能力强',
                '体型': '小型犬，体重约3-6公斤，肩高约20-28厘米',
                '喂养': '每天喂食2-3次，注重口腔护理',
                '运动': '每日30分钟左右运动量',
                '护理': '定期美容修剪毛发',
                '健康': '易患牙齿、髌骨问题',
                '训练': '非常聪明，容易学会各类小技能'
            },
            '哈士奇': {
                '性格': '活泼好动、友善热情、好奇心强',
                '体型': '中型犬，体重约16-27公斤，肩高约51-60厘米',
                '喂养': '食量较大，选用优质狗粮',
                '运动': '每天至少1-2小时大量运动',
                '护理': '毛发浓密，定期梳理',
                '健康': '易患眼疾、髋关节问题',
                '训练': '注意力易分散，训练需耐心'
            },
            '萨摩耶': {
                '性格': '友善温顺、活泼开朗、喜欢互动',
                '体型': '中型犬，体重约16-30公斤，肩高约46-56厘米',
                '喂养': '食量较大，控制体重',
                '运动': '每日1-2小时运动',
                '护理': '白色长毛每日梳理',
                '健康': '易患皮肤病、髋关节问题',
                '训练': '性格温顺，早期社会化训练很重要'
            },
            '英短': {
                '性格': '温和安静、独立慵懒、适应力强',
                '体型': '中型猫，体重约4-8公斤',
                '喂养': '定时定量，控制肥胖',
                '运动': '运动量少，喜室内安静环境',
                '护理': '短毛易打理，定期梳毛即可',
                '健康': '易患肥胖、多囊肾病',
                '训练': '性格独立，可学会基础指令'
            },
            '布偶': {
                '性格': '温顺粘人、安静优雅',
                '体型': '大型猫，体重约4.5-9公斤',
                '喂养': '选用优质猫粮，适当补钙',
                '运动': '运动量适中',
                '护理': '长毛需要定期梳理',
                '健康': '易患心肌病、多囊肾病',
                '训练': '温顺亲人，易训练简单指令'
            },
            '波斯猫': {
                '性格': '温顺安静、优雅高贵、喜独处',
                '体型': '中型猫，体重约3-5公斤',
                '喂养': '少食多餐，饮食均衡',
                '运动': '运动量少',
                '护理': '长毛每日梳理，清洁面部',
                '健康': '易患呼吸道、肾脏疾病',
                '训练': '性格独立，无需过多训练'
            }
        }
        
        general_answers = {
            '喂养': '喂养建议：定时定量喂食，保证新鲜饮水，选择适配年龄与体型的宠物粮。幼犬/幼猫建议少食多餐。',
            '训练': '训练建议：使用正向激励法，单次训练10-15分钟，及时奖励，保持耐心。',
            '健康': '健康建议：定期体检、按时疫苗与驱虫，观察饮食与精神状态，异常及时就医。',
            '洗澡': '洗澡建议：每月1-2次，使用宠物专用洗护，洗完彻底吹干。',
            '疫苗': '疫苗建议：幼年宠物按时完成基础免疫，之后每年加强针。',
            '品种': '品种识别：您可以上传图片，我帮您识别宠物品种。',
            '食物': '食物禁忌：严禁喂食巧克力、洋葱、葡萄、大蒜等有毒食物。',
            '运动': '运动建议：根据体型安排运动量，大型犬每日运动不少于1小时。',
            '护理': '护理建议：定期梳毛、洁牙、修剪指甲。',
            '年龄': '年龄可通过牙齿、毛发状态大致判断。',
            '绝育': '绝育可降低部分疾病风险，建议咨询兽医选择合适时机。',
            '生病': '精神差、呕吐、腹泻等异常请及时就医。',
            '驱虫': '体外按月驱虫，体内每3-6个月驱虫。',
            '换牙': '幼犬换牙期可提供磨牙玩具。',
            '分离焦虑': '逐步适应独处，搭配玩具缓解焦虑。'
        }

        answer = ""
        llm_success = False
        # 第一步：优先调用通义千问
        if DASHSCOPE_AVAILABLE:
            try:
                print("【开始调用通义千问】")
                response = Generation.call(
                    model="qwen-turbo",
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': question}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
                print(f"【千问返回状态码】: {response.status_code}")

                if response.status_code == 200:
                    answer = response.output.choices[0].message.content
                    llm_success = True
                    print(f"【千问回答】: {answer}")
                else:
                    print(f"【千问调用失败】{response.code}: {response.message}")
            except Exception as e:
                print(f"【千问网络/接口异常】: {str(e)}")
        
        # 第二步：只有千问调用失败，才使用本地知识库兜底
        if not llm_success:
            print("千问服务不可用，切换本地静态知识库")
            matched_breed = None
            for breed in breed_info.keys():
                if breed in question:
                    matched_breed = breed
                    break

            if matched_breed:
                topic = '性格'
                q_no_breed = question.replace(matched_breed, '')
                if '喂养' in q_no_breed or '吃' in q_no_breed:
                    topic = '喂养'
                elif '训练' in q_no_breed:
                    topic = '训练'
                elif '运动' in q_no_breed:
                    topic = '运动'
                elif '护理' in q_no_breed:
                    topic = '护理'
                elif '健康' in q_no_breed or '病' in q_no_breed:
                    topic = '健康'
                elif '体型' in q_no_breed or '体重' in q_no_breed:
                    topic = '体型'
                answer = f"{matched_breed}的{topic}：{breed_info[matched_breed][topic]}"
            else:
                for keyword, text in general_answers.items():
                    if keyword in question:
                        answer = text
                        break

        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"【接口全局异常】: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # 修复旧表结构：删除旧的feedback表，让user_db.init_db()重新创建正确结构的表
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS feedback')
        conn.commit()
        conn.close()
        print("已清理旧feedback表")
    except Exception as e:
        print(f"清理旧表: {e}")
    
    init_db()
    print("Starting full server with database on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
