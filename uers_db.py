import sqlite3
import os
import hashlib
from datetime import datetime

DB_FILE = "users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库，创建用户表、知识库表、难样本表、系统日志表和反馈表"""
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
    
    # 知识库表 - 存储品种信息和百度链接
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            breed_name TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,  -- dog 或 cat
            description TEXT,
            baidu_url TEXT,  -- 百度百科链接
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 难样本表 - 存储难以识别的图片
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hard_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            original_result TEXT,  -- 原始识别结果
            actual_breed TEXT,     -- 实际品种
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
    
    # 公告表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
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
        ('哈士奇', 'dog', '西伯利亚雪橇犬', 'https://baike.baidu.com/item/西伯利亚雪橇犬'),
        ('萨摩耶', 'dog', '萨摩耶犬', 'https://baike.baidu.com/item/萨摩耶犬'),
        ('柯基', 'dog', '威尔士柯基犬', 'https://baike.baidu.com/item/威尔士柯基犬'),
        ('英短', 'cat', '英国短毛猫', 'https://baike.baidu.com/item/英国短毛猫'),
        ('布偶', 'cat', '布偶猫', 'https://baike.baidu.com/item/布偶猫'),
        ('橘猫', 'cat', '橘猫', 'https://baike.baidu.com/item/橘猫'),
        ('暹罗', 'cat', '暹罗猫', 'https://baike.baidu.com/item/暹罗猫'),
        ('蓝猫', 'cat', '俄罗斯蓝猫', 'https://baike.baidu.com/item/俄罗斯蓝猫'),
        ('波斯', 'cat', '波斯猫', 'https://baike.baidu.com/item/波斯猫')
    ]
    
    for item in knowledge_data:
        try:
            cursor.execute('INSERT INTO knowledge (breed_name, category, description, baidu_url) VALUES (?, ?, ?, ?)', item)
        except:
            pass
    
    conn.commit()
    conn.close()

# ==================== 用户管理 ====================
def create_user(username, password, role="user"):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",(username,password,role))
        conn.commit()
        conn.close()
        return True, "注册成功"
    except sqlite3.IntegrityError:
        return False, "用户名已存在"

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, username, role FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    if user:
        return True, {"id": user[0], "username": user[1], "role": user[2]}
    return False, "用户名或密码错误"

def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, username, role FROM users")
    data = c.fetchall()
    conn.close()
    return [{"id":x[0], "username":x[1], "role":x[2]} for x in data]

def delete_user(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return True if c.rowcount > 0 else False

# ==================== 知识库管理 ====================
def get_all_knowledge():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM knowledge ORDER BY id DESC')
    data = cursor.fetchall()
    conn.close()
    return [dict(row) for row in data]

def add_knowledge(breed_name, category, description, baidu_url):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO knowledge (breed_name, category, description, baidu_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (breed_name, category, description, baidu_url, datetime.now().isoformat(), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True, '添加成功'
    except sqlite3.IntegrityError:
        return False, '品种已存在'
    except Exception as e:
        return False, str(e)

def update_knowledge(id, breed_name, category, description, baidu_url):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE knowledge SET breed_name=?, category=?, description=?, baidu_url=?, updated_at=?
            WHERE id=?
        ''', (breed_name, category, description, baidu_url, datetime.now().isoformat(), id))
        conn.commit()
        conn.close()
        return True, '更新成功'
    except Exception as e:
        return False, str(e)

def delete_knowledge(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM knowledge WHERE id=?', (id,))
        conn.commit()
        conn.close()
        return True, '删除成功'
    except Exception as e:
        return False, str(e)

# ==================== 难样本管理 ====================
def get_all_hard_samples():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM hard_samples ORDER BY id DESC')
    data = cursor.fetchall()
    conn.close()
    return [dict(row) for row in data]

def add_hard_sample(filename, filepath, original_result, actual_breed, uploaded_by):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO hard_samples (filename, filepath, original_result, actual_breed, created_at, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, filepath, original_result, actual_breed, datetime.now().isoformat(), uploaded_by))
        conn.commit()
        conn.close()
        return True, '上传成功'
    except Exception as e:
        return False, str(e)

def delete_hard_sample(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT filepath FROM hard_samples WHERE id=?', (id,))
        row = cursor.fetchone()
        if row:
            filepath = row['filepath']
            if os.path.exists(filepath):
                os.remove(filepath)
        
        cursor.execute('DELETE FROM hard_samples WHERE id=?', (id,))
        conn.commit()
        conn.close()
        return True, '删除成功'
    except Exception as e:
        return False, str(e)

# ==================== 系统日志 ====================
def get_system_logs(level=None, limit=100):
    conn = get_db_connection()
    cursor = conn.cursor()
    if level:
        cursor.execute('SELECT * FROM system_logs WHERE level=? ORDER BY id DESC LIMIT ?', (level, limit))
    else:
        cursor.execute('SELECT * FROM system_logs ORDER BY id DESC LIMIT ?', (limit,))
    data = cursor.fetchall()
    conn.close()
    return [dict(row) for row in data]

def add_system_log(level, message, module=''):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_logs (level, message, module, created_at)
            VALUES (?, ?, ?, ?)
        ''', (level, message, module, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Failed to add log: {e}")
        return False

# ==================== 用户反馈 ====================
def create_feedback(fid, uid, feedback_type, content, create_time):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO feedback (fid,uid,feedback_type,content,status,create_time) VALUES (?,?,?,?,?,?)",
                  (fid, uid, feedback_type, content, "待处理", create_time))
        conn.commit()
        conn.close()
        return True, "提交成功"
    except Exception as e:
        return False, str(e)

def get_all_feedback():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT fid,uid,feedback_type,content,status,create_time FROM feedback ORDER BY create_time DESC")
    rows = c.fetchall()
    conn.close()
    arr = []
    for r in rows:
        arr.append({
            "fid": r[0],
            "uid": r[1],
            "type": r[2],
            "content": r[3],
            "status": r[4],
            "time": r[5]
        })
    return arr

def update_feedback_status(fid, status):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE feedback SET status=? WHERE fid=?", (status, fid))
        conn.commit()
        conn.close()
        return True, "操作成功"
    except Exception as e:
        return False, str(e)

def delete_feedback(fid):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM feedback WHERE fid=?", (fid,))
    conn.commit()
    conn.close()
    return c.rowcount > 0

# ==================== 敏感词管理 ====================
def get_all_sensitive_words():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sensitive_words ORDER BY id DESC')
    data = cursor.fetchall()
    conn.close()
    return [dict(row) for row in data]

def add_sensitive_word(word):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sensitive_words (word) VALUES (?)', (word,))
        conn.commit()
        conn.close()
        return True, '添加成功'
    except sqlite3.IntegrityError:
        return False, '敏感词已存在'
    except Exception as e:
        return False, str(e)

def delete_sensitive_word(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sensitive_words WHERE id=?', (id,))
        conn.commit()
        conn.close()
        return True, '删除成功'
    except Exception as e:
        return False, str(e)

# ==================== 公告管理 ====================
def get_all_notices():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM notices ORDER BY id DESC')
    data = cursor.fetchall()
    conn.close()
    return [dict(row) for row in data]

def add_notice(title, content):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO notices (title, content, create_time)
            VALUES (?, ?, ?)
        ''', (title, content, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        return True, '发布成功'
    except Exception as e:
        return False, str(e)

def update_notice(id, title, content):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE notices SET title=?, content=? WHERE id=?', (title, content, id))
        conn.commit()
        conn.close()
        return True, '更新成功'
    except Exception as e:
        return False, str(e)

def delete_notice(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM notices WHERE id=?', (id,))
        conn.commit()
        conn.close()
        return True, '删除成功'
    except Exception as e:
        return False, str(e)
