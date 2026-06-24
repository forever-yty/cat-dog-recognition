import sys
import os
import requests
import dashscope
from dashscope import Generation

from flask import Flask, request, jsonify, send_from_directory, Response
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import time
from werkzeug.utils import secure_filename
import cv2
from shutil import copyfile
from torchvision.models import resnet101
import numpy as np
from user_db import init_db, create_user, authenticate_user, get_all_users, delete_user, \
    get_all_knowledge, add_knowledge, update_knowledge, delete_knowledge, \
    get_all_hard_samples, add_hard_sample, delete_hard_sample, \
    get_system_logs, add_system_log, \
    create_feedback, get_all_feedback, update_feedback_status, delete_feedback

# 千问API配置（硬编码密钥，本地调试专用）
DASHSCOPE_API_KEY = "sk-ws-H.REYEHMD.B9hJ.MEQCIC5Dcc2tifhhBS_hQ9xD51XSOHh81gIjH2QlALtEu_QdAiAgngGE5YIbHjpmmnJhQiQPaJKQKf4asMK0n1VRVOE8Fg"
DASHSCOPE_AVAILABLE = True if DASHSCOPE_API_KEY else False

# 配置dashscope API密钥
if DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY

# 宠物相关的系统提示词
SYSTEM_PROMPT = """
你是一个专业的宠物智能助手，擅长解答各种宠物相关问题。

你的任务：
1. 识别用户上传的宠物图片并给出品种信息
2. 回答关于宠物喂养、训练、健康护理等问题
3. 提供品种-specific的建议和信息

支持的宠物品种：
狗：金毛、柴犬、泰迪、哈士奇、萨摩耶、博美、巴哥、比格等
猫：英短、布偶、波斯猫、暹罗猫、橘猫等

请用友好、专业的语气回答问题，确保信息准确可靠。
"""

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# 初始化数据库
init_db()

# 手动添加CORS支持
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    if request.method == "OPTIONS":
        return jsonify(code=200)
    return response

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
RECORD_FOLDER = 'record'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(RECORD_FOLDER, exist_ok=True)

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

# 加载模型
det_model = YOLO("yolov8m.pt")

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

# 文件格式校验函数
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    used_label_areas = []

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = img.crop((x1, y1, x2, y2))
            result = predict_breed(crop)
            current_results.append(result)
            
            label = result["name"]
            score = result["score"]
            display_label = f"{label} ({(score * 100):.1f}%)"

            draw.rectangle([x1, y1, x2, y2], outline="#00D9FF", width=2)
            text_bbox = draw.textbbox((0, 0), display_label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            label_y = y1 - text_h - 4
            if label_y < 0:
                label_y = y2 + 4

            for (lx1, ly1, lx2, ly2) in used_label_areas:
                if not (x1 + text_w < lx1 or x1 > lx2 or label_y + text_h < ly1 or label_y > ly2):
                    label_y = y2 + 4
                    break

            while True:
                conflict = False
                for (lx1, ly1, lx2, ly2) in used_label_areas:
                    if not (x1 + text_w < lx1 or x1 > lx2 or label_y + text_h < ly1 or label_y > ly2):
                        conflict = True
                        label_y = ly2 + 4
                        break
                if not conflict:
                    break

            label_x2 = x1 + text_w + 4
            label_y2 = label_y + text_h + 4
            used_label_areas.append((x1, label_y, label_x2, label_y2))
            draw.rectangle([x1, label_y, label_x2, label_y2], fill=(0, 0, 0, 180))
            draw.text((x1 + 2, label_y + 2), display_label, fill="white", font=font, stroke_width=1, stroke_fill="black")

    img.save(path_out)
    return current_results

def process_video(path_in, path_out):
    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        print(f"错误：无法打开输入视频 {path_in}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"开始处理视频：{w}x{h}，{fps}fps，共{total_frames}帧")

    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(path_out, fourcc, fps, (w, h))
        if not out.isOpened():
            raise Exception("avc1编码不可用")
        print("使用H.264编码(avc1)")
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path_out, fourcc, fps, (w, h))
        print("回退到MP4V编码")

    if not out.isOpened():
        print(f"致命错误：无法创建输出视频 {path_out}")
        cap.release()
        return []

    try:
        font = ImageFont.truetype("simhei.ttf", 14)
    except:
        font = ImageFont.load_default(size=14)

    from collections import defaultdict, Counter

    track_records = defaultdict(list)
    cat_ids = list(range(12))
    dog_ids = list(range(12, 37))
    
    temp_cap = cv2.VideoCapture(path_in)
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = temp_cap.read()
        if not ret:
            break
        frame_count += 1

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        results = det_model.track(pil_img, classes=[15, 16], persist=True, conf=0.4, iou=0.6)

        for r in results:
            if r.boxes.id is None:
                continue
            ids = r.boxes.id.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for tid, box, conf in zip(ids, boxes, confs):
                if conf < 0.4:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = pil_img.crop((x1, y1, x2, y2))
                
                try:
                    x = transform(crop).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out_pred = classify_model(x)
                        prob = torch.softmax(out_pred, dim=1)
                        top1_val, idx = torch.max(prob, 1)
                        top1_val = top1_val.item()
                        idx = idx.item()
                    
                    if top1_val > 0.6:
                        track_records[tid].append(idx)
                except:
                    continue

    temp_cap.release()
    print(f"第一阶段完成，收集到{len(track_records)}个目标的有效预测")

    final = {}
    for tid, idxs in track_records.items():
        if not idxs or len(idxs) < 3:
            final[tid] = "识别中"
            continue
        cnt = Counter(idxs)
        best_idx = cnt.most_common(1)[0][0]
        final[tid] = CLASS_NAMES[best_idx]

    cap.release()
    cap = cv2.VideoCapture(path_in)
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        results = det_model.track(pil_img, classes=[15, 16], persist=True, conf=0.4, iou=0.6)

        for r in results:
            if r.boxes.id is None:
                continue
            ids = r.boxes.id.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy()

            for tid, box in zip(ids, boxes):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                lab = final.get(tid, "识别中")

                draw.rectangle([x1, y1, x2, y2], outline="#00D9FF", width=2)
                tb = draw.textbbox((0, 0), lab, font=font)
                tw, th = tb[2]-tb[0], tb[3]-tb[1]
                
                if y1 - th - 10 > 10:
                    ly = y1 - th - 6
                else:
                    ly = y2 + 6

                lx = max(0, min(x1, w - tw - 10))
                draw.rectangle([lx, ly, lx+tw+4, ly+th+4], fill=(0,0,0,180))
                draw.text((lx+2, ly+2), lab, fill="white", font=font)

        out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(out_frame)

        if frame_count % 10 == 0:
            print(f"处理进度：{frame_count}/{total_frames} ({int(frame_count/total_frames*100)}%)")

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    breeds = list(final.values())
    return breeds

def save_record(ori, out, is_video=False, results=None):
    folder = os.path.join(RECORD_FOLDER, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)

    if results is None:
        results = []

    with open(os.path.join(folder, "result.txt"), "w", encoding="utf-8") as f:
        for result in results:
            if isinstance(result, dict) and "name" in result:
                score = result.get("score", 0.0)
                confidence = f"({(score * 100):.1f}%)"
                f.write(f"{result['name']}{confidence}\n")
            else:
                f.write(f"{result}\n")

    if is_video:
        copyfile(ori, os.path.join(folder, "原视频.mp4"))
        copyfile(out, os.path.join(folder, "识别视频.mp4"))
    else:
        Image.open(ori).save(os.path.join(folder, "原图.jpg"))
        Image.open(out).save(os.path.join(folder, "识别图.jpg"))
    
    return folder



# 用户认证接口
@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        role = data.get('role', 'user')
        
        if len(username) < 3 or len(username) > 20:
            return jsonify({'success': False, 'message': '用户名长度需要在3-20个字符之间'})
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': '密码至少需要6个字符'})
        
        success, message = create_user(username, password, role)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        success, result = authenticate_user(username, password)
        
        if success:
            return jsonify({'success': True, 'user': result})
        else:
            return jsonify({'success': False, 'message': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/users', methods=['GET'])
def api_users():
    try:
        users = get_all_users()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def api_delete_user(user_id):
    try:
        deleted = delete_user(user_id)
        if deleted:
            return jsonify({'success': True, 'message': '用户已删除'})
        else:
            return jsonify({'success': False, 'message': '无法删除该用户'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 图片识别接口
@app.route('/api/img_predict', methods=['POST'])
def api_img_predict():
    try:
        f = request.files['file']
        name = secure_filename(f.filename)
        in_path = os.path.join(UPLOAD_FOLDER, name)
        out_path = os.path.join(RESULT_FOLDER, name)
        f.save(in_path)
        
        results = process_img(in_path, out_path)
        record_folder = save_record(in_path, out_path, is_video=False, results=results)
        
        return jsonify({
            'success': True,
            'result_url': f'/static/results/{name}',
            'original_url': f'/static/uploads/{name}',
            'breeds': results,
            'record_id': os.path.basename(record_folder)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 视频识别接口
@app.route('/api/video_predict', methods=['POST'])
def api_video_predict():
    try:
        f = request.files['file']
        name = secure_filename(f.filename)
        in_path = os.path.join(UPLOAD_FOLDER, name)
        out_path = os.path.join(RESULT_FOLDER, name)
        f.save(in_path)
        
        breeds = process_video(in_path, out_path)
        record_folder = save_record(in_path, out_path, is_video=True, results=breeds)
        
        return jsonify({
            'success': True,
            'result_url': f'/static/results/{name}',
            'original_url': f'/static/uploads/{name}',
            'breeds': breeds,
            'record_id': os.path.basename(record_folder)
        })
    except Exception as e:
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

# 删除识别记录
@app.route('/api/delete_record/<record_id>', methods=['DELETE'])
def api_delete_record(record_id):
    import shutil
    folder_path = os.path.join(RECORD_FOLDER, record_id)
    if not os.path.exists(folder_path):
        return jsonify({'success': False, 'message': '记录不存在'}), 404
    
    try:
        shutil.rmtree(folder_path)
        return jsonify({'success': True, 'message': '删除成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'删除失败: {str(e)}'}), 500

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
        print(f"【千问密钥状态】: {DASHSCOPE_AVAILABLE} , 密钥前缀: {DASHSCOPE_API_KEY[:10]}******")

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

# 数据统计接口
@app.route('/api/stats', methods=['GET'])
def api_stats():
    folders = os.listdir(RECORD_FOLDER) if os.path.exists(RECORD_FOLDER) else []
    total = len(folders)
    image_count = 0
    video_count = 0
    breed_count = {}
    total_confidence = 0.0
    confidence_count = 0
    
    for fd in folders:
        folder_path = os.path.join(RECORD_FOLDER, fd)
        is_video = os.path.exists(os.path.join(folder_path, "原视频.mp4"))
        if is_video:
            video_count += 1
        else:
            image_count += 1
        
        res_path = os.path.join(folder_path, "result.txt")
        if os.path.exists(res_path):
            with open(res_path, "r", encoding="utf-8") as f:
                for line in f:
                    breed = line.strip()
                    if breed:
                        breed_count[breed] = breed_count.get(breed, 0) + 1
                        if '(' in breed and '%)' in breed:
                            try:
                                start = breed.rfind('(') + 1
                                end = breed.rfind('%')
                                confidence = float(breed[start:end])
                                total_confidence += confidence
                                confidence_count += 1
                            except:
                                pass
    
    accuracy = 0
    if confidence_count > 0:
        accuracy = round(total_confidence / confidence_count, 1)
    
    return jsonify({
        'success': True,
        'total': total,
        'image_count': image_count,
        'video_count': video_count,
        'breed_distribution': breed_count,
        'accuracy': accuracy
    })

# 静态资源路由
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/record/<ts>/<fname>')
def record_file(ts, fname):
    return send_from_directory(os.path.join(RECORD_FOLDER, ts), fname)

# 前端页面路由
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/index.html')
def home_page():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/login.html')
def login_page():
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/register.html')
def register_page():
    return send_from_directory(app.static_folder, 'register.html')

@app.route('/img_predict.html')
def img_predict_page():
    return send_from_directory(app.static_folder, 'img_predict.html')

@app.route('/img_result.html')
def img_result_page():
    return send_from_directory(app.static_folder, 'img_result.html')

@app.route('/video_track.html')
def video_track_page():
    return send_from_directory(app.static_folder, 'video_track.html')

@app.route('/video_result.html')
def video_result_page():
    return send_from_directory(app.static_folder, 'video_result.html')

@app.route('/record.html')
def record_page():
    return send_from_directory(app.static_folder, 'record.html')

@app.route('/agent.html')
def agent_page():
    return send_from_directory(app.static_folder, 'agent.html')

@app.route('/batch_predict.html')
def batch_predict_page():
    return send_from_directory(app.static_folder, 'batch_predict.html')

@app.route('/data_analysis.html')
def data_analysis_page():
    return send_from_directory(app.static_folder, 'data_analysis.html')

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

# 知识库API
@app.route('/api/knowledge', methods=['GET'])
def api_get_knowledge():
    try:
        data = get_all_knowledge()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/knowledge', methods=['POST'])
def api_add_knowledge():
    try:
        data = request.get_json()
        success, message = add_knowledge(
            data['breed_name'],
            data['category'],
            data.get('description', ''),
            data.get('baidu_url', '')
        )
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/knowledge/<int:knowledge_id>', methods=['PUT'])
def api_update_knowledge(knowledge_id):
    try:
        data = request.get_json()
        success, message = update_knowledge(
            knowledge_id,
            data['breed_name'],
            data['category'],
            data.get('description', ''),
            data.get('baidu_url', '')
        )
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/knowledge/<int:knowledge_id>', methods=['DELETE'])
def api_delete_knowledge(knowledge_id):
    try:
        deleted = delete_knowledge(knowledge_id)
        return jsonify({'success': deleted, 'message': '删除成功' if deleted else '删除失败'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 难样本API
@app.route('/api/hard_samples', methods=['GET'])
def api_get_hard_samples():
    try:
        data = get_all_hard_samples()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/hard_samples', methods=['POST'])
def api_add_hard_sample():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '未上传文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名不能为空'})
        
        if file:
            filename = secure_filename(file.filename)
            hard_samples_folder = os.path.join(app.static_folder, 'hard_samples')
            os.makedirs(hard_samples_folder, exist_ok=True)
            
            filepath = os.path.join('hard_samples', filename)
            full_path = os.path.join(app.static_folder, filepath)
            file.save(full_path)
            
            original_result = request.form.get('original_result', '')
            actual_breed = request.form.get('actual_breed', '')
            
            sample_id = add_hard_sample(filename, filepath, original_result, actual_breed)
            return jsonify({'success': True, 'message': '上传成功', 'id': sample_id})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/hard_samples/<int:sample_id>', methods=['DELETE'])
def api_delete_hard_sample(sample_id):
    try:
        deleted, filepath = delete_hard_sample(sample_id)
        if deleted and filepath:
            full_path = os.path.join(app.static_folder, filepath)
            if os.path.exists(full_path):
                os.remove(full_path)
        return jsonify({'success': deleted, 'message': '删除成功' if deleted else '删除失败'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 系统日志API
@app.route('/api/system_logs', methods=['GET'])
def api_get_system_logs():
    try:
        limit = int(request.args.get('limit', 100))
        data = get_system_logs(limit)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ==================== 用户反馈 API（已改为query传参，解决#导致405/404报错） ====================
import uuid as _uuid
import time as _time

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        uid = data.get("uid", "")
        feedback_type = data.get("feedback_type", "")
        content = data.get("content", "")
        fid = f"#FB{_uuid.uuid4().hex[:4].upper()}"
        now_time = _time.strftime("%m/%d %H:%M")
        success, msg = create_feedback(fid, uid, feedback_type, content, now_time)
        return jsonify({"success": success, "msg": msg})
    except Exception as e:
        return jsonify({"success": False, "msg": str(e)})

# 管理员获取全部反馈
@app.route('/api/get_all_feedback', methods=['GET'])
def get_all_feedback_api():
    try:
        feedback_list = get_all_feedback()
        map_type = {
            "功能建议":"suggestion",
            "问题反馈":"bug",
            "其他":"other"
        }
        res_list = []
        for item in feedback_list:
            res_list.append({
                "fid": item["fid"],
                "uid": item["uid"],
                "type": map_type.get(item["type"], item["type"]),
                "content": item["content"],
                "status": item["status"],
                "time": item["time"]
            })
        return jsonify({"success": True, "list": res_list})
    except Exception as e:
        return jsonify({"success": False, "msg": str(e)})

# 标记反馈已处理（query参数接收fid）
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

# 删除反馈（query参数接收fid，兼容#开头ID）
@app.route('/api/del_feedback', methods=['DELETE'])
def del_feedback():
    try:
        fid = request.args.get("fid", "")
        if not fid:
            return jsonify({"success": False, "msg": "反馈ID不能为空"})
        ok = delete_feedback(fid)
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"success": False, "msg": str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)