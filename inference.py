import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from .model import CLASS_NAMES, CAT_IDS, DOG_IDS, build_classify_model
from .utils import is_ai_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PetBreedClassifier:
    def __init__(self, det_model_path="yolov8m.pt", classify_model_path=None):
        self.det_model = YOLO(det_model_path)
        
        self.classify_model = build_classify_model()
        if classify_model_path and torch.cuda.is_available():
            self.classify_model.load_state_dict(torch.load(classify_model_path))
        elif classify_model_path:
            self.classify_model.load_state_dict(torch.load(classify_model_path, map_location=torch.device('cpu')))
        self.classify_model.to(DEVICE)
        self.classify_model.eval()

    def predict_breed(self, crop_pil):
        try:
            x = transform(crop_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.classify_model(x)
                prob = torch.softmax(out, dim=1)
                top1_val, idx = torch.max(prob, 1)
                top1_val = top1_val.item()
                idx = idx.item()

                top2 = torch.topk(prob, k=2, dim=1)
                idx2 = top2.indices[0][1].item()
                top2_val = top2.values[0][1].item()

            is_ai = is_ai_image(crop_pil)

            if idx in CAT_IDS and top1_val < 0.85:
                dog_probs = prob[:, DOG_IDS]
                _, d_idx = torch.max(dog_probs, 1)
                idx = DOG_IDS[d_idx.item()]
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

    def process_image(self, image_path, output_path):
        current_results = []

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("simhei.ttf", 14)
        except:
            font = ImageFont.load_default(size=14)

        results = self.det_model(img, classes=[15, 16], conf=0.25, iou=0.45)
        used_label_areas = []

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = img.crop((x1, y1, x2, y2))
                result = self.predict_breed(crop)
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

        img.save(output_path)
        return current_results

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开输入视频 {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"开始处理视频：{w}x{h}，{fps}fps，共{total_frames}帧")

        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if not out.isOpened():
                raise Exception("avc1编码不可用")
            print("使用H.264编码(avc1)")
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            print("回退到MP4V编码")

        if not out.isOpened():
            print(f"致命错误：无法创建输出视频 {output_path}")
            cap.release()
            return []

        try:
            font = ImageFont.truetype("simhei.ttf", 14)
        except:
            font = ImageFont.load_default(size=14)

        from collections import defaultdict, Counter

        track_records = defaultdict(list)
        
        temp_cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while frame_count < total_frames:
            ret, frame = temp_cap.read()
            if not ret:
                break
            frame_count += 1

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            results = self.det_model.track(pil_img, classes=[15, 16], persist=True, conf=0.4, iou=0.6)

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
                            out_pred = self.classify_model(x)
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
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            results = self.det_model.track(pil_img, classes=[15, 16], persist=True, conf=0.4, iou=0.6)

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