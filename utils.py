import cv2
import numpy as np
from PIL import Image

def is_ai_image(pil_img):
    try:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return np.std(lap) < 15
    except:
        return False

def draw_label(draw, x1, y1, x2, y2, label, font):
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    label_y = y1 - text_h - 4
    if label_y < 0:
        label_y = y2 + 4

    draw.rectangle([x1, label_y, x1 + text_w + 4, label_y + text_h + 4], fill=(0, 0, 0, 180))
    draw.text((x1 + 2, label_y + 2), label, fill="white", font=font, stroke_width=1, stroke_fill="black")