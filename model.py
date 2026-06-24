import torch
import torch.nn as nn
from torchvision.models import resnet101

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

CAT_IDS = list(range(12))
DOG_IDS = list(range(12, 37))

def build_classify_model(num_classes=37):
    model = resnet101(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    return model