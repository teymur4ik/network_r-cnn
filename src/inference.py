# infer.py
import os
import torch
import torchvision
from PIL import Image, ImageDraw
import numpy as np
import yaml

def load_model(cfg, device):
    num_classes = cfg['model']['num_classes']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(cfg['infer']['ckpt'], map_location=device))
    model.to(device).eval()
    return model

def main():
    # Конфиг
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['infer']['device'] if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)

    img = Image.open(cfg['infer']['image_path']).convert("RGB")
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)

    # Инференс
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    # Отрисовка результатов
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
        if score < cfg['infer']['score_thresh']:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{label}:{score:.2f}", fill="red")

    out_path = cfg['infer']['output_path']
    img.save(out_path)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
