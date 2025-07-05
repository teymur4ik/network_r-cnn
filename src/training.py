# train.py
import os
import yaml
import time
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np

# --- Dataset --------------------------------------------------------------
class DrawingDataset(Dataset):
    def __init__(self, images_dir, ann_file, transforms=None):
        """
        images_dir: путь к папке с PNG
        ann_file: COCO-like JSON с полями images[{id,file_name,...}], annotations[{image_id, bbox, category_id,...}]
        """
        self.images_dir = images_dir
        with open(ann_file) as f:
            data = json.load(f)
        # индексируем
        self.imgs = {img['id']: img for img in data['images']}
        annots = {}
        for a in data['annotations']:
            annots.setdefault(a['image_id'], []).append(a)
        self.annots = annots
        self.ids = list(self.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info   = self.imgs[img_id]
        img_path = os.path.join(self.images_dir, info['file_name'])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # собираем таргет
        ann = self.annots.get(img_id, [])
        boxes = []
        labels = []
        for obj in ann:
            x, y, w_box, h_box = obj['bbox']
            boxes.append([x, y, x + w_box, y + h_box])
            labels.append(obj['category_id'])
        boxes  = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# --- Utils ---------------------------------------------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# --- Main ---------------------------------------------------------------
def main():
    # 1. Конфиг
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")

    # 2. Датасеты и Dataloaders
    ds_train = DrawingDataset(
        cfg['dataset']['train_images'],
        cfg['dataset']['train_anns'],
        transforms=get_transform(train=True)
    )
    ds_val   = DrawingDataset(
        cfg['dataset']['val_images'],
        cfg['dataset']['val_anns'],
        transforms=get_transform(train=False)
    )

    loader_train = DataLoader(
        ds_train, batch_size=cfg['train']['batch_size'],
        shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    loader_val = DataLoader(
        ds_val, batch_size=1,
        shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    # 3. Модель
    num_classes = cfg['model']['num_classes']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=cfg['model']['pretrained'],
        pretrained_backbone=cfg['model']['pretrained']
    )
    # заменяем хэды
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # 4. Оптимизатор и scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['train']['lr'], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 5. Обучение
    best_loss = float('inf')
    for epoch in range(cfg['train']['epochs']):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for images, targets in loader_train:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        epoch_loss /= len(loader_train)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']}  loss={epoch_loss:.4f}  time={elapsed:.0f}s")

        # 6. Validation (простой подсчёт loss на val)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in loader_val:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        val_loss /= len(loader_val)
        print(f" → val_loss={val_loss:.4f}")

        # 7. Сохранение чекпоинта
        ckpt_path = os.path.join(cfg['train']['output_dir'], f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg['train']['output_dir'], "model_best.pth"))

    print("Training finished.")

if __name__ == "__main__":
    main()
