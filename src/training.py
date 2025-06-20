import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

class SchemeDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms
        
    def __getitem__(self, idx):
        sample = self.dataset.samples[idx]
        image = cv2.imread(str(sample['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = sample['boxes']
        labels = sample['labels']
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
        }
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, target
    
    def __len__(self):
        return len(self.dataset.samples)

def train_model(dataset, num_classes, num_epochs=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Создание модели
    model = create_faster_rcnn_model(num_classes)
    model.to(device)
    
    # Оптимизатор и learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # DataLoader
    dataset = SchemeDatasetTorch(dataset)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # Цикл обучения
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        lr_scheduler.step()
    
    return model