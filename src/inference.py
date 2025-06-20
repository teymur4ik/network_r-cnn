def detect_errors(model, image_path, threshold=0.7):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    
    # Загрузка и преобразование изображения
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Фильтрация предсказаний по порогу
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    
    # Отбор только тех предсказаний, которые выше порога
    mask = pred_scores >= threshold
    error_boxes = pred_boxes[mask]
    error_labels = pred_labels[mask]
    error_scores = pred_scores[mask]
    
    return {
        'error_boxes': error_boxes,
        'error_labels': error_labels,
        'error_scores': error_scores,
        'original_image': image
    }