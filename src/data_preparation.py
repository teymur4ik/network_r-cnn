import cv2
import numpy as np
from xml.etree import ElementTree as ET
from pathlib import Path

class SchemeDataset:
    def __init__(self, data_dir, annotation_dir):
        self.data_dir = Path(data_dir)
        self.annotation_dir = Path(annotation_dir)
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for ann_file in self.annotation_dir.glob('*.xml'):
            # Парсинг XML аннотаций (PASCAL VOC формат)
            tree = ET.parse(ann_file)
            root = tree.getroot()
            
            img_path = self.data_dir / root.find('filename').text
            boxes = []
            labels = []
            
            for obj in root.iter('object'):
                label = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                
            samples.append({
                'image_path': img_path,
                'boxes': np.array(boxes, dtype=np.float32),
                'labels': np.array(labels)
            })
        return samples