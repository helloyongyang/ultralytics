from ultralytics import YOLO
import torch

model = YOLO('yolov8n.yaml')
weights = torch.load("yolov8n_v2.pth")
model.load_state_dict(weights)

model.train(data='coco.yaml', cfg='custom.yaml')
