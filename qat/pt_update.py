from ultralytics import YOLO
import torch


model = YOLO('yolov8n.pt')
weights = model.state_dict()

new_weights = {}

for k in weights:
    if "dfl" in k:
        continue
    new_k = k.replace("model.model", "model.model_clear.model")
    new_weights[new_k] = weights[k]

new_weights["model.model_clear.model.22.detect_post.dfl.conv.weight"] = weights["model.model.22.dfl.conv.weight"]

torch.save(new_weights, "yolov8n_v2.pth")
