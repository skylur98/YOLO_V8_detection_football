from pyexpat import model
from ultralytics import YOLO
import torch

#if you use gpu
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('yolov8x')

results = model.predict('input_videos/test_1.mp4', save=True)

print(results[0])
print("=======================")
for box in results[0].boxes:
    print(box)