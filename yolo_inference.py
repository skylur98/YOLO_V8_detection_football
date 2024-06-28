from pyexpat import model
from ultralytics import YOLO
import torch
import os
import subprocess

input_file = 'input_videos/test_1.mp4'
output_file = 'output_videos/test_1_compatible.mp4'


subprocess.run(['ffmpeg', '-i', input_file, '-vcodec', 'libx264', '-acodec', 'aac', output_file])


# import torch
# print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}")
# print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}")

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = YOLO('yolov8x')

results = model.predict('input_videos/test_1.mp4', save=True)

print(results[0])
print("=======================")
for box in results[0].boxes:
    print(box)