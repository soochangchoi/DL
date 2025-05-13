import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 모델 정의
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
model = CNNModel(num_classes=10).to(device)
state_dict = torch.load("../project/models/best_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 흑백으로
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 예측할 이미지 로딩
image_path = "../project/0.png"  # ← 여기에 실제 경로 입력
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]

# 예측
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(1).item()

print(f"예측된 클래스: {pred_class}")
