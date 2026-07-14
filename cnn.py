import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 이미지 채널: 1 (흑백), 출력 채널: 16, 커널 크기: 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # 출력 채널: 16, 다음 출력 채널: 32, 커널 크기: 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 맥스 풀링: 이미지 크기를 반으로 줄임 (2x2 크기)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 드롭아웃 (과적합 방지)
        self.dropout = nn.Dropout(p=0.25)
        
        # 완전연결층 (Fully Connected Layer)
        # 예: 28x28 이미지 풀링을 2번 거치면 7x7 크기가 됨 (32개 채널 * 7 * 7)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 최종 클래스 10개 (0~9 숫자 인식)

    def forward(self, x):
        # 입력 -> Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # 1차원 벡터로 펼치기 (Flatten)
        x = x.view(-1, 32 * 7 * 7)
        
        # 완전연결층 통과 및 드롭아웃 적용
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def count_parameters(model):
    """모델의 학습 가능한 총 파라미터 개수를 반환합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 모델 생성 및 구조 확인
model = SimpleCNN()
print(model)
print(f"학습 가능한 총 파라미터 수: {count_parameters(model):,}개")

# 테스트용 가상 데이터 주입 (배치 크기 1, 채널 1, 가로 28, 세로 28)
sample_input = torch.randn(1, 1, 28, 28)
output = model(sample_input)
print("\n출력 텐서 형태(Shape):", output.shape) # 예상 결과: torch.Size([1, 10])