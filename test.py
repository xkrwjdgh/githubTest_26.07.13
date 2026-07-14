import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ssl

# macOS에서 MNIST 데이터 다운로드 시 발생하는 SSL 인증서 오류 방지
ssl._create_default_https_context = ssl._create_unverified_context

# 1. GPU/CPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
epochs = 5

# 3. 데이터 로드 및 전처리 (MNIST 데이터셋 사용)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 평균 및 표준편차로 정규화
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 4. CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 첫 번째 합성곱 레이어: 입력 채널 1 (흑백 이미지), 출력 채널 16, 커널 크기 3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # 맥스 풀링 레이어: 커널 크기 2, 스트라이드 2 (크기가 절반으로 감소: 28x28 -> 14x14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 두 번째 합성곱 레이어: 입력 채널 16, 출력 채널 32, 커널 크기 3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # 맥스 풀링 레이어: 크기가 절반으로 감소 (14x14 -> 7x7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 완전 연결 레이어 (Fully Connected Layer)
        # 32개 채널 * 7x7 이미지 크기 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # MNIST는 0~9까지 10개의 클래스
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten: 배치 크기를 제외한 나머지 차원을 1차원으로 펼침
        x = x.view(-1, 32 * 7 * 7)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 모델 객체 생성 및 장치로 이동
model = SimpleCNN().to(device)

# 5. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. 모델 학습 함수
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 순전파 (Forward pass)
        output = model(data)
        
        # 손실 계산
        loss = criterion(output, target)
        
        # 역전파 (Backward pass)
        loss.backward()
        
        # 가중치 업데이트
        optimizer.step()
        
        if batch_idx % 200 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# 7. 모델 평가 함수
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 평가 시에는 그래디언트 계산 비활성화
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0) # 배치 손실 누적
            pred = output.argmax(dim=1, keepdim=True) # 가장 높은 확률을 가진 클래스 예측
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")

# 8. 학습 실행
if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
