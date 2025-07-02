import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Define model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Use log_softmax for NLLLoss

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load data
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=1000,
    shuffle=False
)

# Initialize model, loss, optimizer
model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# Training loop
def train(n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Testing loop
def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Naive Test Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({100. * correct / len(test_loader.dataset):.2f}%)")




if __name__ == "__main__":
    # train(n_epochs=5)
    # model_path = Path(__file__).parent/ "models" / "digit_recon.pt"
    # print(model_path)
    model = torch.jit.load("../models/digit_recon.pt")
    test()
    # torch.save(model.state_dict(), "../models/mnist_digit_model.pth")
    # example_input = torch.randn(1, 1, 28, 28)
    # traced_model = torch.jit.trace(model, example_input)
    # traced_model.save("../models/digit_recon.pt")
