import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Classifier(nn.Module):
    def __init__(self, input_size, h_layer1, h_layer2, h_layer3, output_size):
        self.layer1 = nn.Linear(input_size, h_layer1)
        self.layer2 = nn.Linear(h_layer1, h_layer2)
        self.layer3 = nn.Linear(h_layer2, h_layer3)
        self.out = nn.Linear(h_layer3, output_size)
        self.activation = nn.Relu()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.out(x)

        return x


# initializing the model
models = Classifier(input_size=64*64*3, h_layer1=128, h_layer2=64, h_layer3=32, output_size=1)

#loading the data
data = datasets.ImageFolder(
    root = "./data",
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

data_loader = DataLoader(data, batch_size=32, shuffle=True)

#loss function
criterion = nn.BCELoss()
optimizer = optim.SGD(models.parameters(), lr=0.01)

for epoch in range(10):
    for images, labels in data_loader:
        total_loss = 0
        optimizer.zero_grad()
        output = models(images)
        loss = criterion(output, labels)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(data_loader):.4f}")

# evaluate the model
models.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in data_loader:
        output = models(images)
        predicted = (output > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    pass
