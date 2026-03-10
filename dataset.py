from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.CIFAR10(
    root = "./data",
    train = False,
    download = True,
    transform = transform
)

train_loader = DataLoader(train_dataset, batch_size=32)

plt.figure(figsize=(10,10))
for i in range(10):
    image, label = train_dataset[i]
    plt.subplot(5, 5, i+1)
    plt.imshow(image.permute(1, 2, 0) * 0.5 + 0.5)  # Unnormalize the image
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()