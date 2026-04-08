import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image

# Create folder
os.makedirs("dataset/images", exist_ok=True)

# Download CIFAR10
transform = transforms.ToTensor()

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Save first 1000 images as JPG
for i in range(1000):
    img, label = dataset[i]
    img = transforms.ToPILImage()(img)
    img.save(f"dataset/images/img_{i}.jpg")

print("Dataset downloaded and saved!")
