import os
import torch
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import PromptGenerator

device = "cpu"

# Load CLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)
model = model.to(device)
model.eval()

# Dataset
class ImageDataset(Dataset):
    def __init__(self, folder):
        self.images = [os.path.join(folder, img) for img in os.listdir(folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.images[idx]).convert("RGB"))
        return image

dataset = ImageDataset("dataset/images")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Prompt Generator
prompt_model = PromptGenerator().to(device)

criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(prompt_model.parameters(), lr=1e-4)

losses = []

# Training
for epoch in range(10):
    total_loss = 0

    for image in loader:
        image = image.to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        predicted_prompt = prompt_model(image_features)
        predicted_prompt = predicted_prompt / predicted_prompt.norm(dim=-1, keepdim=True)

        target = torch.ones(image_features.size(0)).to(device)

        loss = criterion(predicted_prompt, image_features, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(prompt_model.state_dict(), "checkpoints/prompt_generator.pth")

print("Model Saved!")

# Plot loss graph
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("loss.png")
plt.show()
