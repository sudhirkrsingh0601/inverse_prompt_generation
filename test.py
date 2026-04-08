import torch
import open_clip
from PIL import Image
from model import PromptGenerator

device = "cpu"

# Load CLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)
model = model.to(device)
model.eval()

# Load trained model
prompt_model = PromptGenerator().to(device)
prompt_model.load_state_dict(torch.load("checkpoints/prompt_generator.pth", map_location=device))
prompt_model.eval()

# Candidate texts
texts = [
    "a dog",
    "a cat",
    "a car",
    "a flower",
    "a mountain",
    "a person",
    "a building"
]

text_tokens = open_clip.tokenize(texts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

def predict(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        predicted_prompt = prompt_model(image_features)
        predicted_prompt = predicted_prompt / predicted_prompt.norm(dim=-1, keepdim=True)

        similarity = predicted_prompt @ text_features.T
        best_match = texts[similarity.argmax()]

    return best_match

if __name__ == "__main__":
    print("Prediction:", predict("temp.jpg"))

