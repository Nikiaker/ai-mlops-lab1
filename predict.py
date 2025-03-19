import torch
from PIL import Image
from torchvision import transforms
from typing import Any
from main import LightningModel

def predict_image(model_checkpoint: str, image_path: str) -> int:
    model = LightningModel.load_from_checkpoint(model_checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("L")
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        logits = model(tensor_image)
        prediction = logits.argmax(dim=1).item()

    print(f"Predicted digit: {prediction}")
    return prediction

if __name__ == "__main__":
    predict_image("checkpoints/best_model-epoch=03-val_loss=0.15.ckpt", "img.png")