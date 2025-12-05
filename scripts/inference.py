# scripts/inference.py
import os
import torch
from PIL import Image
from torchvision import transforms
from models.baseline import get_model   # matches your baseline.py

MODEL_PATH = os.path.join("saved_models", "resnet18_freshness.pth")  # adjust if different

def load_model(device):
    model = get_model(num_classes=2)     # baseline.py earlier used get_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(model, image_path, device):
    if not os.path.exists(image_path):
        print(f"[ERROR] file not found: {image_path}")
        return None

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out, 1)
    return "fresh" if pred.item() == 0 else "spoiled"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading model...")
    model = load_model(device)
    print("Model loaded. Type 'exit' to quit.\n")

    while True:
        try:
            path = input("Enter image path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if path.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        pred = predict_image(model, path, device)
        if pred is not None:
            print(f">>> Prediction: {pred.upper()}\n")

if __name__ == "__main__":
    main()
