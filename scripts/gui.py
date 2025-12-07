import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from models.baseline import get_model
from torchvision import transforms

# -------------------- CONFIG --------------------
MODEL_PATH = os.path.join("saved_models", "resnet18_freshness.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- LOAD MODEL --------------------
def load_model(device):
    model = get_model(num_classes=2)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

model = load_model(DEVICE)

# -------------------- TRANSFORMS --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- PREDICTION FUNCTION --------------------
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        messagebox.showerror("Error", f"Cannot open image: {e}")
        return None
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out, 1)
    return "Fresh" if pred.item() == 0 else "Spoiled"

# -------------------- GUI --------------------
class FoodFreshnessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Freshness Detector")
        self.root.geometry("650x600")
        self.root.configure(bg="#f5f5f5")

        # Title
        self.title_label = tk.Label(root, text="Food Freshness Detector", font=("Helvetica", 22, "bold"), bg="#f5f5f5")
        self.title_label.pack(pady=15)

        # Browse Button
        self.btn_browse = ttk.Button(root, text="Browse Image", command=self.browse_image)
        self.btn_browse.pack(pady=10)

        # Image Display Frame
        self.image_frame = tk.Frame(root, bg="#dcdcdc", width=400, height=400)
        self.image_frame.pack(pady=15)
        self.image_label = tk.Label(self.image_frame, bg="#dcdcdc")
        self.image_label.pack(expand=True)

        # Result Label
        self.result_label = tk.Label(root, text="", font=("Helvetica", 20, "bold"), bg="#f5f5f5")
        self.result_label.pack(pady=20)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            return
        self.show_image(file_path)
        result = predict_image(file_path)
        if result == "Fresh":
            self.result_label.config(text=f"Prediction: {result}", fg="green")
        else:
            self.result_label.config(text=f"Prediction: {result}", fg="red")

    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

# -------------------- RUN --------------------
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.configure("TButton", font=("Helvetica", 14), padding=10)
    app = FoodFreshnessApp(root)
    root.mainloop()
