import torch
from torchvision import transforms, utils
from PIL import Image
import os

from Code3 import ResnetGenerator  # or your generator class

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 256
INPUT_DIR = "/Users/bhuvankumar/PycharmProjects/PythonProject2/data/A"  # input images
OUTPUT_DIR = "outputs/AtoB"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load generator
G_A2B = ResnetGenerator().to(DEVICE)
checkpoint = torch.load("checkpoints/cyclegan_epoch_050.pth", map_location=DEVICE)
G_A2B.load_state_dict(checkpoint["G_A2B"])
G_A2B.eval()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def denorm(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".tif")):
        continue
    path = os.path.join(INPUT_DIR, fname)
    img = Image.open(path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        fake = G_A2B(inp)
    out_img = transforms.ToPILImage()(denorm(fake.squeeze()))
    out_img.save(os.path.join(OUTPUT_DIR, fname))
    print("Saved:", fname)

print("✅ All generated images saved to:", OUTPUT_DIR)
