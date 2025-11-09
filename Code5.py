import os
import torch
from torchvision import transforms
from PIL import Image

# ---- Import your trained model architecture ----
# (Use whichever file has your ResnetGenerator class)
from Code3 import ResnetGenerator  # or from cyclegan_train_mps import ResnetGenerator

# ---- Select the best device (Apple GPU if available) ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple GPU via MPS")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available, using CPU")

# ---- Paths ----
CHECKPOINT = "/Users/bhuvankumar/PycharmProjects/PythonProject2/.venv/checkpoints/cyclegan_epoch_050.pth"
INPUT_DIR  = "/Users/bhuvankumar/PycharmProjects/PythonProject2/data/B"
OUTPUT_DIR = "outputs/BtoA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load the Generator for B→A ----
G_B2A = ResnetGenerator().to(device)
ckpt = torch.load(CHECKPOINT, map_location=device)
G_B2A.load_state_dict(ckpt["G_B2A"])
G_B2A.eval()

# ---- Define transforms ----
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

# ---- Run inference on all Domain B images ----
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    img = Image.open(in_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_A = G_B2A(x)

    out = transforms.ToPILImage()(denorm(fake_A.squeeze().cpu()))
    out.save(os.path.join(OUTPUT_DIR, fname))
    print(f"✅ Converted: {fname}")

print(f"\n🎉 All B→A translated images saved to: {OUTPUT_DIR}")
