import os
import itertools
import time
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# --- Multiprocessing fix for macOS ---
multiprocessing.set_start_method('spawn', force=True)

# --- Select device (Apple GPU with fallback) ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Apple GPU via MPS backend")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ MPS not available, using CPU")

# ---------- CONFIG ----------
DATA_ROOT = "/Users/bhuvankumar/PycharmProjects/PythonProject2/data"           # expects data/A and data/B
SAVE_DIR = "checkpoints"
SAMPLES_DIR = "samples"
IMG_SIZE = 256
BATCH_SIZE = 2               # keep low for MPS memory
LR = 2e-4
EPOCHS = 50
LAMBDA_CYCLE = 10.0
LAMBDA_ID = 0.5
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
# ----------------------------


# ---------- Dataset ----------
class UnpairedImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.A_dir = os.path.join(root, "A")
        self.B_dir = os.path.join(root, "B")
        self.A_paths = sorted([os.path.join(self.A_dir, f) for f in os.listdir(self.A_dir)])
        self.B_paths = sorted([os.path.join(self.B_dir, f) for f in os.listdir(self.B_dir)])
        self.transform = transform

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

    def __getitem__(self, idx):
        a_path = self.A_paths[idx % len(self.A_paths)]
        b_path = self.B_paths[random.randrange(len(self.B_paths))]

        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("RGB")

        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        return {"A": a_img, "B": b_img}


transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.12), Image.BICUBIC),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = UnpairedImageDataset(DATA_ROOT, transform=transform)
# ✅ macOS fix: num_workers = 0
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
# ----------------------------


# ---------- Models ----------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# --- ResNet Generator ---
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        # Downsample
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, 2, 1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        # Resnet blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        # Upsample
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, 2, 1, 1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# --- PatchGAN Discriminator ---
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# ----------------------------


# ---------- Initialize ----------
G_A2B = ResnetGenerator().to(DEVICE)
G_B2A = ResnetGenerator().to(DEVICE)
D_A = NLayerDiscriminator().to(DEVICE)
D_B = NLayerDiscriminator().to(DEVICE)

G_A2B.apply(weights_init_normal)
G_B2A.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Losses
criterion_GAN = nn.MSELoss().to(DEVICE)
criterion_cycle = nn.L1Loss().to(DEVICE)
criterion_id = nn.L1Loss().to(DEVICE)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=LR, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))
# ----------------------------


# ---------- Training ----------
print(f"Training on: {DEVICE}")
for epoch in range(1, EPOCHS + 1):
    loop = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for i, batch in enumerate(loop):
        real_A = batch["A"].to(DEVICE)
        real_B = batch["B"].to(DEVICE)

        # Adversarial labels
        # Dynamically create label tensors that match the discriminator output shape
        with torch.no_grad():
            pred_shape = D_A(real_A).shape  # get output size dynamically
        valid = torch.ones(pred_shape, device=DEVICE)
        fake = torch.zeros(pred_shape, device=DEVICE)

        # --------------------
        #  Train Generators
        # --------------------
        optimizer_G.zero_grad()

        # Identity
        loss_id_A = criterion_id(G_B2A(real_A), real_A) * LAMBDA_ID
        loss_id_B = criterion_id(G_A2B(real_B), real_B) * LAMBDA_ID

        # GAN loss
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)
        loss_GAN_A2B = criterion_GAN(D_B(fake_B), valid)
        loss_GAN_B2A = criterion_GAN(D_A(fake_A), valid)

        # Cycle loss
        recov_A = G_B2A(fake_B)
        recov_B = G_A2B(fake_A)
        loss_cycle_A = criterion_cycle(recov_A, real_A) * LAMBDA_CYCLE
        loss_cycle_B = criterion_cycle(recov_B, real_B) * LAMBDA_CYCLE

        # Total generator loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminators
        # -----------------------
        optimizer_D_A.zero_grad()
        loss_real = criterion_GAN(D_A(real_A), valid)
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = 0.5 * (loss_real + loss_fake)
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        loss_real = criterion_GAN(D_B(real_B), valid)
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = 0.5 * (loss_real + loss_fake)
        loss_D_B.backward()
        optimizer_D_B.step()

        loop.set_postfix(loss_G=loss_G.item(), D_A=loss_D_A.item(), D_B=loss_D_B.item())

    torch.save({
        "G_A2B": G_A2B.state_dict(),
        "G_B2A": G_B2A.state_dict(),
        "epoch": epoch
    }, os.path.join(SAVE_DIR, f"cyclegan_epoch_{epoch:03d}.pth"))
    print(f"✅ Saved checkpoint for epoch {epoch}")
