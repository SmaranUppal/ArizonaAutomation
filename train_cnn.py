"""
CAPTCHA CNN Trainer — Character-slice approach
===============================================
Instead of classifying the whole image at once, we:
  1. Slice each CAPTCHA into 5 equal-width character strips
  2. Train a CNN to classify each strip individually
  3. At inference time, slice → classify each strip → join the 5 chars

This turns 200 CAPTCHA images into 1000 individual character examples,
giving the model enough data to actually learn.

USAGE:
  python train_cnn.py
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "captcha_dataset"
MODEL_PATH   = "captcha_model.pth"
CHARS_PATH   = "captcha_chars.txt"   # saves which chars this model knows
IMG_W, IMG_H = 250, 50              # original image size
CHAR_W       = IMG_W // 5           # 50px per character slice
SLICE_W      = 45                   # final width fed to CNN (trimmed slightly)
SLICE_H      = 50                   # final height fed to CNN
CAPTCHA_LEN  = 5
EPOCHS       = 60
BATCH_SIZE   = 32
LEARNING_RATE = 0.001


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — slice a CAPTCHA image into 5 character strips
# ─────────────────────────────────────────────────────────────────────────────

def slice_captcha(gray: np.ndarray) -> list:
    """
    Split a grayscale CAPTCHA image into 5 equal character strips.
    Resizes each strip to (SLICE_W, SLICE_H) for consistent CNN input.
    """
    img = cv2.resize(gray, (IMG_W, IMG_H))
    _, binary = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)

    slices = []
    for i in range(CAPTCHA_LEN):
        x1 = i * CHAR_W
        x2 = x1 + CHAR_W
        strip = binary[:, x1:x2]
        strip = cv2.resize(strip, (SLICE_W, SLICE_H))
        slices.append(strip)
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class CharDataset(Dataset):
    """
    Builds a dataset of individual character slices from CAPTCHA images.
    200 CAPTCHAs × 5 chars = 1000 training examples.
    """
    def __init__(self, data_dir, chars):
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.samples = []   # list of (slice_array, label_idx)

        skipped = 0
        for fname in os.listdir(data_dir):
            if not fname.endswith('.png'):
                continue
            label = fname.replace('.png', '').split('_')[0].upper()
            if len(label) != CAPTCHA_LEN:
                skipped += 1
                continue
            if not all(c in self.char_to_idx for c in label):
                skipped += 1
                continue

            img = cv2.imread(os.path.join(data_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            slices = slice_captcha(img)
            for i, (strip, char) in enumerate(zip(slices, label)):
                self.samples.append((strip, self.char_to_idx[char]))

        print(f"  Built {len(self.samples)} character samples from {data_dir}/ "
              f"({skipped} files skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        strip, label_idx = self.samples[idx]
        tensor = torch.tensor(strip, dtype=torch.float32).unsqueeze(0) / 255.0
        return tensor, label_idx


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — small CNN for single character classification
# ─────────────────────────────────────────────────────────────────────────────

class CharCNN(nn.Module):
    """
    Small CNN that classifies a single character strip.
    Input:  (1, 50, 45) grayscale image
    Output: logits over NUM_CLASSES characters
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # (1,50,45) → (32,25,22)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # (32,25,22) → (64,12,11)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # (64,12,11) → (128,6,5)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 128 * 6 * 5 = 3840
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 6 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 55)
    print("CAPTCHA CNN Trainer — character slice approach")
    print("=" * 55)

    # Discover which characters are actually in the dataset
    all_chars = set()
    for fname in os.listdir(DATASET_DIR):
        label = fname.replace('.png', '').split('_')[0].upper()
        if len(label) == CAPTCHA_LEN:
            all_chars.update(label)
    chars = sorted(all_chars)
    num_classes = len(chars)
    print(f"Characters in dataset ({num_classes}): {''.join(chars)}")

    # Save character list so inference knows the mapping
    with open(CHARS_PATH, 'w') as f:
        f.write(''.join(chars))

    # Build dataset
    full_dataset = CharDataset(DATASET_DIR, chars)
    if len(full_dataset) == 0:
        print("No valid samples found.")
        return

    # 80/20 split
    val_size   = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

    print(f"Train: {train_size} char samples | Val: {val_size} char samples\n")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = CharCNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        char_acc = correct / total * 100
        # Estimate full 5-char match probability: p^5
        full_est = (char_acc / 100) ** 5 * 100
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Loss: {avg_loss:.3f} | "
              f"Char acc: {char_acc:.1f}% | "
              f"~Full match: {full_est:.1f}%")

        if char_acc > best_acc:
            best_acc = char_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"           ✓ Saved (best char acc: {char_acc:.1f}%)")

    full_est = (best_acc / 100) ** 5 * 100
    print(f"\nDone! Best char accuracy: {best_acc:.1f}%")
    print(f"Estimated full-match accuracy: ~{full_est:.1f}%")
    print(f"Model saved to '{MODEL_PATH}'")
    print(f"Character map saved to '{CHARS_PATH}'")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    with open(CHARS_PATH) as f:
        chars = f.read().strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharCNN(num_classes=len(chars)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device, chars


def predict(gray: np.ndarray, model=None, device=None, chars=None) -> str:
    """Predict CAPTCHA text from a grayscale image array."""
    if model is None:
        model, device, chars = load_model()

    slices = slice_captcha(gray)
    result = []
    for strip in slices:
        tensor = torch.tensor(strip, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        with torch.no_grad():
            idx = model(tensor).argmax(dim=1).item()
        result.append(chars[idx])
    return ''.join(result)


if __name__ == "__main__":
    train()