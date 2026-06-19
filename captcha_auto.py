"""
works for the black and white distorted letters, 5 letters, set of 17 total letters.

Automated CAPTCHA Solver — Arizona Courts Case Lookup
=====================================================
Uses a trained CNN model to solve CAPTCHAs automatically.
Loops until one is solved, since each wrong answer gives a fresh image.

USAGE:
  python captcha_auto.py
"""

import time
import os
import requests
import cv2
import numpy as np
import torch
import torch.nn as nn
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL       = "https://apps.azcourts.gov/publicaccess/"
PAGE_URL       = BASE_URL + "caselookup.aspx?AspxAutoDetectCookieSupport=1"
CAPTCHA_IMG_ID = "caselookup_ctl00_contentplaceholder1_samplecaptcha_CaptchaImage"
INPUT_NAME     = "ctl00$ContentPlaceHolder1$CaptchaCodeTextBox"
SUBMIT_NAME    = "ctl00$ContentPlaceHolder1$btnCaptcha"
MODEL_PATH     = "captcha_model.pth"
CHARS_PATH     = "captcha_chars.txt"
DEBUG_DIR      = "captcha_debug"
MAX_ATTEMPTS   = 30

# ── Image dimensions (must match training) ────────────────────────────────────
IMG_W, IMG_H = 250, 50
CHAR_W       = IMG_W // 5
SLICE_W      = 45
SLICE_H      = 50
CAPTCHA_LEN  = 5


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — must match train_cnn.py exactly
# ─────────────────────────────────────────────────────────────────────────────

class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
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
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    with open(CHARS_PATH) as f:
        chars = f.read().strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharCNN(num_classes=len(chars)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded — knows {len(chars)} characters: {chars}")
    return model, device, chars


def slice_captcha(gray: np.ndarray) -> list:
    """Split a CAPTCHA image into 5 character strips."""
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


def predict(gray: np.ndarray, model, device, chars) -> str:
    """Predict CAPTCHA text from a grayscale image."""
    slices = slice_captcha(gray)
    result = []
    for strip in slices:
        tensor = torch.tensor(strip, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        with torch.no_grad():
            idx = model(tensor).argmax(dim=1).item()
        result.append(chars[idx])
    return ''.join(result)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_image(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG — save image to captcha_debug/ with guess as filename
# ─────────────────────────────────────────────────────────────────────────────

def save_debug(gray: np.ndarray, guess: str, correct: bool):
    """Save the CAPTCHA image to captcha_debug/ named after the guess.
    Appends _CORRECT if this attempt solved it.
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    suffix = "_CORRECT" if correct else ""
    filename = f"{guess}{suffix}.png"
    path = os.path.join(DEBUG_DIR, filename)

    # Handle duplicates by appending a counter
    counter = 1
    while os.path.exists(path):
        filename = f"{guess}{suffix}_{counter}.png"
        path = os.path.join(DEBUG_DIR, filename)
        counter += 1

    cv2.imwrite(path, gray)
    print(f"  Debug saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load model once before starting the browser
    model, device, chars = load_model()

    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"Debug images will be saved to '{DEBUG_DIR}/'")

    print("Starting browser...")
    driver = webdriver.Chrome()
    driver.get(PAGE_URL)
    wait = WebDriverWait(driver, 10)

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1

        # ── Find the CAPTCHA image ────────────────────────────────────────────
        try:
            img_el = wait.until(
                EC.presence_of_element_located((By.ID, CAPTCHA_IMG_ID))
            )
        except Exception:
            print("Could not find CAPTCHA — refreshing...")
            driver.refresh()
            time.sleep(2)
            continue

        src = img_el.get_attribute("src")
        if not src:
            driver.refresh()
            time.sleep(2)
            continue

        img_url = src if src.startswith("http") else BASE_URL + src

        # ── Download and predict ──────────────────────────────────────────────
        try:
            gray = download_image(img_url)
        except Exception as e:
            print(f"  Download failed: {e}")
            driver.refresh()
            time.sleep(1)
            continue

        guess = predict(gray, model, device, chars)
        print(f"Attempt {attempt:2d}: submitting '{guess}'")

        # ── Type and submit ───────────────────────────────────────────────────
        try:
            text_input = driver.find_element(By.NAME, INPUT_NAME)
            text_input.clear()
            text_input.send_keys(guess)
        except Exception as e:
            print(f"  Could not find text input: {e}")
            break

        try:
            submit_btn = driver.find_element(By.NAME, SUBMIT_NAME)
            submit_btn.click()
        except Exception as e:
            print(f"  Could not find submit button: {e}")
            break

        time.sleep(1.5)

        # ── Check if CAPTCHA is gone (= success) ──────────────────────────────
        try:
            driver.find_element(By.ID, CAPTCHA_IMG_ID)
            # Still on CAPTCHA page — wrong answer
            save_debug(gray, guess, correct=False)
            print(f"           wrong, trying next image...")
        except Exception:
            # CAPTCHA element gone — we solved it
            save_debug(gray, guess, correct=True)
            print(f"\n✓ CAPTCHA solved on attempt {attempt}!")
            print("  Browser is now past the CAPTCHA.")
            input("Press Enter to close the browser...")
            driver.quit()
            return

    print(f"\nGave up after {attempt} attempts.")
    driver.quit()


if __name__ == "__main__":
    main()