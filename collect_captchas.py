"""
CAPTCHA Dataset Collector
=========================
Automatically collects labelled CAPTCHA images for training a CNN.

USAGE:
  python collect_captchas.py
  python collect_captchas.py --target 300
"""

import time
import requests
import cv2
import numpy as np
import argparse
import os
import subprocess
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
SAVE_DIR       = "captcha_dataset"
PREVIEW_PATH   = "captcha_preview.png"   # temp file for viewing
WRONG_ANSWER   = "AAAAA"


def download_image(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def show_image(gray: np.ndarray):
    """Save a large version of the CAPTCHA and open it in Windows Photos."""
    large = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(PREVIEW_PATH, large)
    # Open with Windows default image viewer (non-blocking)
    subprocess.Popen(["start", PREVIEW_PATH], shell=True)
    time.sleep(0.5)  # small delay so the image has time to open


def main():
    parser = argparse.ArgumentParser(description="CAPTCHA Dataset Collector")
    parser.add_argument("--target", type=int, default=200,
                        help="How many labelled images to collect (default: 200)")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    existing = [f for f in os.listdir(SAVE_DIR) if f.endswith('.png')]
    collected = len(existing)
    print(f"Already have {collected} images in '{SAVE_DIR}/'")
    print(f"Target: {args.target} total\n")
    print("A preview image will open for each CAPTCHA.")
    print("Type the 5 characters you see, then press Enter.\n")

    if collected >= args.target:
        print("Target already reached!")
        return

    print("Starting browser...")
    driver = webdriver.Chrome()
    driver.get(PAGE_URL)
    wait = WebDriverWait(driver, 10)

    try:
        while collected < args.target:
            remaining = args.target - collected

            # ── Get the CAPTCHA image ─────────────────────────────────────────
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

            try:
                gray = download_image(img_url)
            except Exception as e:
                print(f"Download failed: {e}")
                driver.refresh()
                time.sleep(1)
                continue

            # ── Show it and ask for the label ─────────────────────────────────
            show_image(gray)
            print(f"[{collected+1}/{args.target}] What does this CAPTCHA say? ", end="", flush=True)
            label = input().strip().upper()

            # Validate
            if len(label) != 5 or not label.isalnum():
                print(f"  ✗ '{label}' is not 5 alphanumeric characters — skipping")
            else:
                # Save with label as filename, handle duplicates
                save_path = os.path.join(SAVE_DIR, f"{label}.png")
                counter = 1
                while os.path.exists(save_path):
                    save_path = os.path.join(SAVE_DIR, f"{label}_{counter}.png")
                    counter += 1

                cv2.imwrite(save_path, gray)
                collected += 1
                print(f"  ✓ Saved as '{os.path.basename(save_path)}' — {remaining-1} to go")

            # ── Submit wrong answer to get a fresh CAPTCHA ────────────────────
            try:
                text_input = driver.find_element(By.NAME, INPUT_NAME)
                text_input.clear()
                text_input.send_keys(WRONG_ANSWER)
                submit_btn = driver.find_element(By.NAME, SUBMIT_NAME)
                submit_btn.click()
                time.sleep(1)
            except Exception as e:
                print(f"  Submit failed: {e} — refreshing")
                driver.refresh()
                time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\nStopped early. Collected {collected} images so far.")

    finally:
        # Clean up preview file
        if os.path.exists(PREVIEW_PATH):
            os.remove(PREVIEW_PATH)
        driver.quit()
        print(f"\nDone! {collected} images saved to '{SAVE_DIR}/'")
        print("Next step: run train_cnn.py to train your model.")


if __name__ == "__main__":
    main()