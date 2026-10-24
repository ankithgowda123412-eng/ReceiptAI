
import cv2
import numpy as np


def preprocess_image(image):
    # 1. Upscale the original image
    height, width = image.shape[:2]

    scale = 2
    resized = cv2.resize(
        image,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC
    )

    # 2. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Improve contrast gently
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # 4. Remove light noise without aggressive cleaning
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 5. Create a clean black-and-white version
    _, binary = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # IMPORTANT:
    # Do not use morphological opening here.
    # It can remove thin characters and small text.

    return resized, binary
