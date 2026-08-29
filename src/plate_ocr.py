import re

import cv2
from fast_plate_ocr import LicensePlateRecognizer

from config import PLATE_OCR_MODEL


def create_reader():
    return LicensePlateRecognizer(PLATE_OCR_MODEL, device="cpu")


def read_plate_text(reader, plate_crop):
    """Run the plate-specific OCR model on a cropped plate image and
    return a cleaned (text, confidence) pair."""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    prediction = reader.run(gray, return_confidence=True)[0]

    text = prediction.plate or ""
    confidence = (
        float(prediction.char_probs.mean())
        if prediction.char_probs is not None
        else 0.0
    )

    text = re.sub(r"[^A-Za-z0-9]", "", text).upper()

    return text, confidence
