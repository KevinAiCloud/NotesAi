from paddleocr import PaddleOCR
import os
from preprocess import preprocess_for_ocr
from text_corrector import correct_ocr_text
import re
from assist_corrector import assist_correct_text

def clean_ocr_artifacts(text):
    # Remove stray OCR symbols
    text = re.sub(r"[\\*]+", "", text)

    # Fix spacing around punctuation
    text = re.sub(r"\s+([.,])", r"\1", text)

    return text

# -------------------------------
# PATHS
# -------------------------------
RAW_IMAGE = r"C:\Users\kevin\NotesAI\input_images\Screenshot 2025-12-29 205447.jpg"
PREPROCESSED_IMAGE = r"C:\Users\kevin\NotesAI\input_images\preprocessed.png"
OUTPUT_PATH = r"C:\Users\kevin\NotesAI\output_text\ocr_output.txt"

assert os.path.exists(RAW_IMAGE), "❌ Raw image not found"

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
preprocess_for_ocr(RAW_IMAGE, PREPROCESSED_IMAGE)

# -------------------------------
# INIT OCR
# -------------------------------
ocr = PaddleOCR(lang="en")

# -------------------------------
# RUN OCR
# -------------------------------
result = ocr.ocr(PREPROCESSED_IMAGE)

# -------------------------------
# EXTRACT TEXT (CORRECT WAY)
# -------------------------------
lines = []

if result and isinstance(result, list):
    result_dict = result[0]
    if "rec_texts" in result_dict:
        lines = result_dict["rec_texts"]

final_text = "\n".join(lines).strip()

if not final_text:
    final_text = "[No readable text detected]"

# -------------------------------
# SAVE OUTPUT
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(final_text)

print("✅ OCR completed")
print("------")
print(final_text)

f_text = clean_ocr_artifacts(final_text)
assist_output = assist_correct_text(f_text)

print("✨ Assist Mode Output")
print("------")
print(assist_output)
