from paddleocr import PaddleOCR
import os

# -------------------------------
# PATHS
# -------------------------------
IMAGE_PATH = r"C:\Users\kevin\NotesAI\input_images\image0013.png"
OUTPUT_PATH = r"C:\Users\kevin\NotesAI\output_text\ocr_output.txt"

print("Reading image from:", IMAGE_PATH)
assert os.path.exists(IMAGE_PATH), "❌ Image path does not exist"

# -------------------------------
# INIT OCR (VERSION SAFE)
# -------------------------------
ocr = PaddleOCR(lang="en")

# -------------------------------
# RUN OCR
# -------------------------------
result = ocr.ocr(IMAGE_PATH)

# -------------------------------
# EXTRACT TEXT (CORRECT WAY)
# -------------------------------
lines = []

if result and isinstance(result, list):
    result_dict = result[0]  # PaddleOCR returns list of dicts

    if "rec_texts" in result_dict:
        lines = result_dict["rec_texts"]

final_text = "\n".join(lines).strip()

# -------------------------------
# FAIL-SAFE
# -------------------------------
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
