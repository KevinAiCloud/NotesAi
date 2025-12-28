from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import os

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

IMAGE_PATH = r"C:\Users\kevin\OneDrive\NotesAI\input_images\imagea1.jpg"
OUTPUT_PATH = r"C:\Users\kevin\OneDrive\NotesAI\output_text\result.txt"

# ---------- BETTER PREPROCESSING FOR LIGHT HANDWRITING ----------
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# Increase contrast
img = cv2.convertScaleAbs(img, alpha=3.0, beta=50)


# Slight blur to smooth strokes
img = cv2.GaussianBlur(img, (3, 3), 0)

temp_path = "temp_processed.png"
cv2.imwrite(temp_path, img)

# ---------- OCR ----------
image = Image.open(temp_path).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(
    pixel_values,
    max_new_tokens=50,
    num_beams=1,
    do_sample=False
)

text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("✅ OCR completed. Output saved to result.txt")
