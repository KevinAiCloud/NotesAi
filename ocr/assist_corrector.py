import subprocess

def assist_correct_text(ocr_text):
    prompt = f"""
You are an academic OCR correction assistant.

Task:
Correct the OCR text into clean academic notes WHILE preserving the original format.

MANDATORY RULES:
- Keep the SAME line breaks
- Keep the SAME headings
- Keep the SAME numbering and bullet points
- Do NOT merge lines into paragraphs
- Do NOT change the order of lines
- Correct spelling, grammar, and terminology
- Restore standard academic wording where clearly intended
- Do NOT add new concepts
- Do NOT remove information
- Format equations properly if already present
- Output ONLY the corrected text

OCR Text:
{ocr_text}

Corrected Text (same format):
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="ignore",
        capture_output=True
    )

    return result.stdout.strip()
