import subprocess
from difflib import SequenceMatcher

# -------------------------------------------------
# Similarity check (prevents hallucinations)
# -------------------------------------------------
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def safe_merge(original, corrected, threshold=0.72):
    """
    Accept a corrected word ONLY if it is similar
    enough to the original word.
    """
    orig_words = original.split()
    corr_words = corrected.split()

    # If LLM changed word count → reject
    if len(orig_words) != len(corr_words):
        return original

    final_words = []
    for o, c in zip(orig_words, corr_words):
        if similarity(o.lower(), c.lower()) >= threshold:
            final_words.append(c)
        else:
            final_words.append(o)  # keep original if risky

    return " ".join(final_words)


# -------------------------------------------------
# Correct a single OCR line (STRICT)
# -------------------------------------------------
def correct_line(line):
    prompt = f"""
You are an OCR spell corrector.

STRICT RULES:
- Fix spelling mistakes ONLY
- Do NOT rephrase
- Do NOT add or remove words
- Do NOT expand abbreviations
- Do NOT replace technical terms
- Keep word count EXACTLY the same
- If unsure, keep the original word
- Return ONLY the corrected line

OCR line:
{line}

Corrected line:
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="ignore",
        capture_output=True
    )

    raw_corrected = result.stdout.strip()

    # Safety merge to prevent hallucinations
    return safe_merge(line, raw_corrected)


# -------------------------------------------------
# Correct full OCR text line-by-line
# -------------------------------------------------
def correct_ocr_text(ocr_text):
    corrected_lines = []

    for line in ocr_text.splitlines():
        line = line.strip()

        if not line:
            corrected_lines.append("")
            continue

        corrected_lines.append(correct_line(line))

    return "\n".join(corrected_lines)
