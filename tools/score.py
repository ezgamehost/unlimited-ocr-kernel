"""Backend-independent OCR scoring; raw text is always retained by the runner."""
import re
import html


def normalize_ocr(text: str) -> str:
    """Remove model layout annotations and collapse whitespace, preserving content."""
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|/?ref\|>", "", text)
    text = re.sub(r"</?(?:table|thead|tbody|tfoot|tr|td|th)\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = html.unescape(text)
    return " ".join(text.split())


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance / reference codepoints, including insertions."""
    if not reference:
        if hypothesis:
            raise ValueError("CER is undefined for a nonempty hypothesis and empty reference")
        return 0.0
    # Keep the DP row on the shorter dimension without changing denominator.
    denominator = len(reference)
    left, right = reference, hypothesis
    if len(right) > len(left):
        left, right = right, left
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1] / denominator
