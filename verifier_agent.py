# verifier_agent.py

from PIL import Image
import logging
from api_utils import call_model, parse_json_response

# -------- LOGGER SETUP --------
logger = logging.getLogger("Verifier")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def verifier_agent(image_path, steps):

    logger.info(f"Verifier agent started — {len(steps)} step(s)")

    # -------- LOAD IMAGE --------
    img = Image.open(image_path)
    logger.info(f"Loaded image: {image_path} | size={img.size}")

    # -------- PREPARE INPUT --------
    steps_summary = []

    for s in steps:
        text = s.get("text", "")
        latex = s.get("latex", "")

        # Safe truncation (avoid breaking math)
        if len(text) > 200:
            text = text[:200] + "..."

        if len(latex) > 200:
            latex = latex[:200] + "..."

        steps_summary.append({
            "step_id": s.get("step_id"),
            "text": text,
            "latex": latex
        })

    logger.info("Prepared step summaries for verifier")

    # -------- PROMPT --------
    prompt = f"""
You are a strict verification system for handwritten math extraction.

You are given:
1. An image of a handwritten math solution
2. Extracted steps with:
   - text (full OCR content)
   - latex (math-only extraction)

----------------------
TASKS
----------------------

For EACH step:

1. Visual Grounding:
   - Check if BOTH text and latex match what is written in the image
   - text = full content
   - latex = math part only

2. Consistency Check:
   - Verify that latex correctly represents the math in the text
   - Detect missing symbols or incorrect extraction

3. Error Detection:
   Identify:
   - text-latex mismatch
   - missing symbols
   - incorrect numbers or operators
   - extra content not present in image

4. Correction (ONLY if clear):
   - Provide corrected LaTeX strictly based on the image
   - If uncertain → corrected_latex = null

----------------------
STRICT RULES
----------------------

- DO NOT invent steps
- DO NOT solve the problem
- DO NOT improve math
- DO NOT guess corrections
- Prefer marking incorrect over guessing

----------------------
CONFIDENCE
----------------------

- 0.9–1.0 → exact match
- 0.7–0.9 → minor issue
- 0.5–0.7 → mismatch
- <0.5 → incorrect

Visual correctness > logical correctness

----------------------
INPUT STEPS
----------------------

{steps_summary}

----------------------
OUTPUT FORMAT (STRICT JSON)
----------------------

[
  {{
    "step_id": 1,
    "verifier_confidence": 0.0,
    "issues": [],
    "corrected_latex": null
  }}
]

----------------------
FINAL INSTRUCTION
----------------------

- Output ONLY valid JSON
- No extra text
"""

    # -------- CALL MODEL --------
    logger.info("Calling verifier model...")
    response = call_model(contents=[prompt, img], temperature=0.2)

    logger.info(f"Received response ({len(response.text)} chars)")
    logger.debug(f"Raw verifier output:\n{response.text}")

    # -------- PARSE JSON --------
    try:
        result = parse_json_response(response.text)
    except Exception as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.warning("Returning fallback results")

        fallback = [
            {
                "step_id": s.get("step_id"),
                "verifier_confidence": 0.7,
                "issues": [],
                "corrected_latex": None
            }
            for s in steps
        ]

        return fallback, response.text

    # -------- LOG RESULTS --------
    logger.info(f"Verifier returned {len(result)} result(s)")

    for r in result:
        sid = r.get("step_id")
        conf = r.get("verifier_confidence")
        issues = r.get("issues", [])
        corrected = r.get("corrected_latex")

        logger.info(
            f"Step {sid} | conf={conf} | "
            f"issues={len(issues)} | "
            f"corrected={bool(corrected)}"
        )

    logger.info("Verifier agent finished")

    return result, response.text