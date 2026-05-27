# extractor_agent.py

from PIL import Image
import logging
from api_utils import call_model, parse_json_response

logger = logging.getLogger("Extractor")


def extractor_agent(image_path):

    logger.info("Extractor agent started (TEXT + LATEX mode)")

    img = Image.open(image_path)

    prompt = """
You are a strict handwritten math OCR system.

You must extract BOTH:
1. Full text (including words + math)
2. Pure math expressions (LaTeX only)

----------------------
OUTPUT RULES
----------------------

For each step return:

- "text":
  - FULL content exactly as seen
  - includes words + math

- "latex":
  - ONLY mathematical expression
  - NO words
  - NO explanations
  - NO $ symbols

----------------------
EXAMPLES
----------------------

Input:
"Observe that 4y^2 = (2y)^2"

Output:
{
  "text": "Observe that 4y^2 = (2y)^2",
  "latex": "4y^2 = (2y)^2"
}

----------------------

Input:
"(2y - 3)^2 (required factorisation)"

Output:
{
  "text": "(2y - 3)^2 (required factorisation)",
  "latex": "(2y - 3)^2"
}

----------------------
RULES
----------------------

- DO NOT mix text into latex
- DO NOT skip any text
- If no math exists → latex = ""
- If unclear → latex = "[unclear]"

----------------------
OUTPUT FORMAT
----------------------

{
  "problem": "...",
  "steps": [
    {
      "step_id": 1,
      "text": "...",
      "latex": "...",
      "confidence": 0.0
    }
  ],

}

Output ONLY JSON.
"""

    response = call_model(contents=[prompt, img], temperature=0.1)

    result = parse_json_response(response.text)

    logger.info(f"Extracted {len(result.get('steps', []))} steps")

    return result