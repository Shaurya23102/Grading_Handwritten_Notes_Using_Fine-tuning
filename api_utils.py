# api_utils.py — Shared API utilities (client, retry, JSON parsing)

from google import genai
from PIL import Image
from dotenv import load_dotenv
import os
import re
import json
import time
import logging

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# -------- API SETUP --------
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not set in environment variables")

client = genai.Client(api_key=API_KEY)

logger = logging.getLogger("API")

MODEL = "gemma-4-31b-it" 

# Max image dimension sent to the API
MAX_IMAGE_DIM = 512


# -------- ROBUST JSON PARSER --------
def parse_json_response(text):
    """Robust JSON extraction from LLM output."""

    # Remove markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # Try direct parse
    try:
        return json.loads(cleaned)
    except:
        pass

    # Fallback: extract JSON object/array
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    logger.error("Failed to parse JSON from model output")
    raise ValueError("Invalid JSON response")


# -------- IMAGE RESIZING --------
def resize_for_api(img):
    """Resize a PIL Image so the longest side is at most MAX_IMAGE_DIM."""
    w, h = img.size

    if max(w, h) <= MAX_IMAGE_DIM:
        return img

    scale = MAX_IMAGE_DIM / max(w, h)
    new_size = (int(w * scale), int(h * scale))

    logger.info(f"Resizing image for API: {w}x{h} → {new_size[0]}x{new_size[1]}")

    return img.resize(new_size, Image.LANCZOS)


def _prepare_contents(contents):
    """Resize any PIL Images in the contents list before sending to API."""
    if isinstance(contents, list):
        return [
            resize_for_api(c) if isinstance(c, Image.Image) else c
            for c in contents
        ]
    return contents


# -------- MODEL CALL WITH RETRY --------
def call_model(contents, temperature=0.1, max_retries=3):
    """Call the model with retry + logging."""

    contents = _prepare_contents(contents)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Calling model ({MODEL}) — attempt {attempt}")

            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": 600,  # reduced for speed
                }
            )

            logger.info("Model call successful")
            return response

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = 30 * attempt
                logger.warning(
                    f"Rate limited (attempt {attempt}/{max_retries}). Waiting {wait}s..."
                )
                time.sleep(wait)

            elif any(x in error_str for x in ["500", "503", "INTERNAL", "UNAVAILABLE"]):
                wait = 15 * attempt
                logger.warning(
                    f"Server error (attempt {attempt}/{max_retries}). Retrying in {wait}s..."
                )
                time.sleep(wait)

            else:
                logger.error(f"Unhandled API error: {e}")
                raise

    raise RuntimeError(f"API calls failed after {max_retries} retries")