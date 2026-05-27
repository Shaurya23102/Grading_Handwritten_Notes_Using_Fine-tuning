# sympy_agent.py

import logging
import re
from sympy.parsing.latex import parse_latex

# -------- LOGGER SETUP --------
logger = logging.getLogger("SymPy")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def is_matrix(latex):
    return "\\begin{bmatrix}" in latex

# -------- LATEX CLEANER --------
def clean_latex(latex):
    """Clean LaTeX string for SymPy parsing."""
    if not latex:
        return latex

    original = latex

    # Remove $ delimiters
    latex = latex.replace("$", "")

    # Remove common words
    latex = re.sub(r"\b(Observe|Therefore|Hence|Thus|So)\b", "", latex)

    # Remove explanatory text in parentheses
    latex = re.sub(r"\([^)]*(required|factorisation|factorization)[^)]*\)", "", latex, flags=re.IGNORECASE)

    # Remove stray text before equations
    latex = re.sub(r"^[A-Za-z\s]+(?==)", "", latex)

    latex = latex.strip()

    logger.debug(f"    Cleaned: '{original}' → '{latex}'")
    return latex


# -------- VALIDATION FUNCTION --------
def validate_expression(step_id, latex_str):
    """Validate LaTeX syntax (OCR stage only)."""

    logger.info(f"  Step {step_id}: validating LaTeX")

    # -------- EMPTY CHECK --------
    if not latex_str or latex_str.strip() == "":
        logger.warning(f"  Step {step_id}: ✗ empty LaTeX string")

        return {
            "math_valid": False,
            "sympy_confidence": 0.0,
            "error": "empty"
        }

    try:
        # -------- CLEAN LATEX --------
        cleaned = clean_latex(latex_str)
        logger.info(f"  Cleaned LaTeX: {cleaned}")

        if cleaned == "" or cleaned == "[unclear]":
            raise ValueError("Empty or unclear expression")

        # =========================================================
        # 🔥 NEW: MATRIX DETECTION (CRITICAL FIX)
        # =========================================================
        if "\\begin{bmatrix}" in cleaned or "\\begin{pmatrix}" in cleaned:
            logger.warning(f"  Step {step_id}: ⚠ matrix detected — skipping SymPy parsing")

            return {
                "math_valid": True,              # treat as valid
                "sympy_confidence": 0.7,         # medium confidence
                "error": "matrix_skipped"
            }

        # =========================================================
        # -------- HANDLE EQUATIONS --------
        # =========================================================
        if "=" in cleaned:
            parts = cleaned.split("=")

            for i, part in enumerate(parts):
                part = part.strip()

                if part == "":
                    raise ValueError("Empty equation side")

                # Skip matrix parts inside equations
                if "\\begin{bmatrix}" in part:
                    logger.warning(f"    Side {i+1}: matrix skipped")
                    continue

                expr = parse_latex(part)
                logger.info(f"    Side {i+1}: ✓ {str(expr)[:60]}")

        else:
            # -------- SINGLE EXPRESSION --------
            expr = parse_latex(cleaned)
            expr_str = str(expr)

            if expr_str.strip() == "":
                raise ValueError("Parsed empty expression")

            logger.info(f"  ✓ valid expression → {expr_str[:80]}")

        return {
            "math_valid": True,
            "sympy_confidence": 0.9,
            "error": None
        }

    except Exception as e:
        logger.warning(f"  Step {step_id}: ✗ parse failed — {e}")

        return {
            "math_valid": False,
            "sympy_confidence": 0.2,
            "error": str(e)
        }
# -------- MAIN AGENT --------
def sympy_agent(steps):
    logger.info(f"SymPy agent started — validating {len(steps)} step(s)")
    results = []

    for step in steps:
        step_id = step.get("step_id", "?")

        # ✅ FIX: define latex first
        latex = step.get("latex", "")

        # Fallback if latex is missing
        if not latex or latex == "[unclear]":
            logger.warning(f"Step {step_id}: latex unclear, falling back to text")
            latex = step.get("text", "")

        res = validate_expression(step_id, latex)
        results.append(res)

    valid_count = sum(1 for r in results if r["math_valid"])
    logger.info(f"SymPy agent finished — {valid_count}/{len(results)} valid")

    return results