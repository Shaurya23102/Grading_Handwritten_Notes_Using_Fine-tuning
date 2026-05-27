# main_pipeline.py

import logging
import json
import os

from preprocessing_agent import preprocess_image
from extractor_agent import extractor_agent
from sympy_agent import sympy_agent
from verifier_agent import verifier_agent


# -------- LOGGER SETUP --------
logger = logging.getLogger("Pipeline")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# -------- SAVE JSON --------
def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


# -------- CONFIDENCE FUSION --------
def compute_final_confidence(extractor_conf, sympy_conf, verifier_conf, issues, has_math):

    # 🔥 IMPORTANT: if no math, don't penalize heavily
    if not has_math:
        return round(0.6 * extractor_conf + 0.4 * verifier_conf, 2)

    final_conf = (
        0.3 * extractor_conf +
        0.4 * sympy_conf +
        0.3 * verifier_conf
    )

    # -------- GATING --------
    if sympy_conf < 0.5:
        final_conf = min(final_conf, 0.5)

    if sympy_conf < 0.3:
        final_conf = min(final_conf, 0.3)

    if abs(sympy_conf - verifier_conf) > 0.5:
        final_conf *= 0.7

    if verifier_conf < 0.5 and issues:
        final_conf = min(final_conf, 0.4)

    return round(final_conf, 2)


from sympy import simplify
from sympy.parsing.latex import parse_latex

def is_equivalent(expr1, expr2):
    try:
        e1 = parse_latex(expr1)
        e2 = parse_latex(expr2)
        return simplify(e1 - e2) == 0
    except:
        return False

def run_pipeline(image_path):

    logger.info("========== PIPELINE START ==========")
    logger.info(f"Input image: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # -------- STEP 1: PREPROCESS --------
    logger.info("Step 1: Preprocessing image")
    processed_path = preprocess_image(image_path)
    logger.info(f"Processed image saved: {processed_path}")

    # -------- STEP 2: EXTRACTION --------
    logger.info("Step 2: Extractor agent")
    extracted = extractor_agent(processed_path)
    steps = extracted.get("steps", [])

    logger.info(f"Extractor returned {len(steps)} step(s)")
    logger.info(f"Problem: {extracted.get('problem', '')[:100]}")

    save_json(extracted, "extraction_output.json")

    # -------- STEP 3: SYMPY --------
    logger.info("Step 3: SymPy validation")
    sympy_results = sympy_agent(steps)

    save_json({
        "steps": steps,
        "sympy_results": sympy_results
    }, "sympy_output.json")

    # -------- STEP 4: VERIFIER --------
    logger.info("Step 4: Verifier agent")
    verifier_results, raw_verifier = verifier_agent(processed_path, steps)

    save_json({
        "steps": steps,
        "verifier_results": verifier_results
    }, "verifier_output.json")

    with open("verifier_raw.txt", "w") as f:
        f.write(raw_verifier)

    # -------- STEP 5: FUSION --------
    logger.info("Step 5: Fusion + Gating + Correction")

    for i, step in enumerate(steps):

        step_id = step.get("step_id", i + 1)

        text = step.get("text", "")
        latex = step.get("latex", "")

        extractor_conf = step.get("confidence", 0.7)
        sympy_conf = sympy_results[i]["sympy_confidence"]

        verifier_conf = verifier_results[i]["verifier_confidence"]
        issues = verifier_results[i].get("issues", [])
        corrected = verifier_results[i].get("corrected_latex")

        has_math = bool(latex)

        # -------- CORRECTION --------
        if has_math and verifier_conf < 0.5 and issues and corrected:
            step["final_latex"] = corrected
            step["correction_applied"] = True
        else:
            step["final_latex"] = latex
            step["correction_applied"] = False

        # -------- ERROR TYPE --------
        if step["correction_applied"]:

            # If verifier strongly disagrees → extraction issue
            if verifier_conf < 0.5:
                step["error_type"] = "extraction_error"

            else:
                # Now check math consistency with previous step
                if i > 0:
                    prev_latex = steps[i-1].get("final_latex", "")

                    if not is_equivalent(prev_latex, corrected):
                        step["error_type"] = "student_error"
                    else:
                        step["error_type"] = "extraction_error"
                else:
                    step["error_type"] = "student_error"

        else:
            step["error_type"] = "none"
        # -------- CONFIDENCE --------
        final_conf = compute_final_confidence(
            extractor_conf,
            sympy_conf,
            verifier_conf,
            issues,
            has_math
        )

        # -------- STATUS --------
        if final_conf > 0.8:
            status = "trusted"
        elif final_conf > 0.6:
            status = "uncertain"
        else:
            status = "low_confidence"

        # -------- ATTACH --------
        step["final_confidence"] = final_conf
        step["status"] = status
        step["sympy_validation"] = sympy_results[i]
        step["verifier_validation"] = verifier_results[i]

        # -------- LOG --------
        logger.info(
            f"Step {step_id} | "
            f"text='{text[:40]}' | "
            f"latex='{latex}' | "
            f"math={has_math} | "
            f"E={extractor_conf:.2f}, S={sympy_conf:.2f}, V={verifier_conf:.2f} "
            f"→ Final={final_conf:.2f} ({status})"
        )

    # -------- FINAL OUTPUT --------
    output = {
        "problem": extracted.get("problem"),
        "steps": steps
    }

    save_json(output, "final_output.json")

    # -------- SUMMARY --------
    trusted = sum(1 for s in steps if s["status"] == "trusted")
    uncertain = sum(1 for s in steps if s["status"] == "uncertain")
    low = sum(1 for s in steps if s["status"] == "low_confidence")

    logger.info("========== PIPELINE SUMMARY ==========")
    logger.info(f"Total steps: {len(steps)}")
    logger.info(f"Trusted: {trusted}")
    logger.info(f"Uncertain: {uncertain}")
    logger.info(f"Low confidence: {low}")

    logger.info("========== PIPELINE END ==========")

    return output


if __name__ == "__main__":
    IMAGE_PATH = "/media/anish/HDD PART 1/backup/Anish Backup/college/6 Sem/Minor 6/multi-agent/dataset/page_0003.jpg"
    result = run_pipeline(IMAGE_PATH)
    print(json.dumps(result, indent=2))