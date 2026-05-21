# Grading of Handwritten Mathematics Using Multi-Agent Extraction and LLM Fine-Tuning with Process Reward Modeling

## Overview

An end-to-end automated system for grading handwritten mathematical submissions. The pipeline combines a **multi-agent confidence fusion architecture** for robust OCR extraction with a **LoRA fine-tuned Qwen2.5-7B-Instruct** model trained on Process Reward Model (PRM) data, enabling step-level grading and pedagogical feedback — without requiring large compute budgets.

---

## Key Features

- **Multi-agent extraction** — OCR, SymPy symbolic validation, and vision verification agents run in parallel and fuse confidence scores
- **Error-type-aware scoring** — distinguishes extraction failures from genuine student mistakes, applying appropriate partial credit
- **Step-level grading** — each reasoning step is scored independently, rewarding correct intermediate work
- **LLM-based semantic adjustment** — fine-tuned model adjusts base scores by up to ±20% with natural-language feedback
- **Efficient fine-tuning** — LoRA (r=8) on a single NVIDIA T4 GPU (16 GB VRAM)

---

## System Architecture

```
Student Answer Sheet (Image / PDF)
        │
        ▼
  1. Preprocessing
     (denoising, binarization, skew correction)
        │
        ▼
  2. Text & Math Extraction
     (OCR + LaTeX parsing)
        │
     ┌──┴──────────────────┐
     ▼                     ▼                    ▼
  3a. OCR Confidence   3b. SymPy Validation   3c. Vision Verification
        │                     │                    │
        └──────────┬──────────┘────────────────────┘
                   ▼
          4. Confidence Fusion
           0.3·OCR + 0.4·SymPy + 0.3·Vision
                   │
                   ▼
          5. Heuristic Base Scoring
           (error-type-aware)
                   │
                   ▼
  6. Fine-tuned Qwen2.5-7B-Instruct Grader
     (±20% semantic adjustment + feedback)
                   │
                   ▼
         Final Score + Feedback Report
```

---

## Multi-Agent Pipeline

| Agent | Role | Output |
|---|---|---|
| **Preprocessing Agent** | Binarisation, deskewing, noise reduction | Clean image |
| **OCR Extractor Agent** | Raw text + per-character confidence | `c_ocr ∈ [0,1]` |
| **SymPy Validation Agent** | Symbolic parse via `sympify` / `parse_latex` | `c_sympy ∈ {0, 0.5, 1}` |
| **Vision Verifier Agent** | Pixel/embedding similarity vs original image | `c_vision ∈ [0,1]` |
| **Grading Agent** | Final score + natural-language feedback | Score tuple + report |

**Confidence Fusion Formula:**

```
c_final = 0.3 × c_ocr + 0.4 × c_sympy + 0.3 × c_vision
```

SymPy receives the highest weight (0.4) because a successfully parsed expression is structurally correct by definition.

---

## Scoring Methodology

### Error-Type Classification

| Error Type | Cause | Credit Policy |
|---|---|---|
| `none` | Correct extraction & math | Full confidence scoring |
| `extraction_error` | OCR / vision failure | Partial credit floor (min 0.4) |
| `student_error` | Actual mathematical mistake | Heavy penalty (factor 0.2) |

### Base Score Formula

```
         m × c_final              if error type = none
s_base = m × max(0.4, c_final)   if error type = extraction_error
         m × 0.2                  if error type = student_error
```

### LLM Score Adjustment

```
s_final = s_base + Δ_AI,    where |Δ_AI| ≤ 0.2 × s_base
```

### Total Score

```
S_total = Σ s_final(i)   for all steps i
```

---

## LLM Fine-Tuning

### Base Model
**Qwen2.5-7B-Instruct** — chosen for strong mathematical reasoning, efficient instruction-following, and suitability for domain-specific fine-tuning.

### Dataset Format (PRM-style)
```json
{
  "question": "Find 3/(sin20)^2 - 1/(cos20)^2 + 64(sin20)^2",
  "intermediate-results": [
    { "result": "...", "step": 1, "score": 1, "branch": 1, "branch-level": 1 },
    { "result": "final answer=2", "step": 7, "score": 7, "branch": "None", "branch-level": "None" }
  ],
  "result": "32"
}
```

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Batch size (per device) | 1 |
| Gradient accumulation steps | 16 (effective batch = 16) |
| Epochs | 3 |
| Learning rate | 2 × 10⁻⁴ |
| Warmup steps | 20 |
| Precision | FP16 |
| Optimizer | Paged AdamW 8-bit |
| Gradient checkpointing | Enabled |
| Evaluation strategy | Every 200 steps |
| Hardware | NVIDIA T4 (16 GB VRAM) |

### Training Objective

```
L = L_CLM + λ · L_PRM
```

- `L_CLM` — standard causal language modelling loss
- `L_PRM` — MSE loss between predicted step score and ground-truth score

---

## Experiments & Results

| Approach | Extraction Quality | Grading Stability |
|---|---|---|
| Traditional OCR | Poor on complex math | — |
| VLM (image-to-LaTeX) | Moderate | — |
| Qwen-70B (untuned) | — | Frequent hallucinations |
| **Qwen2.5-7B-Instruct + LoRA (ours)** | **Robust (multi-agent fusion)** | **Stable, interpretable** |

The fine-tuned 7B model outperformed the untuned 70B model in grading consistency and partial-credit assignment, while using significantly less compute.

---

