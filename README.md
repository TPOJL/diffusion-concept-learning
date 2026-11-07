# Diffusion Model Personalization: LoRA, DreamBooth & Textual Inversion

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B_CUDA-EE4C2C.svg)
![Diffusers](https://img.shields.io/badge/Diffusers-0.36.0-yellow.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA_0.18.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end research and engineering pipeline exploring **few-shot concept personalization** in latent diffusion models (`runwayml/stable-diffusion-v1-5`). 

This project benchmarks three canonical fine-tuning paradigms—**Textual Inversion**, **Low-Rank Adaptation (LoRA)**, and **DreamBooth**—on a custom 10-image dataset (`512×512`) to teach the model a novel compositional concept (`sksfruitface` / `<fruitface>`) while avoiding semantic leakage and catastrophic forgetting.

Full visual comparisons, failure mode analyses, and experimental grids are documented in [`project-report.pdf`](./project-report.pdf).

---

## Target Concept & Compositional Challenge

* **Concept (`fruitface`):** A photorealistic fruit or berry where a single human face is physically integrated/carved into the organic surface, preserving the fruit's native geometry, skin texture, and lighting.
* **Key Compositional Failure Modes:** Pretrained diffusion priors strongly bias `"fruit + human face"` prompts toward disjoint compositions (e.g., a human holding a fruit, or a portrait with fruit in the background). Overcoming this requires precise adapter capacity and prior preservation.

---

## Mathematical & Architectural Overview

### 1. Textual Inversion (Token Embedding Optimization)
Freezes the entire UNet (`ε_θ`) and CLIP text encoder, optimizing only a single new embedding vector `v* ∈ ℝ⁷⁶⁸` in the tokenizer vocabulary for placeholder `<fruitface>` (initialized from `"fruit"`):

    v* = argmin_v  𝔼_{z, y, ε, t} [ ‖ ε - ε_θ(z_t, t, c_θ(y, v)) ‖₂² ]

### 2. Low-Rank Adaptation (LoRA on UNet & CLIP Cross-Attention)
Freezes pretrained weights `W₀ ∈ ℝ^{d×k}` in the UNet cross-attention projection layers (`Q, K, V, Out`) and injects trainable rank-decomposition matrices `A ∈ ℝ^{r×k}` and `B ∈ ℝ^{d×r}` where rank `r ≪ min(d, k)`:

    W = W₀ + ΔW = W₀ + (α / r) · B · A

### 3. DreamBooth with Prior-Preservation Loss
Fine-tunes the full UNet weights using a rare identifier (`sksfruitface`) paired with class noun `"fruit"`. To prevent language drift and overfitting on the 10-image dataset, a class-specific Prior-Preservation Loss (`λ = 1.0`, `N_class = 60` generated class samples) is minimized jointly:

    L(θ) = 𝔼_{x, c, ε, t} [ ‖ ε - ε_θ(z_t, t, c) ‖₂² ] 
         + λ · 𝔼_{x_pr, c_pr, ε', t'} [ ‖ ε' - ε_θ(z'_{t'}, t', c_pr) ‖₂² ]

---

## Methodology & Hyperparameter Configuration

All experiments use a deterministic global seed (`SEED = 123`), `512×512` bicubic center-cropped resolution, constant learning rate schedule (`warmup = 0`), and TensorBoard logging.

| Parameter | Textual Inversion (`TI_base`) | LoRA (`LORA_base_r8`) | DreamBooth (`DB_full_fast`) |
| :--- | :---: | :---: | :---: |
| **Trainable Target** | Token Embedding `<fruitface>` | UNet Cross-Attention (`r = 8`) | Full UNet Weights |
| **Identifier Token** | `<fruitface>` | `sksfruitface fruit` | `sksfruitface fruit` |
| **Max Training Steps** | `1200` | `1200` | `400` |
| **Learning Rate** | `5e-4` | `1e-4` | `2e-6` |
| **Precision / Optimizer** | `fp16` / AdamW | `bf16` / AdamW | `bf16` / 8-bit Adam (`bitsandbytes`) |
| **Prior Preservation** | No | No | Yes (`60` class images, `weight = 1.0`) |
| **Batch Size / Grad Accum** | `1 / 1` | `1 / 1` | `1 / 1` |

---

## Ablation Studies & Research Experiments

Beyond the baseline comparison, three controlled ablation studies were conducted using LoRA (`450` steps, `lr = 1e-4`, `seed = 123`):

1. **Experiment 1 — Dataset Cardinality (`n = 3` vs `n = 5` vs `n = 10`):**
   * *Finding:* `n = 3` suffers from severe compositional instability and background human artifacts; `n = 5` stabilizes geometry; `n = 10` achieves consistent surface integration and texture blending.
2. **Experiment 2 — LoRA Rank Capacity (`r = 4` vs `r = 8` vs `r = 16`):**
   * *Finding:* `r = 4` underfits complex surface geometry and requires heavy prompt engineering; `r = 16` enforces strong concept binding and color matching but increases overfitting risk on narrow prompts; `r = 8` provides the optimal bias-variance tradeoff.
3. **Experiment 3 — Text Encoder Fine-Tuning (`UNet-Only` vs `UNet + CLIP Text Encoder`):**
   * *Finding:* Jointly training LoRA adapters on the CLIP Text Encoder (`ON`) significantly sharpens fruit skin textures and token activation reliability compared to `UNet-Only` (`OFF`), at the cost of slightly reduced sample diversity across prompts.

---

## Getting Started

### Option 1: Automated Setup (Linux / macOS / Cloud GPU)

    git clone https://github.com/TPOJL/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    chmod +x run.sh
    ./run.sh

### Option 2: Manual Installation

1. **Clone the repository and create a virtual environment:**

       git clone https://github.com/TPOJL/YOUR_REPO_NAME.git
       cd YOUR_REPO_NAME
       python3 -m venv venv
       source venv/bin/activate

2. **Install pinned dependencies:**

       pip install -r requirements.txt

3. **Launch the training & evaluation notebook:**

       jupyter notebook

---

## Repository Structure

    ├── project-report.pdf   # Full technical report with qualitative grids & ablation results
    ├── notebook.ipynb       # End-to-end training & inference pipeline (TI, LoRA, DreamBooth)
    ├── requirements.txt     # Pinned dependencies (diffusers, peft, accelerate, bitsandbytes)
    ├── run.sh               # Automated environment setup and Jupyter launcher
    └── README.md            # Project documentation and experimental summary
