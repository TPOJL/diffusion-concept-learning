# Concept Personalization in Stable Diffusion v1.5 (fruitface)

This repository contains an exploratory research project investigating how to introduce a novel, non-standard visual concept into a pretrained diffusion model (Stable Diffusion v1.5). 

* **Author:** Matsvei Tonki
* **Base Model:** Stable Diffusion v1.5

---

## 1. Project Overview & Concept
* **Concept Name:** `fruitface`
* **Target Concept:** A realistic fruit or berry where a single human face is physically embedded/carved into the skin, strictly preserving the fruit's texture, color, and geometry.
* **Key Challenge:** Preventing failure modes where the model generates a person holding a fruit or a human portrait in the background instead of an integrated surface feature.

---

## 2. Compared Personalization Techniques
The project evaluates and compares three standard personalization methods using a custom dataset of 10 images:
1. **Textual Inversion (TI):** Learns a new token embedding (`<fruitface>`). Lightweight and fast, but exhibits limited capacity and high prompt sensitivity for strict compositional constraints.
2. **LoRA (Low-Rank Adaptation):** Trains adapters for the token (`sksfruitface`). Delivers the best speed-quality tradeoff, stable training, and reliable concept injection.
3. **DreamBooth (Full):** Fine-tunes full model weights with prior preservation. Provides the strongest concept binding and most consistent geometry, but carries a higher risk of overfitting and reduced diversity.

---

## 3. Key Findings & Experiments
The repository includes notebooks and documentation covering three core experimental axes:
* **Dataset Size (n = 3 vs 5 vs 10):** Increasing instance images improved stability and drastically reduced spurious "person in background" artifacts.
* **LoRA Rank (r = 4 vs 16):** Higher rank (r = 16) increased concept strength and activation reliability, though overly narrow prompts can increase overfitting risks.
* **Text Encoder Training (ON vs OFF):** Enabling text encoder training made token activation significantly stronger and more consistent, with a minor trade-off in output diversity.

---

## 4. Repository Structure
* `*.ipynb` — Jupyter Notebooks containing the training pipelines, configurations, and evaluation loops.
* `project-report.pdf` (or `.docx`) — Comprehensive project report detailing methodology, hyperparameter settings, and qualitative/quantitative visual comparisons.
