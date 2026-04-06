# PEdit: Pareto-Guided Image Editing via Dynamic Latent Trajectory Control

> CVPR 2026  
> Official implementation of **PEdit**

---

## 🔥 Overview

Text instruction-based image editing must balance two conflicting objectives:

- **Semantic Alignment** (follow the text instruction)
- **Structural Preservation** (keep the source image)

However, existing methods often fail to maintain this balance, leading to:
- over-editing (text-dominant)
- structure collapse (image-dominant)

👉 We propose **PEdit**, a diffusion-based image editing method that formulates editing as a **multi-objective optimization problem** and dynamically controls the latent trajectory toward the **Pareto-optimal region**.

---

## 🧠 Key Contributions

- **Pareto-guided editing framework**  
  → explicitly balances semantic fidelity and structural consistency

- **Dynamic condition scaling**  
  → adaptively adjusts text/image embeddings per Transformer block

- **Latent trajectory control**  
  → maintains balance throughout the denoising process

- **General plug-and-play method**  
  → applicable to DiT-based editing models (e.g., Kontext, QwenEdit)

---

## ⚙️ Method

### 🧩 Problem Formulation

We formulate image editing as:

- Objective 1: Semantic alignment (Text guidance)
- Objective 2: Structure preservation (Source image)

These objectives are **conflicting**, so we optimize along the **Pareto front** instead of a single objective.

---

### 🧪 Key Metrics

We introduce two key indicators:

- **TCR (Text Cross-Attention Ratio)**  
  → measures how strongly the model follows text

- **SSNR (Structural Signal-to-Noise Ratio)**  
  → measures how well source structure is preserved

---

### 🚀 Pipeline

PEdit consists of two stages:

#### Stage 1: Pareto Initialization
- Optimize condition scaling in early timesteps
- Find a balanced latent near the Pareto front

#### Stage 2: Editing Pathway Control
- Maintain balance during denoising
- Prevent collapse toward either objective

---

## 🖼️ Results

### ✨ Qualitative Results

PEdit achieves:

- Precise region-wise editing
- Strong semantic alignment
- Robust structure preservation

Compared to prior methods:
- avoids over-editing (ReFlex)
- avoids structure collapse (Kontext, QwenEdit)
- handles both local and global edits effectively

---

### 📊 Quantitative Results

PEdit consistently improves:

- **CLIP-T ↑** (semantic alignment)
- **CLIP-I / DINO-I ↑** (structure preservation)
- **L2 distance ↓** (better reconstruction)
- **FID ↓** (better perceptual quality)

👉 Achieves strong performance across both:
- Local editing
- Global editing

---

### 👤 User Study

- Highest preference in:
  - Edit Fidelity
  - Visual Quality
- Competitive in:
  - Structure Consistency

---

### 🔁 Multi-Edit Performance

- Existing text-only methods fail to control multiple regions
- **PEdit enables precise region-wise multi-editing**

---

## 📊 Benchmarks

We evaluate on:

- **HQ-Edit** (synthetic dataset)
- **Emu-Edit Bench** (real-world dataset)

---

## ⚡ Advantages

- No inversion required (faster than inversion-based methods)
- No retraining per step
- Stable across different editing tasks
- Works with existing DiT pipelines

---

## 🚀 Installation

```bash
git clone https://github.com/yourname/pedit
cd pedit
pip install -r requirements.txt
