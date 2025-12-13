# Text-to-Image Generation System  
**Stable Diffusion with CLIP Conditioning**

Author: Neha Dharanu  
Date: December 2025  

---

## 📌 Overview
This project implements a **text-to-image generation system** using a Stable Diffusion pipeline conditioned on CLIP text embeddings. The system converts natural language prompts into high-quality 512×512 images and includes **evaluation, parameter sensitivity analysis, and reproducible testing**.

The project is designed to run **entirely on CPU**, making it portable and reproducible without requiring GPU access.

---

## 🚀 Key Features
- Text-to-image generation using Stable Diffusion
- CLIP-based text conditioning
- Fine-tuned local checkpoint loading
- CPU-only execution
- Parameter sensitivity analysis (CFG scale, inference steps, schedulers)
- Quantitative evaluation using **FID** and **Inception Score**
- Negative prompt analysis
- Clean Streamlit-based web demo
- Reproducible inference testing script

---

## 🧠 Model Architecture
- **Text Encoder:** CLIP
- **Diffusion Core:** UNet + Scheduler
- **Latent Decoder:** VAE
- **Schedulers Evaluated:** Euler, DDIM, PNDM

---

## 🗂️ Project Structure
See the *Project Structure* section below for full details.

---

# 2️⃣ PROJECT STRUCTURE (PASTE INTO TECHNICAL DOCUMENT)

Use this **exact block** under a section titled:

### **Project Structure**

```text
FINALPROJECT/
│
├── models/
│   └── best_evaluated_model/
│       ├── feature_extractor/
│       ├── scheduler/
│       ├── text_encoder/
│       ├── tokenizer/
│       ├── unet/
│       ├── vae/
│       ├── best_config.json
│       └── model_index.json
│
├── outputs/
│   ├── comparison_proper_metrics.png
│   ├── metrics_analysis_proper.png
│   ├── parameter_sensitivity_analysis.png
│   └── my_custom_generation.png
│
├── tests/
│   └── test_inference.py
│
├── app.py
├── Generative_Project.ipynb
├── dataset.md
├── readme.md
├── requirements.txt
└── Technical_documentation.pdf

Explanation

models/: Fine-tuned Stable Diffusion checkpoint

outputs/: Evaluation results and example generations

tests/: Standalone inference validation script

app.py: Streamlit web application

Generative_Project.ipynb: Experiments and analysis notebook

dataset.md: Dataset description and usage

Technical_documentation.pdf: Full system documentation
---

## 🧪 Testing
A standalone inference testing script is provided:

```bash
python tests/test_inference.py
This script:

Loads the fine-tuned model

Runs inference on CPU

Saves an output image

Verifies the pipeline works end-to-end

⏱️ Expected runtime (CPU):

~30–60 seconds per image

📊 Evaluation & Analysis

The project includes:

Scheduler comparison grids

FID and Inception Score heatmaps

CFG scale sensitivity analysis

Inference steps vs quality/speed trade-off

Final production recommendations

All evaluation images are stored in the outputs/ folder.

🖥️ Web Demo

A Streamlit application is included:

streamlit run app.py


Features:

Prompt input

Example prompt gallery

Image generation with loading spinner

Downloadable output image

📦 Setup Instructions (CPU Only)
pip install -r requirements.txt


Then run:

streamlit run app.py

⚖️ Ethical Considerations

Uses publicly available pretrained models

No personal data collected

Potential misuse documented in technical report

Users encouraged to use responsible prompts

📄 Documentation

Technical_documentation.pdf

dataset.md

Inline code documentation

📌 Notes

This project is CPU-only by design

Results may take longer compared to GPU execution

All experiments are reproducible

🔗 Resources

Hugging Face Diffusers

OpenAI CLIP

PyTorch

© 2025 Neha Dharanu