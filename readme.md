# Text-to-Image Generation System

**Stable Diffusion with CLIP Conditioning**

Author: Neha Dharanu  

Teammate: Sri Lakshmi Swetha Jalluri

Date: December 2025

**🔗 GitHub Repository:** [https://github.com/nehadharanu/text-to-image-stable-diffusion](https://github.com/nehadharanu/text-to-image-stable-diffusion)

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

## 📦 Model Download

⚠️ **Important:** The fine-tuned model checkpoint (~4GB) is too large for GitHub.

**Download the model from Google Drive:**

🔗 [model](https://drive.google.com/drive/folders/1pYlPWib9pQIl416Ji0KpCTRCjsB4MycA?usp=drive_link)

**Installation Steps:**

1. Download the model folder from Google Drive
2. Extract it to your project directory as: `models/best_evaluated_model/`
3. Your folder structure should be:
   ```
   text-to-image-stable-diffusion/
   ├── models/
   │   └── best_evaluated_model/
   │       ├── text_encoder/
   │       ├── unet/
   │       ├── vae/
   │       ├── scheduler/
   │       └── ... (other model files)
   ├── app.py
   └── ... (other files)
   ```
4. Run: `streamlit run app.py`

---

## 🗂️ Project Structure

```text
text-to-image-stable-diffusion/
│
├── models/
│   └── best_evaluated_model/          # Download from Google Drive
│       ├── feature_extractor/
│       ├── scheduler/
│       ├── text_encoder/
│       ├── tokenizer/
│       ├── unet/
│       ├── vae/
│       ├── best_config.json
│       └── model_index.json
│
├── outputs/                            # Evaluation results
│   ├── comparison_proper_metrics.png
│   ├── metrics_analysis_proper.png
│   ├── parameter_sensitivity_analysis.png
│   ├── test_output.png
│   └── my_custom_generation.png
│
├── tests/
│   └── test_inference.py              # Standalone testing script
│
├── app.py                              # Streamlit web application
├── Generative_Project.ipynb            # Jupyter notebook with experiments
├── dataset.md                          # Dataset details
├── readme.md                           # This file
├── requirements.txt                    # Dependencies
└── Technical_documentation.pdf         # Complete documentation
```

**Key Files:**

- `models/`: Fine-tuned Stable Diffusion checkpoint (download separately)
- `outputs/`: Evaluation results and example generations
- `tests/`: Standalone inference validation script
- `app.py`: Streamlit web application
- `Generative_Project.ipynb`: Experiments and analysis notebook
- `dataset.md`: Dataset description and usage
- `Technical_documentation.pdf`: Full system documentation

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/nehadharanu/text-to-image-stable-diffusion.git
cd text-to-image-stable-diffusion
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Model
🔗 [model](https://drive.google.com/drive/folders/1pYlPWib9pQIl416Ji0KpCTRCjsB4MycA?usp=drive_link)
Download the fine-tuned model from the Google Drive link above and place it in `models/best_evaluated_model/`

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Testing

A standalone inference testing script is provided:

```bash
python tests/test_inference.py
```

This script:

- Loads the fine-tuned model
- Runs inference on CPU
- Saves an output image
- Verifies the pipeline works end-to-end

⏱️ **Expected runtime (CPU):** ~30–60 seconds per image

---

## 📊 Evaluation & Analysis

The project includes:

- **Scheduler comparison grids** - Visual comparison of DDIM, PNDM, and Euler schedulers
- **FID and Inception Score heatmaps** - Quantitative quality metrics across configurations
- **CFG scale sensitivity analysis** - Impact of guidance scale on output quality
- **Inference steps analysis** - Quality vs. speed trade-offs
- **Final production recommendations** - Optimal settings for deployment

All evaluation images are stored in the `outputs/` folder.

**Key Results:**

- **Best FID Score:** 150.9 (DDIM, CFG=10.0)
- **Best Scheduler:** Euler (recommended)
- **Optimal CFG Range:** 5.0-7.5 for balanced quality
- **Optimal Steps:** 30-40 (diminishing returns after)

---

## 🖥️ Web Demo

A Streamlit application provides an interactive demo:

```bash
streamlit run app.py
```

**Features:**

- Prompt input with validation
- Example prompt gallery (Animals, Vehicles, Food, Nature)
- Real-time image generation with progress indicator
- Downloadable output images
- Technical details and documentation tabs

---

## 📄 Documentation

Complete technical documentation is available in:

1. **Technical_documentation.pdf** - Comprehensive system documentation including:

   - System architecture diagram
   - Implementation details
   - Performance metrics (FID, Inception Score, timing)
   - Challenges and solutions
   - Future improvements
   - Ethical considerations

2. **dataset.md** - Dataset details and preparation

3. **Inline code documentation** - Comments throughout the codebase

4. **Jupyter Notebook** - [Generative_Project.ipynb](https://github.com/nehadharanu/text-to-image-stable-diffusion/blob/main/Generative_Project.ipynb) - Full experiments and analysis

---

## ⚖️ Ethical Considerations

- Uses publicly available pretrained models (Stable Diffusion, CLIP)
- No personal data collected
- Potential misuse documented in technical report
- Users encouraged to use responsible prompts
- Academic and educational use only

---

## 🔧 System Requirements

**Minimum:**

- Python 3.8+
- 8 GB RAM
- CPU only (GPU optional)
- 10 GB disk space

**Recommended:**

- Python 3.10
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM (for faster generation)
- 20 GB disk space

---

## 📌 Notes

- This project is **CPU-compatible** by design for maximum portability
- GPU generation: ~8-10 seconds per image
- CPU generation: ~30-60 seconds per image
- All experiments are fully reproducible with fixed seeds

---

## 🔗 Resources

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [PyTorch](https://pytorch.org/)
- [Stable Diffusion](https://stability.ai/stable-diffusion)
- [Streamlit](https://streamlit.io/)

---

## 📧 Contact

**Author:** Neha Dharanu  
**Repository:** [https://github.com/nehadharanu/text-to-image-stable-diffusion](https://github.com/nehadharanu/text-to-image-stable-diffusion)

---

## 📜 License

This project is for academic and educational purposes. Please respect the licenses of the underlying models and libraries used.

---

© 2025 Neha Dharanu
