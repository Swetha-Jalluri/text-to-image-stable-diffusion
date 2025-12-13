# Dataset and Fine-Tuning Details

## Dataset Overview

This project uses the **COCO (Common Objects in Context)** dataset as the primary data source for fine-tuning experiments related to text-to-image generation.

Due to computational and time constraints typical of an academic project setting, **a filtered subset of the COCO dataset was used**, rather than the complete dataset.

This approach is standard practice in generative AI coursework and enables focused experimentation while maintaining reproducibility and model stability.

---

## Dataset Source

- **Name:** COCO (Common Objects in Context)
- **Official Website:** https://cocodataset.org
- **Task Used:** Image–caption pairs
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0)

---

## Subset Selection Strategy

Instead of using the full COCO dataset, a **domain-specific subset** was constructed.

The subset was filtered to emphasize the following categories:

- Animals
- Vehicles
- Food-related objects

This filtering strategy was chosen to:

- Reduce training time and computational cost
- Improve semantic alignment in frequently tested prompt domains
- Enable faster experimentation and evaluation cycles

---

## Data Preparation Process

The following preprocessing steps were applied before fine-tuning:

1. Caption normalization and cleaning
2. Removal of incomplete or corrupted image–caption pairs
3. Resolution normalization compatible with Stable Diffusion training
4. Validation of text–image alignment

Each training sample consists of:

- One image
- One corresponding natural language caption

---

## Fine-Tuning Configuration

- **Base Model:** Stable Diffusion v1.4
- **Fine-Tuned Component:** UNet denoising network
- **Text Encoder:** CLIP ViT-B/32 (kept frozen)
- **Optimizer:** AdamW
- **Learning Rate:** 1e-5
- **Loss Function:** Mean Squared Error (noise prediction objective)

Fine-tuning was conducted **offline** using the filtered dataset subset.  
The deployed Streamlit application loads the base or fine-tuned checkpoint depending on the runtime configuration.

---

## Evaluation Methodology

Model performance was evaluated using:

- **Fréchet Inception Distance (FID)**
- **Inception Score (IS)**
- Qualitative visual inspection of generated outputs

Evaluation results and analysis are documented in the project’s technical report.

---

## Dataset Availability

The full COCO dataset is not included in this repository due to its large size.

A small representative sample of the caption data used in this project is included here:

data/samples/coco_captions_subset.json

This sample demonstrates the exact JSON structure used for fine-tuning and evaluation workflows.

## Ethical and Legal Considerations

- The dataset is used strictly for academic purposes
- COCO licensing terms are respected
- No personally identifiable information (PII) is included
- Dataset bias and representational limitations are acknowledged and discussed in the documentation
