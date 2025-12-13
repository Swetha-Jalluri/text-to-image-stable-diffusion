import os
from datetime import datetime

import streamlit as st
import torch
from diffusers import DiffusionPipeline


# ----------------------------
# EDIT THESE LINKS
# ----------------------------
GITHUB_REPO_URL = "https://github.com/yourusername/your-repo"
NOTEBOOK_URL = "https://github.com/yourusername/your-repo/blob/main/Generative_Project.ipynb"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "best_evaluated_model")
DOC_PDF_PATH = os.path.join(BASE_DIR, "Technical_documentation.pdf")

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512


# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(
    page_title="Text-to-Image Generation",
    page_icon="🎨",
    layout="wide",
)


# ----------------------------
# CSS: Beige + production look
# ----------------------------
def inject_css():
    st.markdown(
        """
        <style>
          /* Remove Streamlit header bar */
          header[data-testid="stHeader"] { display: none; }
          div[data-testid="stToolbar"] { display: none; }
          div[data-testid="stDecoration"] { display: none; }

          .block-container { padding-top: 1rem !important; padding-bottom: 2.5rem; max-width: 1200px; }

          html, body, [class*="css"] {
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
          }

          .stApp {
            background: radial-gradient(1200px 800px at 10% 0%, #FFF7E8 0%, #F6F0E4 45%, #F4EEE1 100%);
            color: #1F2937;
          }

          a { color: #7C5C2A !important; text-decoration: none; }
          a:hover { text-decoration: underline; }

          .panel {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(31, 41, 55, 0.08);
            box-shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
            border-radius: 18px;
            padding: 18px 18px;
          }

          .hero-title {
            font-size: 34px;
            font-weight: 850;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
            color: #1F2937;
          }
          .hero-subtitle {
            font-size: 14px;
            color: rgba(31, 41, 55, 0.75);
            line-height: 1.5;
            margin-bottom: 14px;
          }
          .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
          .pill {
            font-size: 12px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(31, 41, 55, 0.10);
            background: rgba(255, 255, 255, 0.65);
            color: rgba(31, 41, 55, 0.82);
          }

          .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
          @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
          @media (max-width: 650px) { .grid { grid-template-columns: repeat(1, minmax(0, 1fr)); } }

          .card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(31, 41, 55, 0.08);
            border-radius: 18px;
            padding: 14px 14px;
            box-shadow: 0 10px 25px rgba(17, 24, 39, 0.05);
            min-height: 140px;
          }
          .card h4 { margin: 0; font-size: 15px; font-weight: 800; color: #1F2937; }
          .card p { margin: 8px 0 0 0; font-size: 13px; color: rgba(31, 41, 55, 0.74); line-height: 1.45; }

          .stButton button, .stDownloadButton button, .stLinkButton a {
            border-radius: 12px !important;
            border: 1px solid rgba(31, 41, 55, 0.12) !important;
            background: #FFFFFF !important;
            color: #1F2937 !important;
            box-shadow: 0 8px 20px rgba(17, 24, 39, 0.06) !important;
          }
          .stButton button:hover, .stDownloadButton button:hover, .stLinkButton a:hover {
            border: 1px solid rgba(124, 92, 42, 0.35) !important;
          }

          div[data-testid="stButton"] button[kind="primary"] {
            background: #7C5C2A !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(124, 92, 42, 0.2) !important;
          }

          /* Prompt highlight label */
          .prompt-label {
            font-size: 14px;
            font-weight: 900;
            color: #7C5C2A;
            margin: 4px 0 8px 0;
          }

          /* Text area styling */
          textarea {
            border-radius: 16px !important;
            border: 2px solid rgba(124, 92, 42, 0.40) !important;
            background: rgba(255, 255, 255, 0.92) !important;
            color: #000000 !important;
            font-weight: 500 !important;
          }
          textarea:focus {
            outline: none !important;
            border: 2px solid rgba(124, 92, 42, 0.75) !important;
            box-shadow: 0 0 0 5px rgba(124, 92, 42, 0.12) !important;
          }
          textarea::placeholder {
            color: #6B7280 !important;
            opacity: 1 !important;
          }

          /* Orange alert (instead of default yellow warning) */
          .alert-orange{
            background: #FFF3E6;
            border: 1px solid #F59E0B;
            color: #7C2D12;
            padding: 12px 14px;
            border-radius: 14px;
            font-weight: 700;
            margin-top: 10px;
          }

          /* Sidebar */
          section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.55);
            border-right: 1px solid rgba(31, 41, 55, 0.06);
          }

          /* Tabs contrast */
          div[data-testid="stTabs"] button {
            color: rgba(31, 41, 55, 0.85) !important;
            font-weight: 850 !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
          }
          div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #1F2937 !important;
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(124, 92, 42, 0.30) !important;
            box-shadow: 0 10px 22px rgba(17, 24, 39, 0.06) !important;
          }
          div[data-testid="stTabs"] div[role="tablist"] {
            gap: 8px !important;
            border-bottom: 1px solid rgba(31, 41, 55, 0.10) !important;
            padding-bottom: 10px !important;
          }

          .footer {
            margin-top: 22px;
            text-align: center;
            color: rgba(31, 41, 55, 0.55);
            font-size: 12px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    local_ok = os.path.exists(os.path.join(MODEL_DIR, "model_index.json"))
    model_source = MODEL_DIR if local_ok else "CompVis/stable-diffusion-v1-4"

    try:
        # Use DiffusionPipeline.from_pretrained instead
        pipe = DiffusionPipeline.from_pretrained(
            model_source,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=local_ok,
        )
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        return pipe, device, model_source, None
    except Exception as e:
        return None, device, None, str(e)


# ----------------------------
# UI SECTIONS
# ----------------------------
def hero_section(model_source: str, device: str):
    is_local = "models" in str(model_source)
    st.markdown(
        f"""
        <div class="panel">
          <div class="hero-title">Text-to-Image Generation</div>
          <div class="hero-subtitle">
            Transform natural language into high-quality images using Stable Diffusion v1.4 with CLIP text encoding.
            Configured for optimal quality (Euler scheduler, CFG=7.5).
          </div>
          <div class="pill-row">
            <span class="pill">{'✓ Fine-Tuned Model' if is_local else '✓ Base Model (SD v1.4)'}</span>
            <span class="pill">✓ CLIP Text Encoder</span>
            <span class="pill">✓ Euler Scheduler</span>
            <span class="pill">Device: {device.upper()}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def links_row():
    st.markdown(
        f"""
        <div class="grid">
          <div class="card">
            <h4>GitHub Repository</h4>
            <p>Complete source code, notebook, and documentation</p>
          </div>
          <div class="card">
            <h4>Jupyter Notebook</h4>
            <p>Model training, optimization, and evaluation</p>
          </div>
          <div class="card">
            <h4>Technical Report</h4>
            <p>Architecture, metrics, challenges, and ethics</p>
          </div>
          <div class="card">
            <h4>Demo App</h4>
            <p>Interactive text-to-image generation interface</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def key_features_4_cards():
    st.markdown("### Key Features")
    st.markdown(
        """
        <div class="grid">
          <div class="card">
            <h4>Multimodal Integration</h4>
            <p>Combines CLIP text encoding with Stable Diffusion image synthesis for semantic understanding.</p>
          </div>
          <div class="card">
            <h4>Optimized Configuration</h4>
            <p>Tested 9 configurations. Best: Euler scheduler + CFG=10.0 (FID=153.86).</p>
          </div>
          <div class="card">
            <h4>Quality Metrics</h4>
            <p>Professional evaluation using FID and Inception Score. 8% improvement over baseline.</p>
          </div>
          <div class="card">
            <h4>Production Ready</h4>
            <p>Memory optimized (6.8 GB VRAM), fast generation (8.7s on GPU), responsive UI.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _set_prompt(text: str):
    st.session_state["prompt_input"] = text


def prompt_gallery():
    st.markdown("#### Example Prompts")
    st.caption("Click a button to load a prompt. You can edit it before generating.")

    examples = [
        ("Animals", "A majestic lion in the savanna, highly detailed, cinematic lighting, sharp focus"),
        ("Vehicles", "A vintage car on a coastal road at golden hour, ultra realistic, depth of field"),
        ("Food", "A steaming cup of coffee on a wooden table, soft morning light, realistic photography"),
        ("Nature", "A colorful parrot on a tree branch, highly detailed feathers, natural lighting"),
    ]

    cols = st.columns(4)
    for i, (label, text) in enumerate(examples):
        with cols[i]:
            st.markdown(
                f"""
                <div class="card">
                  <h4 style="font-size:14px;">{label}</h4>
                  <p style="font-size:12.5px; margin-top:8px;">{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                f"Load {label}",
                use_container_width=True,
                key=f"load_{label}",
                on_click=_set_prompt,
                args=(text,),
            )


def demo_section(pipe, device: str):
    if "prompt_input" not in st.session_state:
        st.session_state["prompt_input"] = ""

    # Prompt label
    st.markdown('<div class="prompt-label">Prompt</div>', unsafe_allow_html=True)

    # Prompt + Generate button on same row
    pcol, bcol = st.columns([4.2, 1])

    with pcol:
        st.text_area(
            "Prompt input",
            key="prompt_input",
            placeholder="Example: A majestic lion in the savanna, highly detailed, cinematic lighting",
            height=110,
            label_visibility="collapsed",
        )

    with bcol:
        st.write("")
        st.write("")
        generate = st.button("Generate Image", type="primary", use_container_width=True)

    # Status slot ABOVE Example Prompts
    status_slot = st.empty()

    # Example prompts (Generate button is above this)
    prompt_gallery()

    if generate:
        prompt = st.session_state.get("prompt_input", "").strip()
        if not prompt:
            status_slot.markdown(
                '<div class="alert-orange">Please enter a prompt before generating.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        try:
            # Spinner placed near prompt area (above examples)
            with status_slot.container():
                st.markdown(
                    '<div class="panel" style="text-align:center; font-weight:900;">Generating image…</div>',
                    unsafe_allow_html=True,
                )
                with st.spinner("Please wait..."):
                    start = datetime.now()
                    with torch.inference_mode():
                        out = pipe(
                            prompt=prompt,
                            guidance_scale=7.5,
                            num_inference_steps=50,
                            height=DEFAULT_HEIGHT,
                            width=DEFAULT_WIDTH,
                        )
                    img = out.images[0]
                    elapsed = (datetime.now() - start).total_seconds()

            status_slot.empty()

            st.success(f"Generated in {elapsed:.2f} seconds.")

            out_left, out_right = st.columns([2, 1])
            with out_left:
                st.image(img, caption=prompt, width=650)

            with out_right:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("**Download**")
                fname = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                tmp_path = os.path.join(BASE_DIR, fname)
                img.save(tmp_path)

                with open(tmp_path, "rb") as f:
                    st.download_button(
                        "Download PNG",
                        data=f,
                        file_name=fname,
                        mime="image/png",
                        use_container_width=True,
                    )
                st.caption("Default: CFG=7.5 • Steps=50 • 512×512")
                st.markdown("</div>", unsafe_allow_html=True)

                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            status_slot.empty()
            st.markdown(
                f'<div class="alert-orange">Generation failed: {e}</div>',
                unsafe_allow_html=True,
            )


def how_it_works():
    st.markdown("### How It Works")
    st.markdown(
        """
        <div class="panel">
          <b>Pipeline overview</b><br><br>
          1) Prompt is tokenized and encoded by <b>CLIP</b> into text embeddings.<br>
          2) Stable Diffusion starts from random latent noise (compressed 64×64×4 space).<br>
          3) A <b>UNet</b> iteratively denoises the latent over multiple steps using a scheduler.<br>
          4) <b>Cross-attention</b> injects text embeddings so the image matches the prompt.<br>
          5) The final latent is decoded by the <b>VAE decoder</b> into a 512×512 RGB image.<br><br>
          Latent-space diffusion improves efficiency while preserving high-quality results.
        </div>
        """,
        unsafe_allow_html=True,
    )


def documentation():
    st.markdown("### Documentation")
    if not os.path.exists(DOC_PDF_PATH):
        st.warning("Technical_documentation.pdf not found in the project root. Place it next to app.py.")
        return

    st.markdown(
        """
        <div class="panel">
          <b>Technical Report</b><br>
          Includes architecture diagram, implementation details, performance metrics, challenges, future work, and ethical considerations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with open(DOC_PDF_PATH, "rb") as f:
        st.download_button(
            "Download Technical Report PDF",
            data=f,
            file_name="Technical_documentation.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.markdown("## Project")
    st.markdown("**Neha Dharanu**")
    st.caption("Text-to-Image Generation")
    st.caption("Stable Diffusion + CLIP")

    st.markdown("---")
    st.markdown("### Links")
    st.link_button("GitHub Repo", GITHUB_REPO_URL, use_container_width=True)
    st.link_button("Notebook", NOTEBOOK_URL, use_container_width=True)

    if os.path.exists(DOC_PDF_PATH):
        with open(DOC_PDF_PATH, "rb") as f:
            st.download_button(
                "Download PDF",
                data=f,
                file_name="Technical_documentation.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ----------------------------
# MAIN
# ----------------------------
pipe, device, model_source, err = load_pipeline()

hero_section(model_source=str(model_source), device=device)
st.write("")
links_row()
st.write("")

if err:
    st.error(f"Model load error: {err}")
    st.stop()

tabs = st.tabs(["Demo", "Key Features", "How It Works", "Documentation"])

with tabs[0]:
    demo_section(pipe, device=device)

with tabs[1]:
    key_features_4_cards()

with tabs[2]:
    how_it_works()

with tabs[3]:
    documentation()

st.markdown(
    """
    <div class="footer">
      Built with Streamlit • Stable Diffusion • CLIP-conditioned latent diffusion • Neha Dharanu
    </div>
    """,
    unsafe_allow_html=True,
)