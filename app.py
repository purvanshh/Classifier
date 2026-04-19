import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "EfficientNet B0.pth"
NUM_CLASSES = 4
IMG_SIZE    = 224
CLASS_NAMES = ["2390", "Grand Naine", "Ney Poovan", "Poovan"]
CLASS_DESC  = {
    "2390":       "A robust variety known for its thick peel and starchy flesh, commonly used in cooking.",
    "Grand Naine": "The world's most common export banana — mild, sweet, and slightly creamy. Also called Chiquita.",
    "Ney Poovan":  "A South Indian hybrid with a honey-like sweetness and firm texture. Great eaten fresh.",
    "Poovan":      "A popular South Indian variety with a tangy-sweet flavour and thin, easy-to-peel skin.",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ────────────────────────────────────────────────────────────────────
def build_model(num_classes=4, dropout_rate=0.2):
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model

@st.cache_resource
def load_model():
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BananaScope",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0a;
    color: #f0f0f0;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem; max-width: 700px; margin: 0 auto; }

/* ── nav ── */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0 2.5rem;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 3rem;
}
.nav-logo { font-size: 1rem; font-weight: 600; letter-spacing: -0.02em; display: flex; align-items: center; gap: 6px; }
.nav-logo span { color: #b5f23d; }
.nav-badge {
    background: #b5f23d;
    color: #0a0a0a;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 999px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── hero ── */
.hero { text-align: center; margin-bottom: 3rem; }
.hero h1 {
    font-size: clamp(2rem, 6vw, 3rem);
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem;
}
.hero h1 em { color: #b5f23d; font-style: normal; }
.hero p { color: #666; font-size: 0.9rem; max-width: 380px; margin: 0 auto; line-height: 1.6; }

/* ── upload card ── */
.upload-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 16px;
    padding: 2.5rem 2rem 1.5rem;
    text-align: center;
    margin-bottom: 0;
}
.upload-icon {
    width: 48px; height: 48px;
    background: #1a1a1a;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1.2rem;
}
.upload-card h3 { font-size: 1rem; font-weight: 600; margin: 0 0 0.3rem; }
.upload-card p  { color: #555; font-size: 0.8rem; margin: 0; }

/* file uploader sits inside the card visually */
[data-testid="stFileUploader"] {
    background: #111;
    border: 1px solid #1e1e1e;
    border-top: none;
    border-radius: 0 0 16px 16px;
    padding: 0 2rem 1.8rem;
    margin-bottom: 1.5rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: #0a0a0a !important;
    border: 1.5px dashed #2a2a2a !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child,
[data-testid="stFileUploaderDropzoneInstructions"] > div > small { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    content: "Browse files";
    font-size: 0.82rem;
    color: #555;
}
.stButton > button {
    background: #b5f23d !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 0.55rem 1.6rem !important;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── result card ── */
.result-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.result-inner {
    display: flex;
    gap: 0;
}
.result-img-wrap {
    position: relative;
    width: 220px;
    min-width: 220px;
    background: #0d0d0d;
}
.result-img-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.analyzed-badge {
    position: absolute;
    top: 10px; left: 10px;
    background: rgba(181,242,61,0.15);
    border: 1px solid #b5f23d;
    color: #b5f23d;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.result-info {
    padding: 1.5rem;
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.match-label { font-size: 0.65rem; color: #555; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem; }
.match-name  { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.03em; margin: 0 0 0.6rem; }
.match-desc  { font-size: 0.78rem; color: #777; line-height: 1.6; margin-bottom: 1.2rem; }
.conf-label  { font-size: 0.7rem; color: #555; margin-bottom: 0.4rem; display: flex; justify-content: space-between; }
.conf-label span { color: #b5f23d; font-weight: 700; font-size: 0.95rem; }
.conf-bar-bg { background: #1e1e1e; border-radius: 999px; height: 5px; overflow: hidden; }
.conf-bar-fill { background: #b5f23d; height: 100%; border-radius: 999px; transition: width .6s ease; }

/* ── all probs ── */
.probs-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
}
.probs-title { font-size: 0.7rem; color: #555; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem; }
.prob-row { margin-bottom: 0.75rem; }
.prob-row-label { display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.3rem; }
.prob-row-label .pct { color: #888; }

/* ── about accordion ── */
.about-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 16px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 3rem;
}
.about-title { font-size: 0.7rem; color: #555; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.8rem; }
.about-body  { font-size: 0.8rem; color: #666; line-height: 1.7; }

/* ── footer ── */
.footer {
    border-top: 1px solid #1a1a1a;
    padding-top: 1.5rem;
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #444;
}

/* responsive */
@media (max-width: 520px) {
    .result-inner { flex-direction: column; }
    .result-img-wrap { width: 100%; min-width: unset; height: 200px; }
    .nav-links { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ── Nav ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-logo"><span>Banana</span>Scope</div>
  <div class="nav-badge">Classifier</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Identify banana varieties<br><em>instantly</em></h1>
  <p>BananaScope uses computer vision to classify South Indian banana varieties with pinpoint accuracy.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-card">
  <div class="upload-icon">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#aaa" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="16 16 12 12 8 16"></polyline>
      <line x1="12" y1="12" x2="12" y2="21"></line>
      <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path>
    </svg>
  </div>
  <h3>Upload a banana image to get started</h3>
  <p>Supported formats: JPG, PNG. High resolution preferred.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="hidden")

# ── Inference ────────────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    model     = load_model()
    transform = get_transform()

    tensor  = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)[0]

    conf_val, pred_idx = torch.max(probs, 0)
    pred_name = CLASS_NAMES[pred_idx.item()]
    conf_pct  = conf_val.item() * 100

    # encode image for HTML
    import base64, io
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    bar_w = f"{conf_pct:.1f}%"

    st.markdown(f"""
    <div class="result-card">
      <div class="result-inner">
        <div class="result-img-wrap">
          <div class="analyzed-badge">✓ Analyzed</div>
          <img src="data:image/jpeg;base64,{b64}" alt="uploaded banana"/>
        </div>
        <div class="result-info">
          <div>
            <div class="match-label">Match found</div>
            <div class="match-name">{pred_name}</div>
            <div class="match-desc">{CLASS_DESC.get(pred_name, "")}</div>
          </div>
          <div>
            <div class="conf-label">AI Confidence Level <span>{conf_pct:.0f}%</span></div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar_w}"></div></div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # All probabilities
    st.markdown('<div class="probs-card"><div class="probs-title">All probabilities</div>', unsafe_allow_html=True)
    for i, name in enumerate(CLASS_NAMES):
        p = probs[i].item() * 100
        st.markdown(f"""
        <div class="prob-row">
          <div class="prob-row-label"><span>{name}</span><span class="pct">{p:.1f}%</span></div>
          <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{p:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("↺  Try another image"):
        st.rerun()

# ── About ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="about-card">
  <div class="about-title">About this AI model</div>
  <div class="about-body">
    BananaScope is powered by EfficientNet-B0, a lightweight convolutional neural network fine-tuned on
    a curated dataset of South Indian banana varieties — 2390, Grand Naine, Ney Poovan, and Poovan.
    The model achieves high accuracy while running efficiently on standard hardware.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>© 2026 BananaScope</span>
  <span>EfficientNet-B0 · 4 classes</span>
</div>
""", unsafe_allow_html=True)
