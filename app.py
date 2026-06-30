import streamlit as st
from pathlib import Path
import requests

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafScan · Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
css = Path("style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "upload"          # "upload" | "result"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – replace this block with your real model call
# ══════════════════════════════════════════════════════════════════════════════
def run_model(image_bytes: bytes) -> dict:
    resp = requests.post(
        "http://51.20.42.207:8000/predict",
        files={"file": ("image.jpg", image_bytes)}
    )
    data = resp.json()
    top  = data["prediction"][0]
    info = data["disease_info"]

    return {
        "disease":     top["class_name"],
        "confidence":  top["probability"] * 100,
        "severity":    info["severity"],
        "description": info["description"],
        "symptoms":    info["symptoms"],
        "treatment":   info["treatment"],
        "prevention":  info["prevention"],
    }

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def page_upload():
    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div class="hero-badge">🌿 AI-Powered Diagnostics</div>
      <h1 class="hero-title">LeafScan</h1>
      <p class="hero-sub">
        Upload a photograph of your plant and let our model identify
        diseases with precision — instantly.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Divider ───────────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Upload card ───────────────────────────────────────────────────────────
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Upload Plant Image</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(uploaded, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="file-meta">📎 {uploaded.name} · '
            f'{uploaded.size / 1024:.1f} KB</p>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)   # /upload-card

    # ── Predict button ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button(
            "✦  Analyse Plant",
            disabled=uploaded is None,
            use_container_width=True,
            key="predict_btn",
        )

    if predict_clicked and uploaded:
        with st.spinner(""):
            st.markdown(
                '<div class="scanning-msg">🔬 Scanning leaf patterns…</div>',
                unsafe_allow_html=True,
            )
            result = run_model(uploaded.getvalue())

        st.session_state.prediction = result
        st.session_state.uploaded_image = uploaded.getvalue()
        st.session_state.page = "result"
        st.rerun()

    # ── Footer tips ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="tips-row">
      <div class="tip-card">
        <span class="tip-icon">☀️</span>
        <span>Good lighting reveals lesion detail</span>
      </div>
      <div class="tip-card">
        <span class="tip-icon">🔍</span>
        <span>Focus on the most affected leaf</span>
      </div>
      <div class="tip-card">
        <span class="tip-icon">📐</span>
        <span>Fill the frame with the leaf</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def page_result():
    pred = st.session_state.prediction
    img  = st.session_state.uploaded_image

    severity_class = pred["severity"].lower()   # low | moderate | high

    # ── Top bar ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="result-topbar">
      <span class="topbar-brand">🌿 LeafScan</span>
      <span class="topbar-label">Diagnostic Report</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Disease headline card ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="disease-headline">
      <div class="disease-name">{pred["disease"]}</div>
      <div class="disease-meta-row">
        <div class="conf-pill">
          <span class="conf-label">Confidence</span>
          <span class="conf-value">{pred["confidence"]:.1f}%</span>
        </div>
        <div class="severity-pill {severity_class}">
          {pred["severity"]} Severity
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="conf-bar-wrap">
      <div class="conf-bar-fill" style="width:{pred['confidence']}%"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout: image + description ────────────────────────────────
    col_img, col_desc = st.columns([1, 1.6], gap="large")

    with col_img:
        if img:
            st.markdown('<div class="result-image-wrap">', unsafe_allow_html=True)
            st.image(img, use_container_width=True, caption="Analysed specimen")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_desc:
        st.markdown(f"""
        <div class="desc-block">
          <p class="section-label">About This Disease</p>
          <p class="desc-text">{pred["description"]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider-sm"></div>', unsafe_allow_html=True)

    # ── Three info panels ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="medium")

    panels = [
        ("🔬", "Symptoms",   pred["symptoms"]),
        ("💊", "Treatment",  pred["treatment"]),
        ("🛡️", "Prevention", pred["prevention"]),
    ]

    for col, (icon, title, items) in zip([c1, c2, c3], panels):
        items_html = "".join(f'<li>{i}</li>' for i in items)
        col.markdown(f"""
        <div class="info-panel">
          <div class="panel-header">
            <span class="panel-icon">{icon}</span>
            <span class="panel-title">{title}</span>
          </div>
          <ul class="panel-list">{items_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Actions ───────────────────────────────────────────────────────────────
    st.markdown('<div class="divider-sm"></div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 1.4, 1])
    with col_b:
        if st.button("← Analyse Another Plant", use_container_width=True, key="back_btn"):
            st.session_state.page = "upload"
            st.session_state.prediction = None
            st.session_state.uploaded_image = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "upload":
    page_upload()
else:
    page_result()
