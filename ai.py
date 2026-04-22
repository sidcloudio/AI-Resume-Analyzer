"""
AI Resume Analyzer - Main Application
Run with: streamlit run app.py
"""

import PyPDF2
import streamlit as st # pyright: ignore[reportMissingImports]
import time
from PIL import Image
import pytesseract
from utils import (
    extract_text_from_pdf,
    extract_text_from_image, 
    preprocess_text,
    extract_skills,
    calculate_match_score,
    get_missing_skills,
    generate_suggestions,
    extract_job_keywords,
    highlight_missing_keywords,
)

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0a0d14;
    --surface:   #111520;
    --surface2:  #181d2e;
    --border:    #1e2540;
    --accent:    #4f8aff;
    --accent2:   #7c5cfc;
    --success:   #22d3a5;
    --warning:   #f59e0b;
    --danger:    #f43f5e;
    --text:      #e8eaf6;
    --muted:     #6b7280;
    --glow:      rgba(79,138,255,0.18);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.main { background: var(--bg); }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Header ── */
.hero-header {
    text-align: center;
    padding: 3.5rem 0 2rem;
    position: relative;
}
.hero-tag {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    background: linear-gradient(135deg, #e8eaf6 30%, #4f8aff 70%, #7c5cfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.8rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 0.5rem 0 2.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, var(--glow) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
}
.card:hover { border-color: var(--accent); }
.card:hover::before { opacity: 1; }

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}
.card-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
}

/* ── Score Colors ── */
.score-high   { color: var(--success); }
.score-medium { color: var(--warning); }
.score-low    { color: var(--danger);  }

/* ── Progress Bar ── */
.score-bar-wrap {
    background: var(--surface2);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 1rem 0 0.5rem;
    border: 1px solid var(--border);
}
.score-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1.4s cubic-bezier(.16,1,.3,1);
}

/* ── Skill Tags ── */
.tag-wrap { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.6rem; }
.tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    border: 1px solid;
    letter-spacing: 0.02em;
}
.tag-found {
    background: rgba(34,211,165,0.1);
    border-color: rgba(34,211,165,0.35);
    color: var(--success);
}
.tag-missing {
    background: rgba(244,63,94,0.1);
    border-color: rgba(244,63,94,0.35);
    color: var(--danger);
}
.tag-neutral {
    background: rgba(79,138,255,0.1);
    border-color: rgba(79,138,255,0.35);
    color: var(--accent);
}

/* ── Suggestion Items ── */
.suggestion-item {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.9rem 1rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.suggestion-item:hover { border-color: var(--accent2); }
.suggestion-icon {
    font-size: 1.2rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.suggestion-text {
    font-size: 0.88rem;
    line-height: 1.6;
    color: #c5cadb;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
}

/* ── Keyword Highlight ── */
.kw-missing {
    background: rgba(244,63,94,0.2);
    border: 1px solid rgba(244,63,94,0.4);
    color: #fda4af;
    border-radius: 4px;
    padding: 0.05em 0.3em;
    font-weight: 500;
}

/* ── Upload Zone ── */
.upload-hint {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* ── Stat Row ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.stat-box {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Streamlit Overrides ── */
div[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--glow) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.stAlert { border-radius: 10px !important; }

/* ── Spinner override ── */
div[data-testid="stSpinner"] > div { color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--muted);
    font-size: 0.78rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-tag">✦ Powered by NLP + AI</div>
    <h1 class="hero-title">AI Resume Analyzer</h1>
    <p class="hero-sub">
        Upload your resume and paste a job description to get an instant
        match score, skill gap analysis, and actionable improvements.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── Input Columns ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">📄 Resume Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Drop your resume here",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Supports PDF, JPG, JPEG, and PNG formats.",
        label_visibility="collapsed",
    )
    st.markdown('<div class="upload-hint">PDF, JPG, JPEG, or PNG format · Max 10 MB</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-label">💼 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        label="Paste job description",
        placeholder="Paste the full job description here — requirements, responsibilities, skills needed…",
        height=180,
        label_visibility="collapsed",
    )

# ─── Analyze Button ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
btn_col = st.columns([1, 2, 1])[1]
with btn_col:
    analyze_clicked = st.button("⚡  Analyze Resume", use_container_width=True)

# ─── Main Analysis Logic ──────────────────────────────────────────────────────
if analyze_clicked:

    # ── Validation ──
    if not uploaded_file:
        st.error("⚠️  Please upload a resume before analyzing.")
        st.stop()
    if not job_description.strip():
        st.error("⚠️  Please paste a job description before analyzing.")
        st.stop()

    # ── Processing ──
    resume_text = ""

    if uploaded_file is not None:   
        file_type = uploaded_file.type
        

    with st.spinner("Extracting text from resume..."):
        time.sleep(0.3)

        # ✅ IMAGE CASE
        if file_type.startswith("image/"):
            image = Image.open(uploaded_file)
            st.image(image)
            resume_text = pytesseract.image_to_string(image)

        # ✅ PDF CASE
        elif file_type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            resume_text = text

        else:
            st.error("Unsupported file type")
            st.stop()

    if not resume_text.strip():
        st.error("Could not extract text. Try clearer file.")
        st.stop()
    with st.spinner("Running NLP analysis…"):
        resume_clean   = preprocess_text(resume_text)
        jd_clean       = preprocess_text(job_description)
        resume_skills  = extract_skills(resume_text)
        jd_skills      = extract_skills(job_description)
        jd_keywords    = extract_job_keywords(job_description)
        match_score    = calculate_match_score(resume_clean, jd_clean, resume_skills, jd_skills)
        missing_skills = get_missing_skills(resume_skills, jd_skills)
        suggestions    = generate_suggestions(resume_text, resume_skills, missing_skills, match_score)
        time.sleep(0.4)

    # ── Determine Score Color ──
    if match_score >= 70:
        score_cls  = "score-high"
        bar_color  = "linear-gradient(90deg, #22d3a5, #4ade80)"
        verdict    = "Strong Match 🎯"
        verdict_color = "#22d3a5"
    elif match_score >= 40:
        score_cls  = "score-medium"
        bar_color  = "linear-gradient(90deg, #f59e0b, #fbbf24)"
        verdict    = "Moderate Match 📈"
        verdict_color = "#f59e0b"
    else:
        score_cls  = "score-low"
        bar_color  = "linear-gradient(90deg, #f43f5e, #fb7185)"
        verdict    = "Needs Improvement 🔧"
        verdict_color = "#f43f5e"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # RESULTS SECTION
    # ══════════════════════════════════════════════════════

    # ── Score + Stats Row ──
    score_col, stats_col = st.columns([1, 2], gap="large")

    with score_col:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Match Score</div>
            <div class="card-value {score_cls}">{match_score}%</div>
            <div class="score-bar-wrap">
                <div class="score-bar-fill"
                     style="width:{match_score}%; background:{bar_color};">
                </div>
            </div>
            <div style="font-size:0.85rem; color:{verdict_color}; font-weight:600; margin-top:0.4rem;">
                {verdict}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with stats_col:
        found_count   = len(resume_skills)
        missing_count = len(missing_skills)
        jd_count      = len(jd_skills)
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-num" style="color:var(--accent);">{found_count}</div>
                <div class="stat-lbl">Skills Found</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:var(--danger);">{missing_count}</div>
                <div class="stat-lbl">Skills Missing</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:var(--success);">{jd_count}</div>
                <div class="stat-lbl">JD Keywords</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:var(--warning);">{len(suggestions)}</div>
                <div class="stat-lbl">Suggestions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Skills Side-by-Side ──
    skills_col, missing_col = st.columns(2, gap="large")

    with skills_col:
        st.markdown('<div class="section-label">✅ Skills Found in Resume</div>', unsafe_allow_html=True)
        if resume_skills:
            tags_html = "".join(
                f'<span class="tag tag-found">✓ {s}</span>'
                for s in sorted(resume_skills)
            )
            st.markdown(f'<div class="tag-wrap">{tags_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--muted); font-size:0.85rem;">No recognizable skills detected.</p>', unsafe_allow_html=True)

    with missing_col:
        st.markdown('<div class="section-label">❌ Skills Missing (from JD)</div>', unsafe_allow_html=True)
        if missing_skills:
            tags_html = "".join(
                f'<span class="tag tag-missing">✕ {s}</span>'
                for s in sorted(missing_skills)
            )
            st.markdown(f'<div class="tag-wrap">{tags_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--success); font-size:0.85rem;">🎉 No missing skills — great coverage!</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Improvement Suggestions ──
    st.markdown('<div class="section-label">💡 Improvement Suggestions</div>', unsafe_allow_html=True)
    icons = ["🎯", "📊", "🔑", "🚀", "✍️", "🧩", "📌", "⚡"]
    for i, tip in enumerate(suggestions):
        icon = icons[i % len(icons)]
        st.markdown(f"""
        <div class="suggestion-item">
            <span class="suggestion-icon">{icon}</span>
            <span class="suggestion-text">{tip}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Keyword Highlighting Expander ──
    with st.expander("🔍  View Job Description with Missing Keywords Highlighted"):
        highlighted_jd = highlight_missing_keywords(job_description, missing_skills)
        st.markdown(
            f'<div style="font-size:0.88rem; line-height:1.9; color:#c5cadb;">{highlighted_jd}</div>',
            unsafe_allow_html=True,
        )

    # ── Raw Resume Text Expander ──
    with st.expander("📃  View Extracted Resume Text"):
        st.text_area(
            label="Extracted Text",
            value=resume_text[:3000] + ("…[truncated]" if len(resume_text) > 3000 else ""),
            height=250,
            disabled=True,
            label_visibility="collapsed",
        )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    AI Resume Analyzer &nbsp;·&nbsp; Built with Streamlit + spaCy + scikit-learn
    &nbsp;·&nbsp; For educational purposes
</div>
""", unsafe_allow_html=True)
