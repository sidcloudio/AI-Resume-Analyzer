# 🧠 AI Resume Analyzer

A production-quality web app that analyzes your resume against a job description using NLP — giving you a match score, skill gap analysis, and actionable improvement tips.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Upload | Extracts text from any PDF resume |
| 🎯 Match Score | Hybrid TF-IDF cosine similarity + skill overlap |
| ✅ Skill Detection | 150+ skills across tech, data, cloud, soft skills |
| ❌ Gap Analysis | Shows exactly which skills you're missing |
| 💡 Suggestions | Contextual, content-aware improvement tips |
| 🔍 Keyword Highlight | Highlights missing keywords in the JD |
| 🎨 Modern UI | Dark-themed, animated, mobile-friendly Streamlit app |

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/yourname/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
ai_resume_analyzer/
│
├── app.py              # Main Streamlit application + UI
├── utils.py            # NLP helpers (extraction, scoring, suggestions)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🧠 How It Works

```
PDF Upload
    │
    ▼
PyPDF2 Text Extraction
    │
    ▼
spaCy NLP Preprocessing
(tokenize → lowercase → remove stopwords)
    │
    ├──► Skill Extraction (150+ skill DB + NER)
    │
    ▼
TF-IDF Vectorization (scikit-learn)
    │
    ▼
Cosine Similarity + Skill Overlap  →  Match Score (0–100%)
    │
    ▼
Gap Analysis + Contextual Suggestions → Displayed in UI
```

### Match Score Formula

```
Match Score = (TF-IDF Cosine Similarity × 0.60) + (Skill Overlap Ratio × 0.40)
```

---

## 🖥️ Screenshots

> _Run the app and upload a resume to see the full dashboard._

**Key UI Sections:**
- **Hero header** with gradient title
- **Upload + JD input** side-by-side
- **Score card** with animated progress bar
- **Stats row** (skills found / missing / JD keywords / suggestions)
- **Skill tags** (green = found, red = missing)
- **Suggestion cards** with icons
- **Expandable panels** for highlighted JD and raw text

---

## ⚙️ Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `PyPDF2` | PDF text extraction |
| `spacy` | NLP tokenization + NER |
| `scikit-learn` | TF-IDF vectorization + cosine similarity |

---

## 🛠️ Troubleshooting

**`spacy` model not found:**
```bash
python -m spacy download en_core_web_sm
```

**PDF text is empty:**
- The PDF might be scanned (image-only). Try a text-based PDF.
- Use OCR tools like Adobe Acrobat or `pytesseract` to convert first.

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

---

## 📄 License

MIT License — free to use, modify, and distribute.
