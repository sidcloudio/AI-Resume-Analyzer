🧠 AI Resume Analyzer

A production-quality web app that analyzes your resume against a job description using NLP and OCR — giving you a match score, skill gap analysis, and actionable improvement tips.

✨ Features
Feature	Description
📄 PDF Upload	Extracts text from PDF resumes
🖼️ Image Upload	Supports JPG / JPEG / PNG resumes using OCR
🎯 Match Score	TF-IDF cosine similarity + skill overlap
✅ Skill Detection	150+ skills across tech, data, cloud, soft skills
❌ Gap Analysis	Identifies missing skills from job description
💡 Suggestions	Smart improvement recommendations
🔍 Keyword Highlight	Highlights important JD keywords
🎨 Modern UI	Clean and responsive Streamlit interface
🚀 Quick Start
1. Clone / Download
git clone https://github.com/yourname/ai-resume-analyzer.git
cd ai-resume-analyzer
2. Create virtual environment
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
3. Install dependencies
pip install -r requirements.txt
4. Install spaCy model
python -m spacy download en_core_web_sm
5. Install Tesseract OCR (IMPORTANT for images)

👉 Required for image-based resume extraction
Tesseract OCR

Make sure Tesseract is installed and path is set in code:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
6. Run the app
streamlit run app.py

App will open at:

http://localhost:8501
📁 Project Structure
ai_resume_analyzer/
│
├── app.py              # Streamlit UI (PDF + Image upload)
├── utils.py            # NLP + OCR + scoring logic
├── requirements.txt    # Dependencies
├── README.md
└── venv/
🧠 How It Works
Resume Upload (PDF / Image)
        │
        ├── PDF → PyPDF2 text extraction
        └── Image → OCR (Tesseract)
                     │
                     ▼
        Text Preprocessing (spaCy NLP)
                     │
                     ├── Skill Extraction (150+ skills)
                     │
                     ▼
        TF-IDF + Cosine Similarity
                     │
                     ▼
        Match Score (0–100%)
                     │
                     ▼
        Gap Analysis + Suggestions
📊 Match Score Formula
Match Score =
(0.6 × TF-IDF Similarity) + (0.4 × Skill Overlap)
🛠️ Tech Stack
Library	Purpose
Streamlit	Web UI
PyPDF2	PDF parsing
pytesseract	Image OCR
spaCy	NLP processing
scikit-learn	ML similarity
⚠️ Notes
Image resumes must be clear for best OCR results
Scanned PDFs may need OCR conversion
Tesseract must be installed for image support
🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

📄 License

MIT License — free to use and modify.
