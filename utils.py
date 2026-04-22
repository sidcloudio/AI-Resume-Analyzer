"""
utils.py - NLP Helper Functions for AI Resume Analyzer

Contains:
- PDF text extraction
- Text preprocessing
- Skill extraction
- Match score calculation
- Suggestion generation
- Keyword highlighting
"""

import re
import io
import math
from collections import Counter
from PIL import Image
import pytesseract
import PyPDF2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─── PDF Reading ──────────────────────────────────────────────────────────────
try:
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# ─── NLP ──────────────────────────────────────────────────────────────────────
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback: blank English model if the trained model isn't installed
        nlp = spacy.blank("en")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# ─── scikit-learn (optional) ──────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SKILLS MASTER LIST  (tech + soft + domain)
# ══════════════════════════════════════════════════════════════════════════════
SKILLS_DB = {
    # ── Programming Languages ──
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "ruby",
    "golang", "go", "rust", "swift", "kotlin", "scala", "r", "matlab",
    "perl", "php", "bash", "shell", "powershell", "dart", "lua",

    # ── Web / Frontend ──
    "html", "css", "react", "reactjs", "react.js", "angular", "angularjs",
    "vue", "vue.js", "vuejs", "next.js", "nextjs", "nuxt", "svelte",
    "jquery", "bootstrap", "tailwind", "tailwindcss", "sass", "less",
    "webpack", "vite", "rollup", "babel",

    # ── Backend / Frameworks ──
    "node.js", "nodejs", "express", "django", "flask", "fastapi",
    "spring", "spring boot", "rails", "ruby on rails", "laravel",
    "asp.net", ".net", "dotnet", "graphql", "rest", "restful", "grpc",
    "microservices", "serverless",

    # ── Data Science / ML / AI ──
    "machine learning", "deep learning", "neural networks", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "torch",
    "hugging face", "transformers", "langchain", "openai", "llm",
    "large language model", "bert", "gpt", "xgboost", "lightgbm",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "jupyter", "statistics", "data analysis", "feature engineering",
    "model deployment", "mlops", "mlflow", "kubeflow",

    # ── Databases ──
    "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
    "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb",
    "firebase", "supabase", "neo4j", "influxdb",

    # ── Cloud & DevOps ──
    "aws", "azure", "gcp", "google cloud", "heroku", "vercel", "netlify",
    "docker", "kubernetes", "k8s", "helm", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd", "circleci", "travis",
    "linux", "unix", "nginx", "apache",

    # ── Tools & Practices ──
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "agile", "scrum", "kanban", "tdd", "bdd", "unit testing",
    "selenium", "pytest", "jest", "mocha", "cypress",
    "postman", "swagger", "openapi", "figma", "sketch",

    # ── Data Engineering ──
    "spark", "hadoop", "kafka", "airflow", "dbt", "etl", "data pipeline",
    "data warehouse", "snowflake", "bigquery", "redshift", "databricks",
    "dask", "hive", "flink",

    # ── Cybersecurity ──
    "cybersecurity", "penetration testing", "ethical hacking",
    "siem", "soc", "vulnerability assessment", "firewalls",
    "encryption", "oauth", "jwt",

    # ── Mobile ──
    "android", "ios", "react native", "flutter", "xamarin",

    # ── Soft Skills ──
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "project management", "time management",
    "collaboration", "adaptability", "mentoring", "presentation",
    "stakeholder management", "negotiation",

    # ── Business / Domain ──
    "product management", "business analysis", "data visualization",
    "tableau", "power bi", "looker", "excel", "spreadsheets",
    "financial modeling", "crm", "salesforce", "sap", "erp",
    "digital marketing", "seo", "content marketing",
}

# Normalise to lowercase set
SKILLS_DB = {s.lower() for s in SKILLS_DB}


# ══════════════════════════════════════════════════════════════════════════════
# 1. PDF TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from an uploaded PDF file object.
    Supports both PyPDF2 and a simple fallback.
    """
    if not PYPDF2_AVAILABLE:
        return "PyPDF2 not installed. Please run: pip install PyPDF2"

    try:
        pdf_bytes = uploaded_file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        full_text = "\n".join(text_parts)
        return full_text
    except Exception as e:
        return f"ERROR: Could not read PDF — {str(e)}"
    

def extract_text_from_image(uploaded_file) -> str:
    """
    Extract text from JPEG/PNG image using OCR (Tesseract)
    """
    try:
        image = Image.open(uploaded_file)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # OCR se text extract karo
        text = pytesseract.image_to_string(image, lang='eng')
        return text
    except Exception as e:
        return f"ERROR: Image read nahi hui — {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
# Common English stopwords (inline, no NLTK dependency)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "also",
    "this", "that", "these", "those", "it", "its", "we", "our", "you",
    "your", "he", "she", "they", "their", "what", "which", "who", "whom",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "only", "own",
    "same", "so", "than", "too", "very", "just", "as", "if", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "up", "about", "against", "include", "including", "within",
    "without", "across", "along", "around", "work", "working", "worked",
    "experience", "year", "years", "team", "teams", "using", "used", "use",
    "new", "strong", "well", "ability", "role", "position", "job",
    "candidate", "company", "opportunities", "opportunity",
}

def preprocess_text(text: str) -> str:
    """
    Clean and normalise text:
      - Lowercase
      - Remove special characters (keep spaces + alphanumeric)
      - Remove stopwords
      - Return space-joined tokens
    """
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Replace newlines/tabs with space
    text = re.sub(r"[\n\r\t]+", " ", text)
    # Remove non-alphanumeric characters except spaces and dots (for version nums)
    text = re.sub(r"[^a-z0-9\.\+\# ]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenise and remove stopwords
    tokens = [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SKILL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_skills(text: str) -> set:
    """
    Extract skills from text using:
      1. Direct multi-word phrase matching against SKILLS_DB
      2. Single-word token matching
      3. (Optional) spaCy NER for ORG/PRODUCT entities that may be tech tools
    """
    text_lower = text.lower()
    found = set()

    # ── Pass 1: Multi-word phrase matching ──
    # Sort by length descending so longer phrases match first
    sorted_skills = sorted(SKILLS_DB, key=lambda s: len(s.split()), reverse=True)
    for skill in sorted_skills:
        # Use word-boundary aware search
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)

    # ── Pass 2: spaCy NER-assisted extraction ──
    if SPACY_AVAILABLE and nlp and nlp.pipe_names:
        doc = nlp(text[:50000])  # Limit for performance
        for ent in doc.ents:
            ent_lower = ent.text.lower().strip()
            if ent.label_ in ("ORG", "PRODUCT") and ent_lower in SKILLS_DB:
                found.add(ent_lower)
            # Also catch noun chunks
        for chunk in doc.noun_chunks:
            chunk_lower = chunk.text.lower().strip()
            if chunk_lower in SKILLS_DB:
                found.add(chunk_lower)

    return found


# ══════════════════════════════════════════════════════════════════════════════
# 4. JOB DESCRIPTION KEYWORD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_job_keywords(jd_text: str) -> set:
    """
    Extract all meaningful keywords from the job description
    (used for display purposes and highlighting).
    """
    text_lower = jd_text.lower()
    # Remove stopwords and short words
    tokens = re.findall(r"\b[a-z][a-z0-9\+\#\.]*\b", text_lower)
    keywords = {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2}
    return keywords


# ══════════════════════════════════════════════════════════════════════════════
# 5. MATCH SCORE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════
def _cosine_similarity_manual(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two term-frequency dicts."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot   = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def calculate_match_score(
    resume_clean: str,
    jd_clean: str,
    resume_skills: set,
    jd_skills: set,
) -> int:
    """
    Hybrid match score (0–100):
      - 60% weight: TF-IDF cosine similarity (or manual fallback)
      - 40% weight: Skill overlap ratio
    Returns integer percentage.
    """
    # ── Skill overlap score (40%) ──
    if jd_skills:
        skill_overlap = len(resume_skills & jd_skills) / len(jd_skills)
    else:
        skill_overlap = 0.0

    # ── Text similarity score (60%) ──
    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )
            tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
            text_sim = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])
        except Exception:
            text_sim = 0.0
    else:
        # Manual TF cosine fallback
        freq_r = Counter(resume_clean.split())
        freq_j = Counter(jd_clean.split())
        text_sim = _cosine_similarity_manual(dict(freq_r), dict(freq_j))

    # ── Weighted blend ──
    blended = (text_sim * 0.60) + (skill_overlap * 0.40)
    score = int(round(min(blended * 100, 100)))
    return score


# ══════════════════════════════════════════════════════════════════════════════
# 6. MISSING SKILLS
# ══════════════════════════════════════════════════════════════════════════════
def get_missing_skills(resume_skills: set, jd_skills: set) -> set:
    """Return skills present in JD but absent from resume."""
    return jd_skills - resume_skills


# ══════════════════════════════════════════════════════════════════════════════
# 7. IMPROVEMENT SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════════
# Resume tips bank
_BASE_TIPS = [
    "Add quantified achievements (e.g., 'Improved API response time by 35%') to demonstrate measurable impact.",
    "Use strong action verbs at the start of each bullet point: 'Architected', 'Optimised', 'Delivered', 'Led'.",
    "Keep your resume to 1–2 pages — recruiters spend an average of 7 seconds on the first scan.",
    "Include a concise professional summary at the top tailored to this specific role.",
    "Ensure consistent formatting — uniform font sizes, bullet styles, and date formats throughout.",
    "Add links to your GitHub, portfolio, or LinkedIn profile for social proof.",
    "List education details including GPA (if above 3.5) and relevant coursework or certifications.",
    "Tailor your skills section to mirror the exact keywords used in the job description.",
    "Use industry-standard section headers: 'Experience', 'Skills', 'Education', 'Projects'.",
    "Proofread for grammar and spelling — use tools like Grammarly for a final pass.",
]

def generate_suggestions(
    resume_text: str,
    resume_skills: set,
    missing_skills: set,
    match_score: int,
) -> list:
    """
    Generate contextual improvement suggestions based on:
      - Match score
      - Missing skills
      - Resume content patterns
    """
    suggestions = []

    # ── Score-based suggestion ──
    if match_score < 40:
        suggestions.append(
            f"Your match score is {match_score}%. Consider significantly tailoring "
            "your resume to align with this job description — rework your summary and "
            "skills section to mirror the JD language."
        )
    elif match_score < 70:
        suggestions.append(
            f"Your match score is {match_score}%. You're on the right track — "
            "focus on adding the missing skills below and reframing existing experience "
            "using the job description's terminology."
        )
    else:
        suggestions.append(
            f"Excellent match at {match_score}%! Fine-tune by reinforcing your top "
            "skills with specific metrics and project outcomes to stand out further."
        )

    # ── Missing skills suggestions ──
    if missing_skills:
        top_missing = sorted(missing_skills)[:5]
        skills_str  = ", ".join(f"'{s}'" for s in top_missing)
        suggestions.append(
            f"The following skills appear in the job description but are missing from "
            f"your resume: {skills_str}. Add them to your skills section or demonstrate "
            "them through project descriptions."
        )

    # ── Content-based checks ──
    text_lower = resume_text.lower()

    # Quantification check
    has_numbers = bool(re.search(r"\b\d+[\%\+xX]?\b", resume_text))
    if not has_numbers:
        suggestions.append(
            "No quantified metrics were detected. Add numbers to your achievements "
            "(e.g., team size, percentages, scale, timelines) — this is one of the "
            "most impactful resume improvements."
        )

    # Length check
    word_count = len(resume_text.split())
    if word_count < 200:
        suggestions.append(
            f"Your resume appears quite short (~{word_count} words). Expand with more "
            "detailed descriptions of projects, responsibilities, and accomplishments."
        )
    elif word_count > 1000:
        suggestions.append(
            f"Your resume is quite long (~{word_count} words). Consider condensing to "
            "the most impactful bullet points — aim for a 1–2 page document."
        )

    # Action verbs check
    action_verbs = ["developed", "built", "designed", "led", "managed", "created",
                    "implemented", "optimised", "optimized", "architected", "delivered",
                    "launched", "improved", "reduced", "increased", "streamlined"]
    found_verbs = [v for v in action_verbs if v in text_lower]
    if len(found_verbs) < 3:
        suggestions.append(
            "Use more strong action verbs to start your bullet points. Examples: "
            "'Architected', 'Spearheaded', 'Optimised', 'Delivered', 'Accelerated'."
        )

    # LinkedIn/GitHub check
    if "github" not in text_lower and "linkedin" not in text_lower:
        suggestions.append(
            "Include links to your GitHub profile and LinkedIn page — these provide "
            "social proof and give recruiters easy access to your work and network."
        )

    # Certifications check
    cert_keywords = ["certified", "certification", "certificate", "aws certified",
                     "google certified", "pmp", "cpa", "cfa"]
    has_certs = any(kw in text_lower for kw in cert_keywords)
    if not has_certs and match_score < 70:
        suggestions.append(
            "Consider adding relevant certifications. For this role, certifications "
            "in the missing skill areas can significantly strengthen your candidacy."
        )

    # Summary/objective check
    has_summary = any(kw in text_lower for kw in
                      ["summary", "objective", "profile", "about me", "overview"])
    if not has_summary:
        suggestions.append(
            "Add a 2–3 line professional summary at the top of your resume tailored "
            "to this specific role — it's the first thing recruiters read."
        )

    # ── Pad with base tips if needed ──
    for tip in _BASE_TIPS:
        if len(suggestions) >= 6:
            break
        # Avoid near-duplicates
        tip_lower = tip.lower()
        duplicate = any(
            tip_lower[:40] in s.lower() for s in suggestions
        )
        if not duplicate:
            suggestions.append(tip)

    return suggestions[:8]  # Return top 8 suggestions


# ══════════════════════════════════════════════════════════════════════════════
# 8. KEYWORD HIGHLIGHTING
# ══════════════════════════════════════════════════════════════════════════════
def highlight_missing_keywords(jd_text: str, missing_skills: set) -> str:
    """
    Return HTML version of the job description with missing skill keywords
    highlighted in red so the user can visually identify gaps.
    """
    if not missing_skills:
        return jd_text.replace("\n", "<br>")

    # Escape HTML special chars
    safe_text = (
        jd_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    # Sort by length desc to match longer phrases first
    sorted_missing = sorted(missing_skills, key=len, reverse=True)

    for skill in sorted_missing:
        pattern = re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)
        safe_text = pattern.sub(
            lambda m: f'<span class="kw-missing">{m.group(0)}</span>',
            safe_text,
        )

    return safe_text.replace("\n", "<br>")
