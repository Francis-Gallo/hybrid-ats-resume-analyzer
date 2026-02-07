from flask import Flask, render_template, request
import pdfplumber
import requests
import numpy as np
import re

# =========================
# LM Studio Configuration
# =========================
EMBEDDINGS_URL = "http://localhost:1234/v1/embeddings"
CHAT_URL = "http://localhost:1234/v1/chat/completions"

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL = "phi-3.1-mini-4k-instruct"

# =========================
# Flask App
# =========================
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# =========================
# Helper Functions
# =========================

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_sections(text):
    text = text.lower()

    sections = {
        "skills": "",
        "experience": "",
        "projects": ""
    }

    patterns = {
        "skills": r"skills(.*?)(experience|projects|education|$)",
        "experience": r"experience(.*?)(projects|skills|education|$)",
        "projects": r"projects(.*?)(skills|experience|education|$)"
    }

    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.S)
        if match:
            sections[section] = match.group(1).strip()

    return sections


def get_embedding(text):
    response = requests.post(
        EMBEDDINGS_URL,
        json={
            "model": EMBED_MODEL,
            "input": [text]
        },
        timeout=60
    )
    response.raise_for_status()
    return np.array(response.json()["data"][0]["embedding"])


def cosine_similarity(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b))
    )


def get_llm_feedback(skills, experience, projects, final_score):
    prompt = f"""
You are an ATS resume reviewer.

Scores:
- Skills: {skills}%
- Experience: {experience}%
- Projects: {projects}%
- Final ATS score: {final_score}%

Explain briefly:
1) Why the score is at this level
2) Three concrete ways to improve the resume

Be concise and practical.
"""

    try:
        response = requests.post(
            CHAT_URL,
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a professional ATS resume reviewer."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 250,
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.ReadTimeout:
        return (
            "LLM feedback took too long to respond.\n\n"
            "This can happen on limited hardware.\n"
            "Your ATS score is valid — please try again for feedback."
        )

# =========================
# Routes
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    resume_file = request.files["resume"]
    job_description = request.form["job_description"]

    # 1. Extract resume text
    resume_text = extract_text_from_pdf(resume_file)

    # 2. Split resume into sections
    sections = split_sections(resume_text)

    # 3. Embed job description
    job_vec = get_embedding(job_description)

    # 4. Section-wise semantic similarity
    scores = {}
    for section, content in sections.items():
        if content.strip():
            section_vec = get_embedding(content)
            scores[section] = round(
                cosine_similarity(section_vec, job_vec) * 100, 2
            )
        else:
            scores[section] = 0.0

    # 5. Hybrid ATS score (weighted)
    final_score = round(
        0.4 * scores["skills"] +
        0.35 * scores["experience"] +
        0.25 * scores["projects"],
        2
    )

    # 6. LLM feedback (Phi)
    feedback = get_llm_feedback(
        scores["skills"],
        scores["experience"],
        scores["projects"],
        final_score
    )

    return f"""
    <h2>Hybrid ATS Resume Analysis 📊</h2>

    <h3>Section-wise Semantic Match</h3>
    <ul>
        <li><strong>Skills:</strong> {scores['skills']}%</li>
        <li><strong>Experience:</strong> {scores['experience']}%</li>
        <li><strong>Projects:</strong> {scores['projects']}%</li>
    </ul>

    <h2>Final ATS Score: {final_score}%</h2>

    <h3>LLM Feedback (Local • Phi)</h3>
    <pre style="white-space: pre-wrap;">{feedback}</pre>

    <p>
        Generated locally using <strong>LM Studio</strong> with:
        <ul>
            <li>text-embedding-nomic-embed-text-v1.5</li>
            <li>phi-3.1-mini-4k-instruct</li>
        </ul>
    </p>

    <a href="/">Go Back</a>
    """


if __name__ == "__main__":
    app.run(debug=True)
