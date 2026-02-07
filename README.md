project:
  name: Hybrid ATS Resume Analyzer
  description: >
    A production-grade Applicant Tracking System (ATS) that evaluates resumes
    using semantic similarity, section-wise analysis, and local LLM feedback.
    Fully offline, privacy-safe, and OpenAI-compatible via LM Studio.

features:
  - Resume parsing from PDF
  - Section-wise semantic analysis:
      - Skills
      - Experience
      - Projects
  - Semantic similarity using embeddings (not keyword matching)
  - Hybrid ATS score with weighted aggregation
  - Local LLM feedback explaining resume strengths and improvements
  - Fully local and privacy-safe (no cloud APIs)

architecture:
  pipeline:
    - Resume PDF
    - PDF Text Extraction
    - Section Splitter (Skills / Experience / Projects)
    - Semantic Embeddings (LM Studio)
    - Cosine Similarity
    - Hybrid ATS Score
    - LLM Feedback (Phi via LM Studio)

technologies:
  backend:
    - Python
    - Flask
  ai:
    - LM Studio (OpenAI-compatible local inference)
    - text-embedding-nomic-embed-text-v1.5
    - phi-3.1-mini-4k-instruct
  libraries:
    - NumPy
    - PDFPlumber

scoring_logic:
  description: Weighted section-wise semantic similarity
  formula:
    skills: 0.40
    experience: 0.35
    projects: 0.25

setup:
  steps:
    - Clone repository:
        command: git clone https://github.com/Francis-Gallo/hybrid-ats-resume-analyzer.git
    - Create virtual environment:
        command: python -m venv venv
    - Activate virtual environment:
        windows: venv\\Scripts\\activate
    - Install dependencies:
        command: pip install -r requirements.txt
    - Start LM Studio:
        requirements:
          - Enable OpenAI-compatible server
          - Load embedding model: text-embedding-nomic-embed-text-v1.5
          - Load LLM model: phi-3.1-mini-4k-instruct
    - Run application:
        command: python app/app.py
        url: http://127.0.0.1:5000

output:
  - Section-wise semantic match scores
  - Final ATS score
  - LLM-generated resume improvement feedback

design_principles:
  - Privacy-first processing
  - No external API costs
  - Deterministic local inference
  - Production-style decoupled architecture

license: MIT
