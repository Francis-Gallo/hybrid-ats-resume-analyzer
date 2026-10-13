# Hybrid ATS Resume Analyzer

Score a PDF resume against a job description, section by section, using embeddings instead of keyword matching,
then get short improvement feedback from a local LLM. Everything runs on your machine through
[LM Studio](https://lmstudio.ai/)'s OpenAI-compatible server, so the resume never leaves it.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![LM Studio](https://img.shields.io/badge/LLM-LM%20Studio%20(local)-6D28D9)

## How it works

```
resume.pdf ──► pdfplumber text ──► split into Skills / Experience / Projects (regex on headings)
                                              │
job description ──► embedding ──┐             ▼
                                └──► cosine similarity per section (nomic-embed-text-v1.5)
                                              │
                         weighted score: 0.40·skills + 0.35·experience + 0.25·projects
                                              │
                                 phi-3.1-mini-4k-instruct ──► why this score + 3 improvements
```

1. `extract_text_from_pdf` reads every page with pdfplumber.
2. `split_sections` lowercases the text and takes what follows the `skills`, `experience` and `projects`
   headings, up to the next known heading (`education` included).
3. Each non-empty section and the job description are embedded with `text-embedding-nomic-embed-text-v1.5`.
   A missing section scores 0.
4. The final score is the weighted sum above. The section scores are shown as percentages.
5. The scores are sent to `phi-3.1-mini-4k-instruct` (temperature 0.2, max 250 tokens), which explains the score
   and suggests three improvements. If the model takes longer than 180 s, the score is still shown.

## Run it

Requirements: Python 3.10+, and LM Studio with its local server enabled on `http://localhost:1234`, with both models loaded:
- `text-embedding-nomic-embed-text-v1.5`
- `phi-3.1-mini-4k-instruct`

```bash
git clone https://github.com/Francis-Gallo/hybrid-ats-resume-analyzer.git
cd hybrid-ats-resume-analyzer
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app/app.py               # http://127.0.0.1:5000
```

Upload a resume PDF, paste the job description and submit.

The model names and URLs are constants at the top of `app/app.py`. Change them there to use other models.

## Project layout

```
app/app.py            Flask app: PDF parsing, sectioning, embeddings, scoring, LLM feedback
templates/index.html  upload form
static/style.css      styles
requirements.txt      flask, pdfplumber, requests, numpy
```

## Limitations

- **Section detection is a simple regex.** It looks for the words "skills", "experience" and "projects".
  Resumes with other headings (e.g. "Work history"), or that use those words in running text, are split badly.
- **Similarity is not a hiring signal.** The score measures how close each section's embedding is to the job
  description. It is not calibrated against real ATS software or recruiter decisions. The weights are hand-picked.
- **Text-based PDFs only.** Scanned PDFs need OCR first.
- **Development server.** It runs Flask with `debug=True`, and the LLM feedback goes into the result HTML without
  escaping. Run it locally only, not as a public service.

## Possible next steps

- Return the result through a template with auto-escaping and turn off debug mode
- More flexible section detection (heading aliases, or let the LLM split the sections)
- A small labelled set of resume/job pairs to sanity-check the scores
- Make the model names and URLs configurable through environment variables

## License

MIT. See [LICENSE](LICENSE).
