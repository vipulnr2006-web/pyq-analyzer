# PYQ Analyzer & Exam Pattern Predictor — MVP

A Streamlit MVP for analyzing previous-year question papers (PDFs).

## Features
- Upload multiple PDF papers
- Extract text using PyMuPDF
- Detect question-like passages using heuristic rules
- Find potential repeated questions with TF-IDF and cosine similarity
- Rank frequent terms as rough revision signals
- View and download CSV reports

## Run locally
Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy free
1. Push this folder to a GitHub repository.
2. Sign in at https://share.streamlit.io/
3. Choose **Create app**, select the repository, branch `main`, and entrypoint `app.py`.
4. Deploy and copy the `streamlit.app` URL.

## Limitations
This is a prototype, not a validated exam-prediction model. PDF question detection is heuristic, and scanned PDFs need OCR. Similarity and term frequency must be reviewed by a human. It does not currently include user accounts, an admin panel, permanent database storage, or syllabus-aware unit mapping.
