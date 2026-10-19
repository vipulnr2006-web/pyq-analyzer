import re
from collections import Counter
from io import BytesIO

import fitz
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PYQ Analyzer", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main {background: #f7f9fc;}
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #e6eaf0; padding: 16px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("📚 PYQ Analyzer & Exam Pattern Predictor")
st.caption("Upload previous-year question papers to identify repeated questions, frequent topics, and revision priorities.")

with st.sidebar:
    st.header("Analysis settings")
    similarity_threshold = st.slider(
        "Repeated-question similarity", min_value=0.50, max_value=0.95,
        value=0.72, step=0.01,
        help="Higher values require questions to be more similar before they are grouped."
    )
    st.info("This MVP uses text extracted from PDFs. Scanned/image-only PDFs may need OCR before analysis.")

def extract_pdf_text(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        return "\n".join(pages).strip(), len(doc)
    except Exception as exc:
        return "", 0

def clean_line(line):
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^[\s\-\•\*\(\)\[\]\d\.\:]+", "", line)
    return line.strip()

def extract_question_candidates(text):
    lines = [clean_line(x) for x in text.splitlines()]
    lines = [x for x in lines if len(x) >= 18]
    candidates = []
    buffer = ""
    starts_question = re.compile(
        r"^(?:Q(?:uestion)?\s*)?\d{1,2}\s*[\.\):\-]|^(?:[a-h]\s*[\)\.]|\([a-h]\))",
        re.IGNORECASE
    )
    for line in lines:
        if starts_question.search(line):
            if buffer:
                candidates.append(buffer.strip())
            buffer = line
        elif buffer:
            buffer += " " + line
        elif "?" in line or re.search(r"\b(explain|describe|define|discuss|differentiate|compare|write|what|why|how|derive|state|list)\b", line, re.I):
            buffer = line
    if buffer:
        candidates.append(buffer.strip())

    # If numbering was not recognized, fall back to question-like sentences/lines.
    if len(candidates) < 2:
        fallback = []
        for line in lines:
            if "?" in line or re.search(r"\b(explain|describe|define|discuss|differentiate|compare|write|what|why|how|derive|state|list|illustrate)\b", line, re.I):
                if len(line) >= 25:
                    fallback.append(line)
        candidates = fallback

    # Remove duplicate/short entries and obvious page boilerplate.
    out, seen = [], set()
    for q in candidates:
        q = re.sub(r"\s+", " ", q).strip(" -")
        key = re.sub(r"[^a-z0-9]+", "", q.lower())
        if len(q) >= 20 and key not in seen and not re.fullmatch(r"\d+", q):
            seen.add(key)
            out.append(q)
    return out

def make_question_frame(files_data):
    rows = []
    for file_name, text, pages in files_data:
        questions = extract_question_candidates(text)
        for idx, question in enumerate(questions, start=1):
            rows.append({
                "Paper": file_name,
                "Question No. (detected)": idx,
                "Question": question,
                "Pages": pages,
            })
    return pd.DataFrame(rows)

def repeated_questions(df, threshold):
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=["Similarity", "Question 1", "Paper 1", "Question 2", "Paper 2"])
    texts = df["Question"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(matrix)
    found = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if df.iloc[i]["Paper"] == df.iloc[j]["Paper"]:
                continue
            score = float(sims[i, j])
            if score >= threshold:
                found.append({
                    "Similarity": round(score * 100, 1),
                    "Question 1": texts[i],
                    "Paper 1": df.iloc[i]["Paper"],
                    "Question 2": texts[j],
                    "Paper 2": df.iloc[j]["Paper"],
                })
    return pd.DataFrame(found).sort_values("Similarity", ascending=False) if found else pd.DataFrame(
        columns=["Similarity", "Question 1", "Paper 1", "Question 2", "Paper 2"]
    )

STOPWORDS_EXTRA = {
    "question", "following", "using", "used", "given", "write", "explain", "describe",
    "define", "discuss", "state", "what", "why", "how", "with", "from", "into",
    "which", "this", "that", "each", "also", "show", "derive", "illustrate", "differentiate"
}

def topic_frequency(texts):
    words = []
    for text in texts:
        terms = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        words.extend(w for w in terms if w not in STOPWORDS_EXTRA)
    counts = Counter(words)
    return pd.DataFrame(counts.most_common(20), columns=["Term / possible topic", "Frequency"])

uploaded = st.file_uploader(
    "Upload two or more previous-year question papers (PDF)",
    type=["pdf"], accept_multiple_files=True,
    help="For best results, use text-based PDFs rather than scanned images."
)

if not uploaded:
    st.markdown("### Start here")
    c1, c2, c3 = st.columns(3)
    c1.metric("Upload", "PDF papers")
    c2.metric("Analyze", "Repeated questions")
    c3.metric("Prioritize", "Frequent terms")
    st.write("Upload question papers above to generate an analysis. Files are processed for this session and are not saved in a permanent database.")
    st.markdown("**Current MVP includes:** PDF text extraction, heuristic question detection, TF-IDF similarity, term-frequency ranking, and downloadable CSV results.")
    st.stop()

if len(uploaded) < 2:
    st.warning("Upload at least two papers to compare repeated questions across years.")

files_data = []
with st.spinner("Reading PDF files..."):
    for f in uploaded:
        text, pages = extract_pdf_text(f)
        if text:
            files_data.append((f.name, text, pages))
        else:
            st.warning(f"Could not extract text from **{f.name}**. It may be scanned, encrypted, or empty.")

if not files_data:
    st.error("No readable PDF text was found. Try a text-based PDF or run OCR on scanned papers first.")
    st.stop()

all_text = "\n".join(x[1] for x in files_data)
questions_df = make_question_frame(files_data)
repeats_df = repeated_questions(questions_df, similarity_threshold)
topics_df = topic_frequency([row["Question"] for _, row in questions_df.iterrows()]) if not questions_df.empty else pd.DataFrame(columns=["Term / possible topic", "Frequency"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Papers processed", len(files_data))
m2.metric("Pages read", sum(x[2] for x in files_data))
m3.metric("Questions detected", len(questions_df))
m4.metric("Similar pairs", len(repeats_df))

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔁 Repeated questions", "🎯 Topic frequency", "📄 Extracted questions"])

with tab1:
    st.subheader("Paper overview")
    overview = pd.DataFrame([{
        "Paper": name,
        "Pages": pages,
        "Characters extracted": len(text),
        "Questions detected": len(extract_question_candidates(text)),
    } for name, text, pages in files_data])
    st.dataframe(overview, use_container_width=True, hide_index=True)
    if not topics_df.empty:
        chart = px.bar(topics_df.head(12).sort_values("Frequency"), x="Frequency", y="Term / possible topic",
                       orientation="h", title="Most frequent terms in detected questions")
        st.plotly_chart(chart, use_container_width=True)
    st.caption("Term frequency is a simple signal, not a syllabus-aware topic model. Review the results before relying on them.")

with tab2:
    st.subheader("Potential repeated or similar questions")
    st.caption(f"Pairs are shown when TF-IDF cosine similarity is at least {similarity_threshold:.0%}. Similar wording does not guarantee identical marks or intent.")
    if repeats_df.empty:
        st.info("No cross-paper pairs crossed the selected threshold. Try lowering the threshold slightly, or check whether the PDFs contain extractable text.")
    else:
        st.dataframe(repeats_df, use_container_width=True, hide_index=True)
        st.download_button("Download repeated-question pairs (CSV)", repeats_df.to_csv(index=False).encode("utf-8"), "repeated_questions.csv", "text/csv")

with tab3:
    st.subheader("Revision priority signals")
    st.write("These are ranked by frequency in detected question text. They are **not guaranteed exam predictions**.")
    if topics_df.empty:
        st.info("No terms were extracted. Check the PDFs.")
    else:
        st.dataframe(topics_df, use_container_width=True, hide_index=True)
        st.download_button("Download topic-frequency report (CSV)", topics_df.to_csv(index=False).encode("utf-8"), "topic_frequency.csv", "text/csv")

with tab4:
    st.subheader("Detected questions")
    if questions_df.empty:
        st.warning("The question detector did not find enough question-like text. This may happen with unusual PDF layouts.")
    else:
        st.dataframe(questions_df[["Paper", "Question No. (detected)", "Question"]], use_container_width=True, hide_index=True)
        st.download_button("Download extracted questions (CSV)", questions_df.to_csv(index=False).encode("utf-8"), "extracted_questions.csv", "text/csv")

with st.expander("How this MVP works and current limitations"):
    st.markdown("""
    - **PDF text:** extracted using PyMuPDF. Scanned PDFs need OCR.
    - **Question detection:** heuristic rules based on numbering and question-style wording; complex layouts can be missed.
    - **Similarity:** TF-IDF and cosine similarity compare wording. A lecturer should review matches.
    - **Topic frequency:** counts terms, not true syllabus units. Add unit labels or a syllabus mapping in a later version.
    - **Prediction:** frequency is a revision signal only. Past papers cannot guarantee future questions.
    - **Storage:** uploads and results are processed in the current app session; this starter does not save a permanent question-paper library or user accounts.
    """)
