import hashlib
import io
import os
import re
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Document Assistant")
st.caption("PDF, DOCX, TXT, and Markdown • Hybrid search • Groq answers")


# -----------------------------
# Models and clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    pages = []
    reader = PdfReader(io.BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "text": text,
            "filename": filename,
            "page": page_number,
        })

    return pages


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }]


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="replace")

    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }]


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="replace")

    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }]


def extract_document(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    raise ValueError(f"Unsupported file type: {extension}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(page_records, chunk_size=900, overlap=150):
    """Create overlapping chunks while preserving filename/page metadata."""
    chunks = []

    for record in page_records:
        text = re.sub(r"\s+", " ", record["text"]).strip()

        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "filename": record["filename"],
                    "page": record["page"],
                })

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


# -----------------------------
# Embeddings and FAISS
# -----------------------------
def build_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    return embeddings


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in",
    "on", "for", "and", "or", "with", "from", "what", "who", "when",
    "where", "why", "how", "does", "do", "did", "this", "that", "it",
    "about", "tell", "me", "please",
}


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_search(question, chunks):
    words = important_words(question)
    results = []

    for index, chunk in enumerate(chunks):
        text = chunk["text"].lower()

        if not words:
            score = 0.0
        else:
            matches = sum(1 for word in words if word in text)
            score = matches / len(words)

        results.append((index, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, chunks, index, model, top_k=5):
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    semantic_scores, semantic_indices = index.search(
        query_embedding, min(len(chunks), max(top_k * 3, 10))
    )

    semantic = {
        int(chunk_index): float(score)
        for score, chunk_index in zip(semantic_scores[0], semantic_indices[0])
        if chunk_index >= 0
    }

    keyword = dict(keyword_search(question, chunks))

    candidates = set(semantic) | {
        index for index, score in sorted(
            keyword.items(), key=lambda item: item[1], reverse=True
        )[: max(top_k * 3, 10)]
    }

    ranked = []
    for chunk_index in candidates:
        semantic_score = semantic.get(chunk_index, 0.0)
        keyword_score = keyword.get(chunk_index, 0.0)

        # Both scores are in approximately 0..1.
        hybrid_score = (0.75 * semantic_score) + (0.25 * keyword_score)

        ranked.append({
            "index": chunk_index,
            "score": hybrid_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "chunk": chunks[chunk_index],
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


# -----------------------------
# Groq answer generation
# -----------------------------
def answer_question(question, retrieved_chunks, client, model_name):
    context_parts = []

    for number, result in enumerate(retrieved_chunks, start=1):
        chunk = result["chunk"]
        page = f", page {chunk['page']}" if chunk["page"] else ""

        context_parts.append(
            f"[Source {number}: {chunk['filename']}{page}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a document question-answering assistant.

Answer the user's question ONLY from the provided document context.
Do not use outside knowledge.
If the answer is not available in the provided context, say:
"I couldn't find that information in the provided documents."

Be concise and factual. Do not invent details.
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# -----------------------------
# File loading helpers
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def file_signature(filename, file_bytes):
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{filename}:{digest}"


def load_local_files(uploaded_files):
    loaded = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        loaded.append((uploaded_file.name, file_bytes))

    return loaded


def load_drive_files(drive_link):
    if not drive_link.strip():
        return []

    temp_dir = Path(tempfile.mkdtemp(prefix="ai_doc_assistant_"))

    try:
        result = gdown.download_folder(
            url=drive_link.strip(),
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
        )

        if not result:
            # Also try the link as a single public Drive file.
            single_path = temp_dir / "downloaded_file"
            downloaded = gdown.download(
                url=drive_link.strip(),
                output=str(single_path),
                quiet=True,
                fuzzy=True,
            )
            result = [downloaded] if downloaded else []

        files = []

        for path in temp_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append((path.name, path.read_bytes()))

        return files

    finally:
        # The temporary directory is left for the running process and OS cleanup.
        pass


# -----------------------------
# Session state
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=400,
        max_value=1600,
        value=900,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=300,
        value=150,
        step=25,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    groq_model = st.text_input(
        "Groq model",
        value="openai/gpt-oss-120b",
    )

    if st.button("Clear processed documents"):
        st.session_state.documents = {}
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.rerun()


# -----------------------------
# Document sources
# -----------------------------
st.subheader("1. Add documents")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

drive_link = st.text_input(
    "Or paste a public Google Drive file/folder link",
    placeholder="https://drive.google.com/...",
)

if st.button("Process documents", type="primary"):
    source_files = load_local_files(uploaded_files or [])

    if drive_link.strip():
        try:
            drive_files = load_drive_files(drive_link)
            source_files.extend(drive_files)
        except Exception as error:
            st.error(f"Google Drive loading failed: {error}")

    if not source_files:
        st.warning("Add at least one local file or Google Drive link.")
    else:
        model = load_embedding_model()
        new_documents = 0

        progress = st.progress(0)

        for position, (filename, file_bytes) in enumerate(source_files, start=1):
            signature = file_signature(filename, file_bytes)

            # Avoid extracting/embedding the same file again.
            if signature in st.session_state.documents:
                progress.progress(position / len(source_files))
                continue

            try:
                pages = extract_document(file_bytes, filename)
                chunks = chunk_text(
                    pages,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                st.session_state.documents[signature] = {
                    "filename": filename,
                    "pages": pages,
                    "chunks": chunks,
                }

                new_documents += 1

            except Exception as error:
                st.error(f"Could not process {filename}: {error}")

            progress.progress(position / len(source_files))

        # Rebuild the searchable collection only when new documents were added.
        if new_documents:
            all_chunks = []

            for document in st.session_state.documents.values():
                all_chunks.extend(document["chunks"])

            with st.spinner("Creating embeddings and FAISS index..."):
                embeddings = build_embeddings(all_chunks, model)
                faiss_index = build_faiss_index(embeddings)

            st.session_state.chunks = all_chunks
            st.session_state.embeddings = embeddings
            st.session_state.faiss_index = faiss_index

        st.success(
            f"Processed {new_documents} new document(s). "
            f"Total chunks: {len(st.session_state.chunks)}"
        )


# -----------------------------
# Document information
# -----------------------------
if st.session_state.documents:
    st.subheader("2. Document information")

    for document in st.session_state.documents.values():
        filename = document["filename"]
        chunks = document["chunks"]

        with st.expander(f"📄 {filename} — {len(chunks)} chunks"):
            extracted_text = "\n\n".join(
                record["text"]
                for record in document["pages"]
                if record["text"].strip()
            )

            st.write(f"**Filename:** {filename}")
            st.write(f"**Chunks:** {len(chunks)}")
            st.text_area(
                "Extracted text",
                extracted_text,
                height=220,
                key=f"text_{hashlib.md5(filename.encode()).hexdigest()}",
            )


# -----------------------------
# Question answering
# -----------------------------
st.subheader("3. Ask a question")

question = st.text_input(
    "Question",
    placeholder="Ask something about your documents...",
)

if st.button("Ask"):
    if not st.session_state.chunks or st.session_state.faiss_index is None:
        st.warning("Process at least one document first.")
    elif not question.strip():
        st.warning("Enter a question.")
    elif not st.secrets.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing from Streamlit secrets.")
    else:
        model = load_embedding_model()

        with st.spinner("Searching documents..."):
            retrieved = hybrid_search(
                question,
                st.session_state.chunks,
                st.session_state.faiss_index,
                model,
                top_k=top_k,
            )

        try:
            client = get_groq_client(st.secrets["GROQ_API_KEY"])

            with st.spinner("Generating answer..."):
                answer = answer_question(
                    question,
                    retrieved,
                    client,
                    groq_model,
                )

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Retrieved sources")

            for number, result in enumerate(retrieved, start=1):
                chunk = result["chunk"]
                page = (
                    f"Page {chunk['page']}"
                    if chunk["page"]
                    else "Page not available"
                )

                with st.expander(
                    f"{number}. {chunk['filename']} — {page} "
                    f"(hybrid score: {result['score']:.3f})"
                ):
                    st.write(chunk["text"])
                    st.caption(
                        f"Semantic: {result['semantic_score']:.3f} | "
                        f"Keyword: {result['keyword_score']:.3f}"
                    )

        except Exception as error:
            st.error(f"Groq request failed: {error}")
