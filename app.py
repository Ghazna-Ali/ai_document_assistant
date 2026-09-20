```python
import hashlib
import io
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


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Document Assistant")
st.caption(
    "PDF, DOCX, TXT and Markdown • Hybrid Search • Groq"
)


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}

STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "from",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "does",
    "do",
    "did",
    "this",
    "that",
    "it",
    "about",
    "tell",
    "me",
    "please",
}


# ============================================================
# CACHED MODELS / CLIENTS
# ============================================================

@st.cache_resource
def load_embedding_model():
    """
    Load the Sentence Transformer model once.
    """
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_groq_client(api_key):
    """
    Create the Groq client once for the current API key.
    """
    return Groq(api_key=api_key)


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(file_bytes, filename):
    """
    Extract PDF text page by page.

    Page numbers are preserved because PDFs have
    explicit page boundaries.
    """

    pages = []

    reader = PdfReader(
        io.BytesIO(file_bytes)
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        pages.append(
            {
                "text": text,
                "filename": filename,
                "page": page_number,
            }
        )

    return pages


def extract_docx(file_bytes, filename):
    """
    Extract text from a DOCX file.

    DOCX page numbers are not reliably available from
    python-docx, so page is stored as None.
    """

    document = Document(
        io.BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(
                paragraph.text
            )

    text = "\n".join(paragraphs)

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_txt(file_bytes, filename):
    """
    Extract a TXT document.
    """

    text = file_bytes.decode(
        "utf-8",
        errors="replace",
    )

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_md(file_bytes, filename):
    """
    Extract a Markdown document.
    """

    text = file_bytes.decode(
        "utf-8",
        errors="replace",
    )

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_document(file_bytes, filename):
    """
    Select the correct extraction function based
    on the file extension.
    """

    extension = Path(
        filename
    ).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(
            file_bytes,
            filename,
        )

    if extension == ".docx":
        return extract_docx(
            file_bytes,
            filename,
        )

    if extension == ".txt":
        return extract_txt(
            file_bytes,
            filename,
        )

    if extension == ".md":
        return extract_md(
            file_bytes,
            filename,
        )

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    page_records,
    chunk_size=900,
    overlap=150,
):
    """
    Split extracted text into overlapping chunks.

    Every chunk keeps:

    - text
    - filename
    - page
    """

    chunks = []

    for record in page_records:

        text = re.sub(
            r"\s+",
            " ",
            record["text"],
        ).strip()

        if not text:
            continue

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text),
            )

            chunk = text[
                start:end
            ].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "filename": record["filename"],
                        "page": record["page"],
                    }
                )

            if end >= len(text):
                break

            start = max(
                0,
                end - overlap,
            )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def build_embeddings(
    chunks,
    model,
):
    """
    Create embeddings for all chunks.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype(
        "float32"
    )


# ============================================================
# FAISS
# ============================================================

def build_faiss_index(
    embeddings,
):
    """
    Create a FAISS similarity index.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# KEYWORD SEARCH
# ============================================================

def important_words(question):
    """
    Extract meaningful words from the question.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        question.lower(),
    )

    return [
        word
        for word in words
        if (
            word not in STOP_WORDS
            and len(word) > 2
        )
    ]


def keyword_search(
    question,
    chunks,
):
    """
    Simple keyword matching.

    Score = number of question keywords
    found in the chunk / number of keywords.
    """

    words = important_words(
        question
    )

    results = []

    for index, chunk in enumerate(
        chunks
    ):

        text = chunk[
            "text"
        ].lower()

        if not words:
            score = 0.0

        else:
            matches = sum(
                1
                for word in words
                if word in text
            )

            score = (
                matches / len(words)
            )

        results.append(
            (
                index,
                score,
            )
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    question,
    chunks,
    index,
    model,
    top_k=5,
):
    """
    Combine:

    75% semantic similarity
    25% keyword matching
    """

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(
        "float32"
    )

    search_count = min(
        len(chunks),
        max(top_k * 3, 10),
    )

    semantic_scores, semantic_indices = (
        index.search(
            query_embedding,
            search_count,
        )
    )

    semantic = {}

    for score, chunk_index in zip(
        semantic_scores[0],
        semantic_indices[0],
    ):

        if chunk_index >= 0:
            semantic[
                int(chunk_index)
            ] = float(score)

    keyword_results = keyword_search(
        question,
        chunks,
    )

    keyword = dict(
        keyword_results
    )

    keyword_candidates = {
        index
        for index, score in sorted(
            keyword.items(),
            key=lambda item: item[1],
            reverse=True,
        )[
            : max(top_k * 3, 10)
        ]
    }

    candidates = (
        set(semantic)
        | keyword_candidates
    )

    ranked = []

    for chunk_index in candidates:

        semantic_score = semantic.get(
            chunk_index,
            0.0,
        )

        keyword_score = keyword.get(
            chunk_index,
            0.0,
        )

        hybrid_score = (
            0.75 * semantic_score
            + 0.25 * keyword_score
        )

        ranked.append(
            {
                "index": chunk_index,
                "score": hybrid_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "chunk": chunks[
                    chunk_index
                ],
            }
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked[:top_k]


# ============================================================
# GROQ
# ============================================================

def answer_question(
    question,
    retrieved_chunks,
    client,
    model_name,
):
    """
    Send the question and retrieved chunks
    to Groq.

    The model is instructed to use ONLY the
    provided context.
    """

    context_parts = []

    for number, result in enumerate(
        retrieved_chunks,
        start=1,
    ):

        chunk = result["chunk"]

        if chunk["page"]:
            location = (
                f"{chunk['filename']}, "
                f"page {chunk['page']}"
            )

        else:
            location = chunk[
                "filename"
            ]

        context_parts.append(
            f"[Source {number}: {location}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question ONLY using the
provided document context.

Do not use outside knowledge.

Do not invent or assume information.

If the answer cannot be found in the provided
context, say exactly:

"I couldn't find that information in the provided documents."

Be concise and factual.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


QUESTION:

{question}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[
        0
    ].message.content


# ============================================================
# FILE SIGNATURE
# ============================================================

def file_signature(
    filename,
    file_bytes,
):
    """
    Create a unique signature for a document.

    If the same file is uploaded again,
    it does not need to be processed again.
    """

    digest = hashlib.sha256(
        file_bytes
    ).hexdigest()

    return (
        f"{filename}:{digest}"
    )


# ============================================================
# LOCAL FILE LOADING
# ============================================================

def load_local_files(
    uploaded_files,
):
    """
    Convert Streamlit uploaded files
    into (filename, bytes) pairs.
    """

    loaded = []

    for uploaded_file in uploaded_files:

        file_bytes = (
            uploaded_file.getvalue()
        )

        loaded.append(
            (
                uploaded_file.name,
                file_bytes,
            )
        )

    return loaded


# ============================================================
# GOOGLE DRIVE
# ============================================================

def get_drive_folder_id(
    drive_link,
):
    """
    Extract a Google Drive folder ID.

    Expected format:

    https://drive.google.com/drive/folders/FOLDER_ID
    """

    match = re.search(
        r"/folders/([a-zA-Z0-9_-]+)",
        drive_link,
    )

    if match:
        return match.group(1)

    return None


def get_drive_file_id(
    drive_link,
):
    """
    Extract a Google Drive file ID from
    common Google Drive URL formats.
    """

    # Example:
    # /file/d/FILE_ID/view
    match = re.search(
        r"/file/d/([a-zA-Z0-9_-]+)",
        drive_link,
    )

    if match:
        return match.group(1)

    # Example:
    # ?id=FILE_ID
    match = re.search(
        r"[?&]id=([a-zA-Z0-9_-]+)",
        drive_link,
    )

    if match:
        return match.group(1)

    return None


def load_drive_files(
    drive_link,
):
    """
    Download supported files from a public
    Google Drive folder or single file.

    Supported:

    PDF
    DOCX
    TXT
    MD
    """

    drive_link = drive_link.strip()

    if not drive_link:
        return []

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="ai_doc_assistant_"
        )
    )

    folder_id = get_drive_folder_id(
        drive_link
    )

    file_id = get_drive_file_id(
        drive_link
    )

    try:

        # ----------------------------------------------------
        # GOOGLE DRIVE FOLDER
        # ----------------------------------------------------

        if folder_id:

            folder_url = (
                "https://drive.google.com/"
                f"drive/folders/{folder_id}"
            )

            st.info(
                "Downloading files from Google Drive folder..."
            )

            downloaded_files = (
                gdown.download_folder(
                    url=folder_url,
                    output=str(temp_dir),
                    quiet=True,
                    use_cookies=False,
                )
            )

            if downloaded_files is None:
                downloaded_files = []

        # ----------------------------------------------------
        # GOOGLE DRIVE SINGLE FILE
        # ----------------------------------------------------

        elif file_id:

            st.info(
                "Downloading Google Drive file..."
            )

            output_file = (
                temp_dir
                / "drive_file"
            )

            downloaded_file = gdown.download(
                url=drive_link,
                output=str(output_file),
                quiet=True,
                fuzzy=True,
            )

            if not downloaded_file:
                raise ValueError(
                    "Google Drive file could not be downloaded."
                )

        else:

            raise ValueError(
                "Invalid Google Drive link. "
                "Use a folder link such as "
                "https://drive.google.com/drive/folders/FOLDER_ID "
                "or a file link such as "
                "https://drive.google.com/file/d/FILE_ID/view"
            )

        # ----------------------------------------------------
        # FIND SUPPORTED FILES
        # ----------------------------------------------------

        files = []

        for path in temp_dir.rglob("*"):

            if not path.is_file():
                continue

            extension = (
                path.suffix.lower()
            )

            if extension not in SUPPORTED_EXTENSIONS:
                continue

            files.append(
                (
                    path.name,
                    path.read_bytes(),
                )
            )

        if not files:

            raise ValueError(
                "No supported PDF, DOCX, TXT or MD files "
                "were found in the Google Drive location."
            )

        return files

    finally:

        # The temporary directory can remain during
        # the Streamlit session. It is not part of the
        # document processing state.
        pass


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:

    st.session_state.documents = {}


if "chunks" not in st.session_state:

    st.session_state.chunks = []


if "embeddings" not in st.session_state:

    st.session_state.embeddings = None


if "faiss_index" not in st.session_state:

    st.session_state.faiss_index = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

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

    st.divider()

    if st.button(
        "🗑️ Clear processed documents"
    ):

        st.session_state.documents = {}

        st.session_state.chunks = []

        st.session_state.embeddings = None

        st.session_state.faiss_index = None

        st.rerun()


# ============================================================
# ADD DOCUMENTS
# ============================================================

st.subheader(
    "1. Add documents"
)

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=[
        "pdf",
        "docx",
        "txt",
        "md",
    ],
    accept_multiple_files=True,
)

drive_link = st.text_input(
    "Or paste a public Google Drive file/folder link",
    placeholder=(
        "https://drive.google.com/drive/folders/..."
    ),
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if st.button(
    "📥 Process documents",
    type="primary",
):

    source_files = []

    # --------------------------------------------------------
    # LOCAL FILES
    # --------------------------------------------------------

    if uploaded_files:

        source_files.extend(
            load_local_files(
                uploaded_files
            )
        )

    # --------------------------------------------------------
    # GOOGLE DRIVE
    # --------------------------------------------------------

    if drive_link.strip():

        try:

            drive_files = (
                load_drive_files(
                    drive_link
                )
            )

            source_files.extend(
                drive_files
            )

        except Exception as error:

            st.error(
                f"Google Drive loading failed: {error}"
            )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not source_files:

        st.warning(
            "Add at least one local document "
            "or a valid public Google Drive link."
        )

    else:

        model = load_embedding_model()

        new_documents = 0

        progress = st.progress(
            0
        )

        total_files = len(
            source_files
        )

        # ----------------------------------------------------
        # PROCESS EACH DOCUMENT
        # ----------------------------------------------------

        for position, (
            filename,
            file_bytes,
        ) in enumerate(
            source_files,
            start=1,
        ):

            signature = file_signature(
                filename,
                file_bytes,
            )

            # ------------------------------------------------
            # SKIP DUPLICATES
            # ------------------------------------------------

            if (
                signature
                in st.session_state.documents
            ):

                progress.progress(
                    position / total_files
                )

                continue

            try:

                # --------------------------------------------
                # EXTRACTION
                # --------------------------------------------

                pages = extract_document(
                    file_bytes,
                    filename,
                )

                # --------------------------------------------
                # CHUNKING
                # --------------------------------------------

                chunks = chunk_text(
                    pages,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                st.session_state.documents[
                    signature
                ] = {
                    "filename": filename,
                    "pages": pages,
                    "chunks": chunks,
                }

                new_documents += 1

            except Exception as error:

                st.error(
                    f"Could not process "
                    f"{filename}: {error}"
                )

            progress.progress(
                position / total_files
            )

        # ----------------------------------------------------
        # BUILD SEARCH INDEX
        # ----------------------------------------------------

        if new_documents:

            all_chunks = []

            for document in (
                st.session_state.documents.values()
            ):

                all_chunks.extend(
                    document["chunks"]
                )

            if all_chunks:

                with st.spinner(
                    "Creating embeddings and FAISS index..."
                ):

                    embeddings = (
                        build_embeddings(
                            all_chunks,
                            model,
                        )
                    )

                    faiss_index = (
                        build_faiss_index(
                            embeddings
                        )
                    )

                st.session_state.chunks = (
                    all_chunks
                )

                st.session_state.embeddings = (
                    embeddings
                )

                st.session_state.faiss_index = (
                    faiss_index
                )

        st.success(
            f"Processed {new_documents} new "
            f"document(s). "
            f"Total chunks: "
            f"{len(st.session_state.chunks)}"
        )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.documents:

    st.subheader(
        "2. Document information"
    )

    for document in (
        st.session_state.documents.values()
    ):

        filename = document[
            "filename"
        ]

        chunks = document[
            "chunks"
        ]

        with st.expander(
            f"📄 {filename} — "
            f"{len(chunks)} chunks"
        ):

            extracted_text = (
                "\n\n".join(
                    record["text"]
                    for record in document[
                        "pages"
                    ]
                    if record[
                        "text"
                    ].strip()
                )
            )

            st.write(
                f"**Filename:** {filename}"
            )

            st.write(
                f"**Chunks:** {len(chunks)}"
            )

            st.text_area(
                "Extracted text",
                extracted_text,
                height=220,
                key=(
                    "text_"
                    + hashlib.md5(
                        filename.encode()
                    ).hexdigest()
                ),
            )


# ============================================================
# QUESTION ANSWERING
# ============================================================

st.subheader(
    "3. Ask a question"
)

question = st.text_input(
    "Question",
    placeholder=(
        "Ask something about your documents..."
    ),
)


if st.button(
    "🔎 Ask"
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        not st.session_state.chunks
        or st.session_state.faiss_index
        is None
    ):

        st.warning(
            "Process at least one document first."
        )

    elif not question.strip():

        st.warning(
            "Enter a question."
        )

    elif "GROQ_API_KEY" not in st.secrets:

        st.error(
            "GROQ_API_KEY is missing from "
            "Streamlit secrets."
        )

    else:

        model = load_embedding_model()

        # ----------------------------------------------------
        # HYBRID SEARCH
        # ----------------------------------------------------

        with st.spinner(
            "Searching documents..."
        ):

            retrieved = hybrid_search(
                question,
                st.session_state.chunks,
                st.session_state.faiss_index,
                model,
                top_k=top_k,
            )

        try:

            # ------------------------------------------------
            # GROQ
            # ------------------------------------------------

            client = get_groq_client(
                st.secrets[
                    "GROQ_API_KEY"
                ]
            )

            with st.spinner(
                "Generating answer..."
            ):

                answer = answer_question(
                    question,
                    retrieved,
                    client,
                    groq_model,
                )

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.markdown(
                "### 💬 Answer"
            )

            st.write(
                answer
            )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.markdown(
                "### 📚 Retrieved sources"
            )

            for number, result in enumerate(
                retrieved,
                start=1,
            ):

                chunk = result[
                    "chunk"
                ]

                if chunk["page"]:

                    page_text = (
                        f"Page {chunk['page']}"
                    )

                else:

                    page_text = (
                        "Page not available"
                    )

                with st.expander(
                    f"{number}. "
                    f"{chunk['filename']} — "
                    f"{page_text} "
                    f"(hybrid score: "
                    f"{result['score']:.3f})"
                ):

                    st.write(
                        chunk["text"]
                    )

                    st.caption(
                        f"Semantic score: "
                        f"{result['semantic_score']:.3f} | "
                        f"Keyword score: "
                        f"{result['keyword_score']:.3f}"
                    )

        except Exception as error:

            st.error(
                f"Groq request failed: {error}"
            )
```
