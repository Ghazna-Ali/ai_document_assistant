# AI Document Assistant

A simple Streamlit RAG-style document assistant for:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Public Google Drive files/folders

The app extracts document text, creates overlapping chunks, generates Sentence Transformers embeddings, stores them in FAISS, and uses hybrid semantic + keyword search before asking Groq to answer from the retrieved context.

## Architecture

```text
Local Upload / Google Drive
          |
          v
   Document Extraction
          |
          v
       Chunking
          |
          v
 Sentence Transformers
      Embeddings
          |
          v
       FAISS Index
          |
          |
User Question
     |       |
     v       v
 Semantic   Keyword
  Search    Search
     \       /
      \     /
       Hybrid Ranking
            |
            v
    Top Relevant Chunks
            |
            v
       Groq LLM
            |
            v
      Answer + Sources
```

## 1. Install

Create and activate a virtual environment if desired.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Configure Groq

Create this file:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Do not commit this file to GitHub.

A suitable `.gitignore` entry is:

```text
.streamlit/secrets.toml
```

The app never hardcodes the Groq key in `app.py`.

## 3. Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 4. How processing works

### PDF

PDF pages are extracted separately. Each page keeps:

- filename
- page number
- extracted text

### DOCX

Paragraphs are extracted from the Word document.

DOCX files do not provide reliable page numbers through this simple extraction approach, so their page value is `None`.

### TXT and MD

The complete text is extracted as one logical document section. These formats do not have native page numbers, so their page value is `None`.

### Chunking

Text is split into overlapping chunks.

Default:

- chunk size: 900 characters
- overlap: 150 characters

Every chunk keeps:

```text
text
filename
page
```

The UI shows the number of chunks created.

## 5. Embeddings

The app uses:

```text
sentence-transformers
all-MiniLM-L6-v2
```

Document chunks are embedded when new documents are processed.

The embeddings are kept in Streamlit session state, together with the FAISS index.

The app does **not** recreate document embeddings for every question.

The embedding model itself is also cached with `st.cache_resource`.

## 6. FAISS search

FAISS stores normalized chunk embeddings using an inner-product index.

For a question:

1. The question is embedded.
2. FAISS retrieves semantically similar chunks.
3. Keyword search scores chunks containing important question words.
4. The two scores are combined.
5. The highest-ranked chunks are sent to Groq.

The current hybrid score is:

```text
75% semantic similarity
25% keyword matching
```

This is intentionally simple so it is easy to understand and modify.

## 7. Groq

Groq receives:

- the user's question
- the retrieved document chunks

The system prompt instructs the model to answer only from those chunks.

If the information is not present, the model is instructed to say:

> I couldn't find that information in the provided documents.

The default model field is:

```text
openai/gpt-oss-120b
```

You can change it from the Streamlit sidebar if your Groq account uses a different available model.

## 8. Google Drive

Paste a public Google Drive file or folder link.

The app uses `gdown` to download supported files and then sends them through the same pipeline:

```text
Google Drive
    ↓
Download
    ↓
Extract
    ↓
Chunk
    ↓
Embed
    ↓
FAISS
    ↓
Hybrid Search
    ↓
Groq
```

Supported extensions:

```text
.pdf
.docx
.txt
.md
```

For a Drive folder, the folder should be publicly accessible or otherwise downloadable by `gdown`.

## 9. Avoiding repeated processing

Each processed file receives a SHA-256 signature based on:

```text
filename + file contents
```

If the same file is added again during the Streamlit session, the app skips extraction and embedding for that file.

When a genuinely new document is added, the searchable collection is rebuilt so the FAISS index includes the new chunks.

Questions do not recreate document embeddings.

## 10. Important limitation

This is intentionally a simple educational implementation.

The current version stores processed documents, chunks, embeddings, and the FAISS index in Streamlit session state. They persist during the active Streamlit session but are not written to a permanent vector database.

For a production application, the next step would be persistent storage for:

- document IDs/hashes
- chunks
- embeddings
- metadata
- FAISS index

## 11. Project structure

```text
ai-document-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── secrets.toml
```

The implementation intentionally stays in one `app.py` so the complete RAG flow is easy to explain.
