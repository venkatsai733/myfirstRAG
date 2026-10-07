# Local RAG Flask Website

This is the local Flask version of `RAG_Google_Drive_File_Query_Simplified_Flow.ipynb`. It supports TXT, PDF, and DOCX documents, multi-file uploads, incremental additions, local Chroma indexing, and question answering with the same embedding and FLAN-T5 models.

## Run on Windows with Anaconda

Open Anaconda Prompt in this folder:

```powershell
conda activate local-rag
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

The first upload downloads the two Hugging Face models. After that, they are cached locally. Uploaded originals are stored in `uploads/`; the local vector database is stored in `file_rag_database/`.

## Create the environment from scratch

```powershell
conda create -n local-rag python=3.11 -y
conda activate local-rag
python -m pip install -r requirements.txt
python app.py
```

Python 3.11 is used because it has broad support across PyTorch, ChromaDB, Transformers, and SentenceTransformers on Windows. No Google Drive or cloud API is required.

## Supported files

- `.txt` — UTF-8 text
- `.pdf` — text-based PDFs; scanned PDFs need OCR first
- `.docx` — paragraphs and table content

Maximum upload size is 50 MB per request. Files remain on the local machine.
