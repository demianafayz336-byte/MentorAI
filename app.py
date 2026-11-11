
import os
import shutil
import csv
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="MentorAI")

# ==========================================
# Global variables
# ==========================================

embeddings = None
vectorstore = None
llm = None
text_splitter = None

UPLOAD_DIR = Path("uploaded_files")
COURSE_DIR = Path("course_materials")

UPLOAD_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".txt", ".pdf", ".docx", ".xlsx",
    ".xls", ".pptx", ".csv"
}


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():
    return {"status": "ok", "application": "MentorAI"}


# ==========================================
# Text Splitter
# ==========================================

def get_text_splitter():
    global text_splitter

    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    return text_splitter


# ==========================================
# Load Embeddings
# ==========================================

def get_embeddings():
    global embeddings

    if embeddings is None:
        print("Loading Embeddings...")

        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("Embeddings loaded successfully!")

    return embeddings


# ==========================================
# Extract text from all supported files
# ==========================================

def extract_text(file_path: Path):

    ext = file_path.suffix.lower()
    documents = []

    # TXT
    if ext == ".txt":

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="latin-1")

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name}
                )
            )

    # PDF
    elif ext == ".pdf":

        from pypdf import PdfReader

        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page": page_number
                        }
                    )
                )

    # WORD
    elif ext == ".docx":

        from docx import Document as WordDocument

        doc = WordDocument(str(file_path))

        parts = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                parts.append(paragraph.text)

        for table in doc.tables:
            for row in table.rows:
                parts.append(
                    " | ".join(cell.text for cell in row.cells)
                )

        text = "\n".join(parts)

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name}
                )
            )

    # POWERPOINT
    elif ext == ".pptx":

        from pptx import Presentation

        presentation = Presentation(str(file_path))

        for slide_number, slide in enumerate(
            presentation.slides, start=1
        ):

            parts = []

            for shape in slide.shapes:

                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text)

                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        parts.append(
                            " | ".join(
                                cell.text for cell in row.cells
                            )
                        )

            text = "\n".join(parts)

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "slide": slide_number
                        }
                    )
                )

    # EXCEL XLSX
    elif ext == ".xlsx":

        from openpyxl import load_workbook

        workbook = load_workbook(
            filename=str(file_path),
            read_only=True,
            data_only=True
        )

        try:
            for sheet in workbook.worksheets:

                rows = []

                for row in sheet.iter_rows(values_only=True):

                    values = [
                        str(value)
                        for value in row
                        if value is not None
                    ]

                    if values:
                        rows.append(" | ".join(values))

                text = "\n".join(rows)

                if text.strip():
                    documents.append(
                        Document(
                            page_content=f"Sheet: {sheet.title}\n{text}",
                            metadata={
                                "source": file_path.name,
                                "sheet": sheet.title
                            }
                        )
                    )
        finally:
            workbook.close()

    # EXCEL XLS
    elif ext == ".xls":

        import xlrd

        workbook = xlrd.open_workbook(str(file_path))

        for sheet in workbook.sheets():

            rows = []

            for row_index in range(sheet.nrows):

                values = [
                    str(value)
                    for value in sheet.row_values(row_index)
                    if value not in ("", None)
                ]

                if values:
                    rows.append(" | ".join(values))

            text = "\n".join(rows)

            if text.strip():
                documents.append(
                    Document(
                        page_content=f"Sheet: {sheet.name}\n{text}",
                        metadata={
                            "source": file_path.name,
                            "sheet": sheet.name
                        }
                    )
                )

    # CSV
    elif ext == ".csv":

        with file_path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as csv_file:

            rows = []

            for row in csv.reader(csv_file):

                values = [
                    value.strip()
                    for value in row
                    if value.strip()
                ]

                if values:
                    rows.append(" | ".join(values))

        text = "\n".join(rows)

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name}
                )
            )

    return documents


# ==========================================
# Add documents to FAISS
# ==========================================

def add_documents_to_index(documents):

    global vectorstore

    if not documents:
        raise ValueError(
            "No readable text was found in this file."
        )

    chunks = get_text_splitter().split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be created.")

    from langchain_community.vectorstores import FAISS

    current_embeddings = get_embeddings()

    if vectorstore is None:

        vectorstore = FAISS.from_documents(
            chunks,
            current_embeddings
        )

    else:

        vectorstore.add_documents(chunks)


# ==========================================
# Load course materials
# ==========================================

def load_course_materials():

    if not COURSE_DIR.exists():
        return

    for file_path in COURSE_DIR.iterdir():

        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:

            try:
                documents = extract_text(file_path)

                if documents:
                    add_documents_to_index(documents)

            except Exception as error:
                print(
                    f"Error loading {file_path.name}: {error}"
                )


# ==========================================
# Load Qwen LLM
# ==========================================

def load_llm():

    global llm

    if llm is None:

        print("Loading Qwen model...")

        import torch

        from transformers import (
            pipeline,
            AutoModelForCausalLM,
            AutoTokenizer
        )

        from langchain_huggingface import HuggingFacePipeline

        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model.eval()

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=False,
            return_full_text=False
        )

        llm = HuggingFacePipeline(
            pipeline=text_pipeline
        )

        print("Qwen model loaded successfully!")

    return llm


# ==========================================
# Request Model
# ==========================================

class QueryRequest(BaseModel):
    question: str


# ==========================================
# Upload File
# ==========================================

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):

    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported: TXT, PDF, DOCX, XLSX, XLS, PPTX, CSV."
            )
        )

    file_path = UPLOAD_DIR / filename

    try:

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        documents = extract_text(file_path)

        add_documents_to_index(documents)

        return {
            "message": f"File {filename} uploaded successfully!"
        }

    except Exception as error:

        print(f"Upload error: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(error)}"
        )

    finally:
        file.file.close()


# ==========================================
# Ask Questions
# ==========================================

@app.post("/ask")
def ask(request: QueryRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    global vectorstore

    try:

        if vectorstore is None:
            load_course_materials()

        if vectorstore is None:

            return {
                "answer": "No documents uploaded yet.",
                "sources": []
            }

        docs = vectorstore.similarity_search(
            question,
            k=2
        )

        if not docs:

            return {
                "answer": "I couldn't find relevant information.",
                "sources": []
            }

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        prompt = f"""Use the following context to answer the question.
If you don't know the answer, say you don't know.
Keep the answer concise.

Context:
{context}

Question: {question}

Helpful Answer:"""

        current_llm = load_llm()

        response = current_llm.invoke(prompt)

        answer = str(response).strip()

        if "Helpful Answer:" in answer:
            answer = answer.split(
                "Helpful Answer:"
            )[-1].strip()

        sources = sorted({
            doc.metadata.get("source", "Unknown")
            for doc in docs
        })

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as error:

        print(f"Question error: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(error)}"
        )


# ==========================================
# Original MentorAI Interface
# Design preserved
# ==========================================

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MentorAI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: #1e293b; color: #e2e8f0; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
        .chat-container { background: #0f172a; border-radius: 24px; width: 100%; max-width: 800px; display: flex; flex-direction: column; height: 85vh; box-shadow: 0 20px 50px rgba(0,0,0,0.5); overflow: hidden; }
        .header { padding: 20px; text-align: center; border-bottom: 1px solid #334155; background: #0f172a; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 24px; background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .upload-btn { background: #334155; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 14px; }
        .upload-btn:hover { background: #475569; }
        .chat-box { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 80%; padding: 15px 20px; border-radius: 20px; font-size: 15px; line-height: 1.5; }
        .user-msg { align-self: flex-end; background: linear-gradient(135deg, #38bdf8, #818cf8); color: white; border-bottom-right-radius: 5px; }
        .bot-msg { align-self: flex-start; background: #1e293b; color: #e2e8f0; border-bottom-left-radius: 5px; border: 1px solid #334155; }
        .source { font-size: 12px; color: #94a3b8; margin-top: 10px; border-top: 1px solid #334155; padding-top: 8px; }
        .input-area { padding: 20px; background: #0f172a; border-top: 1px solid #334155; display: flex; gap: 10px; }
        input[type="text"] { flex: 1; height: 50px; background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 0 20px; color: white; font-size: 16px; outline: none; }
        input[type="text"]:focus { border-color: #38bdf8; }
        button { background: linear-gradient(135deg, #38bdf8, #818cf8); border: none; border-radius: 12px; padding: 0 25px; color: white; font-weight: 600; cursor: pointer; }
        #fileInput { display: none; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <h1>🧠 MentorAI</h1>
            <label class="upload-btn">
                📂 Upload File
                <input type="file" id="fileInput"
                    accept=".txt,.pdf,.docx,.xlsx,.xls,.pptx,.csv"
                    onchange="uploadFile()">
            </label>
        </div>
        <div class="chat-box" id="chatBox">
            <div class="message bot-msg">Hello! Upload a file and ask me anything about it.</div>
        </div>
        <div class="input-area">
            <input type="text" id="questionInput" placeholder="Type your question..." onkeypress="if(event.key === 'Enter') askQuestion()">
            <button onclick="askQuestion()">Send</button>
        </div>
    </div>

    <script>
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            if (!file) return;

            const chatBox = document.getElementById('chatBox');
            chatBox.innerHTML += `<div class="message user-msg">📄 Uploading: ${file.name}</div>`;

            const formData = new FormData();
            formData.append("file", file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                chatBox.innerHTML += `<div class="message bot-msg">${data.message || data.detail}</div>`;

            } catch (error) {
                chatBox.innerHTML += `<div class="message bot-msg">Error uploading file.</div>`;
            }
        }

        async function askQuestion() {
            const input = document.getElementById('questionInput');
            const question = input.value.trim();
            const chatBox = document.getElementById('chatBox');

            if (!question) return;

            chatBox.innerHTML += `<div class="message user-msg">${question}</div>`;
            input.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingMsg = document.createElement('div');
            loadingMsg.className = 'message bot-msg';
            loadingMsg.innerText = "Thinking...";
            chatBox.appendChild(loadingMsg);
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                const data = await response.json();

                chatBox.removeChild(loadingMsg);

                let sourceHtml = data.sources && data.sources.length > 0
                    ? `<div class="source">Source: ${data.sources.join(', ')}</div>`
                    : '';

                chatBox.innerHTML += `<div class="message bot-msg">${data.answer || data.detail || 'An error occurred.'} ${sourceHtml}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;

            } catch (error) {

                if (loadingMsg.parentNode) {
                    chatBox.removeChild(loadingMsg);
                }

                chatBox.innerHTML += `<div class="message bot-msg">Error connecting to server.</div>`;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return html_content
