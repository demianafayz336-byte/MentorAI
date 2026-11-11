import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredExcelLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="MentorAI")

print("Loading Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("course_materials"):
    loader = DirectoryLoader("course_materials", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
else:
    documents = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

if documents:
    chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
else:
    vectorstore = None

llm = None

def load_llm():
    global llm
    if llm is None:
        print("Loading LLM Model (this may take a while)...")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
        llm = HuggingFacePipeline(pipeline=pipe)
    return llm

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore
    os.makedirs("uploaded_files", exist_ok=True)
    file_path = os.path.join("uploaded_files", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if file.filename.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
        loader = UnstructuredExcelLoader(file_path)
    else:
        return {"message": "Unsupported file type. Please upload .txt, .pdf, or .excel."}
    
    new_documents = loader.load()
    new_chunks = text_splitter.split_documents(new_documents)
    
    if vectorstore is None:
        vectorstore = FAISS.from_documents(new_chunks, embeddings)
    else:
        vectorstore.add_documents(new_chunks)
    
    return {"message": f"File {file.filename} uploaded successfully!"}

@app.post("/ask")
def ask(request: QueryRequest):
    if vectorstore is None:
        return {"answer": "No documents uploaded yet.", "sources": []}
    
    current_llm = load_llm()
        
    docs = vectorstore.similarity_search(request.question, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Use the following context to answer the question. 
If you don't know the answer, say you don't know.
Keep the answer concise.

Context:
{context}

Question: {request.question}
Helpful Answer:"""
    
    response = current_llm.invoke(prompt)
    if "Helpful Answer:" in response:
        response = response.split("Helpful Answer:")[-1].strip()
    
    return {"answer": response, "sources": list(set([doc.metadata['source'] for doc in docs]))}

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
                <input type="file" id="fileInput" accept=".txt, .pdf, .xlsx, .xls" onchange="uploadFile()">
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
                chatBox.innerHTML += `<div class="message bot-msg">${data.message}</div>`;
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
                
                let sourceHtml = data.sources.length > 0 ? `<div class="source">Source: ${data.sources.join(', ')}</div>` : '';
                chatBox.innerHTML += `<div class="message bot-msg">${data.answer} ${sourceHtml}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (error) {
                chatBox.removeChild(loadingMsg);
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

port = int(os.environ.get("PORT", 3000))
config = uvicorn.Config(app, host="0.0.0.0", port=port)
server = uvicorn.Server(config)

print(f"✅ MentorAI is running on port {port}!")
loop = asyncio.get_event_loop()
loop.create_task(server.serve())
