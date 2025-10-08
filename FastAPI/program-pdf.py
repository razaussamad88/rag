import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="PDF RAG API")

# Globals (to store embeddings & vector DB after upload)
vectorstore = None
qa_chain = None

# Embeddings & LLM setup (static)
# Local embeddings using Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Local LLM using Ollama (must be installed and running a model like mistral)
llm = OllamaLLM(model="mistral")
# Define a prompt template for generating responses
template = "Given the following context, answer the user's question:\n\n{context}\n\nUser: {question}\nAnswer:"
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# ---------------------------
# API Models
# ---------------------------
class QueryRequest(BaseModel):
    question: str

# ---------------------------
# Endpoints
# ---------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, qa_chain

    # Save uploaded file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load and split PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.split_documents(pages)

	# Create the FAISS vector store using the embeddings
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Create QA chain
# Set up the retrieval-based QA system using Langchain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    # Cleanup
    os.remove(file_path)

    return {"status": "PDF uploaded and processed successfully."}


@app.post("/ask")
async def ask_question(request: QueryRequest):
    global qa_chain
    if qa_chain is None:
        return {"error": "No PDF uploaded yet. Please upload a PDF first."}

    response = qa_chain.invoke({"query": request.question})
    return {"question": request.question, "answer": response["result"]}
