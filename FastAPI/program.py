from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# ---------------------------
# Setup RAG Pipeline
# ---------------------------

# Example documents
documents = [
    Document(page_content="OpenAI is an AI research lab focused on creating artificial general intelligence (AGI)."),
    Document(page_content="Langchain is a framework for developing applications using large language models."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="Retrieval-augmented generation (RAG) combines document retrieval with generative models to improve answers.")
]

# Embeddings
doc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# FAISS Vector DB
vectorstore = FAISS.from_documents(documents, doc_embeddings)

# Local LLM (Ollama)
llm = OllamaLLM(model="mistral")

# Prompt
template = "Given the following context, answer the user's question:\n\n{context}\n\nUser: {question}\nAnswer:"
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    verbose=True
)

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(title="RAG API with FastAPI")

# Request model
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    response = qa_chain.invoke(request.question)
    return {"question": request.question, "answer": response["result"]}
