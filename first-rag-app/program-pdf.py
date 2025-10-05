import sys
import os
from datetime import datetime
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("\r\n\r\n### RAG System Begins...")
print(f"  * Execution directory: {os.getcwd()}")

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"  * Current script directory:", current_dir)

pdf_path = f"{current_dir}/resume.pdf"
print(f"  * PDF Document Path: '{pdf_path}'")

query_file_path = f"{current_dir}/query.txt"
print(f"  * Query File Path: '{query_file_path}'")

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'



print(f"[{datetime.now().strftime(DATETIME_FORMAT)}] Query File Loading...")
with open(query_file_path, "r", encoding="utf-8") as file:
    query_content = file.read()
print(f"  * Query: '{query_content}'")

print(f"[{datetime.now().strftime(DATETIME_FORMAT)}] PDF Loading...")
# Load your documents from a PDF and chunk for retrieval
loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"[{datetime.now().strftime(DATETIME_FORMAT)}] File Splitting...")
# Split long pages into overlapping chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
documents = text_splitter.split_documents(pages)


## Convert the documents to embeddings (Ensure correct formatting for FAISS) ##

print(f"[{datetime.now().strftime(DATETIME_FORMAT)}] Embedding...")
# Local embeddings using Hugging Face model
doc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print(f"[{datetime.now().strftime(DATETIME_FORMAT)}] Creating in-memory Vector Store...")
# Create the FAISS vector store using the embeddings
vectorstore = FAISS.from_documents(documents, doc_embeddings)

# Local LLM using Ollama (must be installed and running a model like mistral)
llm = OllamaLLM(model="mistral")

# Define a prompt template for generating responses
template = "Given the following context, answer the user's question:\n\n{context}\n\nUser: {question}\nAnswer:"
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

print(f"[{datetime.now().strftime(DATETIME_FORMAT)}] Setting up retrieval-based QA system...")
# Set up the retrieval-based QA system using Langchain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=vectorstore.as_retriever(), 
    chain_type="stuff", 
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)

# Example user query
# user_query = "What is the person's name? What are his skill set in detail? Does this candidate a good option for managerial role or for software arhitect position?"
user_query = query_content

# if needed for dev, Successful exit before invoking RAG
#sys.exit(0)

print(f"[{datetime.now().strftime(DATETIME_FORMAT)}] Invoking the RAG system...")
# Run the RAG system to get the response
response = qa_chain.invoke({"query": user_query})

# Output the generated answer
print(f"[{datetime.now().strftime(DATETIME_FORMAT)}] Response:", response)
print("\r\n\r\n\r\n### RAG System Finished!")