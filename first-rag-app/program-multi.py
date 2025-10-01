from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Load your documents (you can load from files, databases, etc.)
# For simplicity, using a small set of example texts

documents = [
    Document(page_content="OpenAI is an AI research lab focused on creating artificial general intelligence (AGI)."),
    Document(page_content="Langchain is a framework for developing applications using large language models."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="Retrieval-augmented generation (RAG) combines document retrieval with generative models to improve answers.")
]


## Convert the documents to embeddings (Ensure correct formatting for FAISS) ##

# Local embeddings using Hugging Face model
doc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the FAISS vector store using the embeddings
vectorstore = FAISS.from_documents(documents, doc_embeddings)

# Local LLM using Ollama (must be installed and running a model like mistral)
llm = OllamaLLM(model="mistral")

# Define a prompt template for generating responses
template = "Given the following context, answer the user's question:\n\n{context}\n\nUser: {question}\nAnswer:"
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Set up the retrieval-based QA system using Langchain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=vectorstore.as_retriever(), 
    chain_type="stuff", 
    verbose=True
)

# Example user query
# List of user queries
user_queries = [
    "What is Langchain?",
    "What is HuggingFaceEmbeddings?",
    "What is all-MiniLM-L6-v2?",
    "What is OllamaLLM?",
    "What is Mistral?",
    "What is FAISS?",
    "What is RAG?",
    "How does Langchain help with large language models?"
]

# Run the RAG system to get the response
# Loop through and ask each query
for query in user_queries:
    print(f"\nUser Query: {query}")
    response = qa_chain.invoke(query)

# Output the generated answer
    print("Response:", response)
