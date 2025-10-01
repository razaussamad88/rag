import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv  # Import the dotenv library
import os  # To access environment variables

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your documents (you can load from files, databases, etc.)
# For simplicity, using a small set of example texts
documents = [
    "OpenAI is an AI research lab focused on creating artificial general intelligence (AGI).",
    "Langchain is a framework for developing applications using large language models.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Retrieval-augmented generation (RAG) combines document retrieval with generative models to improve answers."
]

# Initialize OpenAI embeddings (you can use other models like Ada, Davinci, etc.)
embedding = OpenAIEmbeddings()

# Convert the documents to embeddings (Ensure correct formatting for FAISS)
# doc_embeddings = [embedding.embed_text(doc) for doc in documents]
# Embed documents (list of strings)
doc_embeddings = embedding.embed_documents(documents)  # instead of embed_text in a loop

# FAISS expects embeddings to be in numpy array form, but let's try to avoid numpy entirely
# Create the FAISS index (we will use the IndexFlatL2 index type)
# Using langchain's internal method for embeddings and index creation

# Create the FAISS vector store using the embeddings
vectorstore = FAISS.from_documents(documents, embedding)

# Initialize OpenAI LLM (Large Language Model) - GPT-3 model
llm = OpenAI(model="text-davinci-003")  # Removed temperature

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
user_query = "What is Langchain?"

# Run the RAG system to get the response
response = qa_chain.run(user_query)

# Output the generated answer
print("Response:", response)
