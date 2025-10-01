import openai
import PyPDF2
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import faiss  # Required for FAISS integration in Langchain
from dotenv import load_dotenv  # Import the dotenv library
import os  # To access environment variables

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to clean the extracted text
def clean_text(text):
    # Remove multiple spaces and replace with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing spaces
    text = text.strip()
    # Remove any unwanted characters like page numbers or footers
    text = re.sub(r'\b(Page \d+|[0-9]+)\b', '', text)
    return text

# Function to extract and clean text from a PDF file
def extract_and_clean_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        
        # Loop through all pages and extract text
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        
        # Clean the extracted text
        cleaned_text = clean_text(text)
        
        return cleaned_text

# Extract and clean text from the PDF
pdf_path = 'example.pdf'  # Replace with the path to your PDF
cleaned_pdf_text = extract_and_clean_pdf(pdf_path)

# Split the cleaned text into chunks (assuming each chunk is a document)
# For simplicity, split by sentences or paragraphs
documents = cleaned_pdf_text.split('\n')  # Adjust to better split by paragraphs if needed

# Initialize OpenAI embeddings
embedding = OpenAIEmbeddings()

# Generate embeddings for the documents (without numpy)
doc_embeddings = [embedding.embed_text(doc) for doc in documents]

# Create FAISS index (IndexFlatL2 is the simplest FAISS index)
dimension = len(doc_embeddings[0])  # Length of the embedding vector
faiss_index = faiss.IndexFlatL2(dimension)

# Convert list of embeddings to a list of lists (FAISS accepts list of lists for adding vectors)
embedding_list = [list(embed) for embed in doc_embeddings]  # No numpy used here
faiss_index.add(np.array(embedding_list).astype('float32'))  # FAISS still needs float32, but we keep it in list form

# Create FAISS vector store
vectorstore = FAISS(faiss_index)

# Initialize OpenAI LLM (Large Language Model)
llm = OpenAI(model="text-davinci-003", temperature=0.7)

# Define the prompt template
template = "Given the following context, answer the user's question:\n\n{context}\n\nUser: {question}\nAnswer:"
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    verbose=True
)

# Example query
user_query = "What is Langchain?"

# Run the RAG system to get the response
response = qa_chain.run(user_query)

# Print the response
print("Response:", response)
