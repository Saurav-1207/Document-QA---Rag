import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import cohere
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for reading PDFs
import docx  # python-docx for reading Word documents
import pandas as pd  # For reading CSV files
import numpy as np
from pptx import Presentation  # Add this import for PPTX support

# Initialize Pinecone client
pc = Pinecone(api_key="5b4097f7-41fe-4f3e-8485-b021b82f707a")

# Check if the index already exists, if not create it
index_name = "rag-qa-bot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Your model's dimension
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(
            cloud='aws',  
            region='us-east-1'  
        )
    )

# Access the index
index = pc.Index(index_name)

# Initialize Cohere API
co = cohere.Client('DSka7LPCNDbFEPvEu6WtAvwZ1iGeoK8OOQi0dsPC')

# Load Sentence-BERT model for embedding generation
model = SentenceTransformer('all-MiniLM-L12-v2')  # 384 dimensions

# Helper functions for file reading
@st.cache_data
def read_pdf(file):
    pdf_reader = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_reader.page_count):
        page = pdf_reader.load_page(page_num)
        text += page.get_text("text")
    return text

@st.cache_data
def read_docx(file):
    doc = docx.Document(file)
    text = [para.text for para in doc.paragraphs]
    return '\n'.join(text)

@st.cache_data
def read_csv(file):
    df = pd.read_csv(file)
    text = df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ')
    return text

@st.cache_data
def read_pptx(file):
    presentation = Presentation(file)
    text = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return '\n'.join(text)

# Function to generate answer using Cohere
@st.cache_data
def generate_answer(query, relevant_chunks):
    context = ' '.join(relevant_chunks)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        max_tokens=300
    )
    return response.generations[0].text.strip()

# Function to query Pinecone for relevant chunks
@st.cache_data
def answer_question(query, namespace):
    query_embedding = model.encode(query).tolist()
    result = index.query(vector=query_embedding, top_k=5, namespace=namespace, include_metadata=True)

    if result['matches']:
        relevant_chunks = [match['metadata']['content'] for match in result['matches']]
        return relevant_chunks
    else:
        return []

# Function to delete a document from Pinecone
def delete_uploaded_text(namespace):
    index.delete(delete_all=True, namespace=namespace)
    st.session_state['uploaded_files'].remove(namespace)
    st.success(f"Deleted all documents under namespace '{namespace}'.")

st.set_page_config(page_title="Document QA Chatbot", layout="wide")

st.title("📄 Document QA Chatbot")

# Create a CSS style for better visuals
st.markdown(
    """
    <style>
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .user-message {
        text-align: right;
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 80%;
        display: inline-block;
    }
    .bot-message {
        text-align: left;
        background-color: #E6E6E6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 80%;
        display: inline-block;
    }
    .relevant-segment {
        background-color: #f1f1f1;
        border: 1px solid #d3d3d3;
        border-radius: 5px;
        padding: 5px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document")
    
    uploaded_file = st.file_uploader("Choose a PDF, Word, PPTX, or CSV file", type=["pdf", "docx", "pptx", "txt", "csv"])
    
    st.write("Uploaded Files:")
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []
        
    for file in st.session_state['uploaded_files']:
        st.write(file)
    
    if uploaded_file is not None and uploaded_file.name not in st.session_state['uploaded_files']:
        namespace = uploaded_file.name

        if uploaded_file.type == "application/pdf":
            document_text = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document_text = read_docx(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            document_text = read_pptx(uploaded_file)  # Add this line for PPTX
        elif uploaded_file.type == "text/plain":
            document_text = uploaded_file.read().decode('utf-8')
        elif uploaded_file.type == "text/csv":
            document_text = read_csv(uploaded_file)

        st.session_state['uploaded_files'].append(namespace)
        
        document_chunks = [document_text[i:i+500] for i in range(0, len(document_text), 500)]
        
        st.session_state[f'{namespace}_text'] = document_text
        st.session_state[f'{namespace}_chunks'] = document_chunks
        
        batch_size = 10
        for i in range(0, len(document_chunks), batch_size):
            batch = document_chunks[i:i + batch_size]
            vectors = [{
                "id": f'chunk-{j+i}',
                "values": model.encode(chunk).tolist(),
                "metadata": {"content": chunk}
            } for j, chunk in enumerate(batch)]
            index.upsert(vectors=vectors, namespace=namespace)
        st.success(f"Document '{uploaded_file.name}' indexed successfully!")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'relevant_segments' not in st.session_state:
    st.session_state.relevant_segments = []
if 'last_processed_input' not in st.session_state:
    st.session_state.last_processed_input = ""

# Callback function to handle form submission
def handle_submit():
    user_input = st.session_state.user_input
    
    if user_input and user_input != st.session_state.last_processed_input:
        st.session_state.last_processed_input = user_input
        
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        if len(st.session_state['uploaded_files']) > 0:
            current_namespace = st.session_state['uploaded_files'][-1]
            document_chunks = st.session_state[f'{current_namespace}_chunks']
            with st.spinner("Generating answer..."):
                relevant_chunks = answer_question(user_input, current_namespace)

                if relevant_chunks:
                    answer = generate_answer(user_input, relevant_chunks)
                    st.session_state.chat_history.append({"role": "bot", "text": answer})
                    st.session_state.relevant_segments = relevant_chunks
                else:
                    st.session_state.chat_history.append({"role": "bot", "text": "No relevant information found."})
        else:
            st.session_state.chat_history.append({"role": "bot", "text": "No documents uploaded."})

# Chatbot-like input and display
st.subheader("Chat with your document:")

# Display the conversation in a scrollable container
def display_chat():
    with st.container():
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-message'>{message['text']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Display chat history
display_chat()

# Create a form for input
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input")
    submit_button = st.form_submit_button("Send", on_click=handle_submit)

if st.session_state.chat_history:
    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.relevant_segments = []
            st.session_state.last_processed_input = ""    

# Display relevant document segments used in the answer
if len(st.session_state.relevant_segments) > 0:
    st.subheader("Relevant Document Segments:")
    for segment in st.session_state.relevant_segments:
        st.markdown(f"<div class='relevant-segment'>{segment}</div>", unsafe_allow_html=True)
