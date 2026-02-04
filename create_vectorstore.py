# create_vectorstore.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_vector_store_from_directory(data_dir="data", store_path="vectorstore"):
    """
    Reads all .txt files from a directory, splits them into chunks,
    creates embeddings, and saves them to a local FAISS vector store.
    """
    all_chunks = []

    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' was not found.")
        print("Please create a 'data' directory and place your .txt files inside it.")
        return

    # Get all .txt files from the data directory
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not txt_files:
        print(f"No .txt files found in the '{data_dir}' directory.")
        return

    print(f"Found {len(txt_files)} files to process: {', '.join(txt_files)}")

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    for txt_file in txt_files:
        file_path = os.path.join(data_dir, txt_file)
        print(f"Reading context from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
        print(f"Split '{txt_file}' into {len(chunks)} chunks.")

    if not all_chunks:
        print("No text was extracted from the files. Vector store not created.")
        return

    print(f"\nTotal chunks to process: {len(all_chunks)}")

    # Load the embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the FAISS vector store from all chunks
    print("Creating FAISS vector store...")
    db = FAISS.from_texts(all_chunks, embeddings)

    # Save the vector store locally
    db.save_local(store_path)
    print(f"✅ Vector store created and saved locally at '{store_path}'.")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    create_vector_store_from_directory()
