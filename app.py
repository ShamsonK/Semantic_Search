import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Dataset
csv_file = r"C:\Users\Samson\Desktop\Seamantic Search\myntra_products_catalog.csv"  
df = pd.read_csv(csv_file).loc[:499]  # Limiting rows for efficiency

# Select Relevant Columns (Modify Based on Your Dataset)
text_column = "Description"  # Change this if needed
title_column = "ProductName"  # Change this if needed

# Check if required columns exist
if text_column not in df.columns or title_column not in df.columns:
    st.error(f"Columns '{text_column}' or '{title_column}' not found in dataset.")
    st.stop()

# Fill missing values
df[text_column] = df[text_column].fillna("")
df[title_column] = df[title_column].fillna("Unknown Product")

# Load Pretrained Embedding Model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Convert Product Descriptions to Embeddings
st.info("Processing dataset and creating search index...")
corpus = df[text_column].astype(str).tolist()
corpus_embeddings = model.encode(corpus, convert_to_numpy=True)

# Normalize embeddings for better performance in FAISS (optional but recommended)
faiss.normalize_L2(corpus_embeddings)

# Create FAISS Index
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# Function to Perform Semantic Search
def semantic_search(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)  # Normalize query embedding
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        results.append((df.iloc[idx][title_column], df.iloc[idx][text_column], distances[0][i]))
    
    return results

# Streamlit UI
st.title("🛍️ Myntra Product Search with Semantic Search")

# Search Bar
query_text = st.text_input("🔍 Enter product search query:", "")

# Number of Results Slider
top_k = st.slider("Select number of results:", 1, 10, 5)

# Search Button
if st.button("Search"):
    if query_text.strip():
        results = semantic_search(query_text, top_k)
        st.subheader("🔎 Search Results:")
        
        for i, (title, description, score) in enumerate(results):
            st.write(f"**{i+1}. {title}**  \n_Description: {description}_  \n_Similarity Score: {score:.4f}")
            st.markdown("---")
    else:
        st.warning("Please enter a search query.")
