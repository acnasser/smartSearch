import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the dataset
file_path = 'netflix_movies_detailed_up_to_2025.csv'
df = pd.read_csv(file_path)

# Load semantic model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for descriptions
print("[INFO] Creating embeddings...")
descriptions = df['description'].fillna('').tolist()
embeddings = semantic_model.encode(descriptions, convert_to_tensor=False)

# Build FAISS index
print("[INFO] Building FAISS index...")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# 🚀 Search Function
def search_movie(query, top_n=5):
    print(f"\n[INFO] Processing search query: '{query}'")
    query_embedding = semantic_model.encode(query, convert_to_tensor=False).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_n)
    
    # Format results
    results = df.iloc[indices[0]][['title', 'director', 'cast', 'genres', 'description']]
    
    print(f"\n✅ Found {len(results)} related movies!\n")
    return results

# Example query
query = "I'm looking for a movie. It has some actors from Shutter Island in it. I remember that it was a movie about a dream within a dream."
results = search_movie(query)

# Pretty print results
print(results.to_string(index=False))
