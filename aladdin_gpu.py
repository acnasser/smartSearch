import pandas as pd
import networkx as nx
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Load the dataset
file_path = 'netflix_movies_detailed_up_to_2025.csv'
df = pd.read_csv(file_path)

# Initialize graph
G = nx.Graph()

# 🔥 Load semantic model on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 🔎 Step 1: Schema Detection
categorical_cols = []
numerical_cols = []
text_cols = []

print("[INFO] Detecting schema...")
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'string':
        if df[col].nunique() < 1000:
            categorical_cols.append(col)
        else:
            text_cols.append(col)
    elif df[col].dtype in ['int64', 'float64']:
        numerical_cols.append(col)

print(f"✅ Categorical Columns: {categorical_cols}")
print(f"✅ Numerical Columns: {numerical_cols}")
print(f"✅ Text Columns: {text_cols}")

# 🧠 Step 2: Add Nodes (with Progress Bar)
print("[INFO] Adding nodes to the graph...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building Nodes"):
    node_data = {col: row[col] for col in df.columns}
    G.add_node(idx, **node_data)

print(f"✅ Added {len(G.nodes)} nodes to the graph.")

# 🔥 Precompute embeddings for text fields
print("[INFO] Precomputing text embeddings on GPU...")
text_embeddings = {}
for col in text_cols + ['cast']:  # Boost actor similarity
    valid_rows = df[col].dropna().astype(str).tolist()
    encoded_embeddings = semantic_model.encode(valid_rows, convert_to_tensor=True, device=device)
    text_embeddings[col] = {idx: encoded_embeddings[i] for i, idx in enumerate(df[col].dropna().index)}

print("✅ Precomputed text embeddings.")

# 🏗️ Step 3: Auto-Generate Edges (Fully GPU Optimized)
def calculate_edge(i):
    edges = []
    row1 = df.iloc[i]

    for j in range(i + 1, len(df)):
        row2 = df.iloc[j]
        weight = 0

        # Shared categorical values (actors, directors, genres)
        for col in categorical_cols:
            if pd.notna(row1[col]) and pd.notna(row2[col]):
                shared_values = set(str(row1[col]).split(', ')) & set(str(row2[col]).split(', '))
                weight += len(shared_values) * 10  # 🔥 Boosted from 3 to 10

        # Similar numerical values (budget, revenue)
        for col in numerical_cols:
            if pd.notna(row1[col]) and pd.notna(row2[col]):
                diff = abs(row1[col] - row2[col])
                weight += max(1 - (diff / (row1[col] + row2[col] + 1e-9)), 0) * 2

        # Semantic similarity of text fields (descriptions, titles) using **precomputed** embeddings
        for col in text_cols:
            if i in text_embeddings[col] and j in text_embeddings[col]:
                sim = util.pytorch_cos_sim(text_embeddings[col][i], text_embeddings[col][j]).item()
                weight += sim * 5

        # 🔥 Boost Actor Similarity in Semantic Comparison
        if i in text_embeddings['cast'] and j in text_embeddings['cast']:
            cast_sim = util.pytorch_cos_sim(text_embeddings['cast'][i], text_embeddings['cast'][j]).item()
            weight += cast_sim * 8  # Boosting from 5 to 8

        if weight > 0:
            edges.append((i, j, weight))

    return edges

def build_edges():
    print("[INFO] Creating edges between nodes...")
    with Pool(processes=cpu_count()) as pool:
        edge_list = list(
            tqdm(pool.imap(calculate_edge, range(len(df))), total=len(df), desc="Building Edges")
        )

    for edges in edge_list:
        for edge in edges:
            G.add_edge(*edge)

    print(f"✅ Created {len(G.edges)} edges between nodes.")

if __name__ == '__main__':
    build_edges()

# 🚀 Step 4: Search Function (GPU Optimized)
def search_movie(query, top_n=10):
    print(f"\n[INFO] Processing search query: '{query}'")
    query_embedding = semantic_model.encode(query, convert_to_tensor=True, device=device)

    # Measure similarity to descriptions and rank them
    similarities = {}
    print("[INFO] Calculating similarity on GPU...")
    desc_embeddings = torch.stack([text_embeddings['description'][idx] for idx in G.nodes if idx in text_embeddings['description']])
    indices = [idx for idx in G.nodes if idx in text_embeddings['description']]
    
    query_embedding = query_embedding.unsqueeze(0)  # Ensure batch size is correct
    similarity_scores = util.pytorch_cos_sim(query_embedding, desc_embeddings).squeeze(0)
    
    similarities = {indices[i]: similarity_scores[i].item() for i in range(len(indices))}

    # Get top similar movies based on semantic similarity
    top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:30]

    # Pathfinding - Strength of relationship with other nodes
    final_results = []
    print("[INFO] Finding connected nodes...")
    for idx, score in tqdm(top_similar, desc="Finding Paths"):
        paths = nx.single_source_dijkstra_path_length(G, idx, weight='weight')
        connected_movies = sorted(paths.items(), key=lambda x: x[1])[:15]
        
        for movie_idx, strength in connected_movies:
            # 🔥 Prioritize edge strength over description similarity
            if movie_idx not in final_results and strength > 0.1:
                final_results.append(movie_idx)

    # Return final results
    results = df.iloc[final_results][['title', 'director', 'cast', 'genres', 'description']]
    print(f"\n✅ Found {len(results)} related movies!\n")
    return results

# Example query
query = "I'm looking for a movie. It has some actors from Shutter Island in it. I remember that it was a movie about a dream within a dream."
results = search_movie(query)

# Pretty print results
print(results.to_string(index=False))
