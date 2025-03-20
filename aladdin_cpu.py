import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Load the dataset
file_path = 'netflix_movies_detailed_up_to_2025.csv'
df = pd.read_csv(file_path)

# Initialize graph
G = nx.Graph()

# Load semantic model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# 🏗️ Step 3: Auto-Generate Edges (with Multiprocessing)
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

        # Semantic similarity of text fields (descriptions, titles)
        for col in text_cols:
            if pd.notna(row1[col]) and pd.notna(row2[col]):
                text1 = semantic_model.encode(row1[col], convert_to_tensor=True)
                text2 = semantic_model.encode(row2[col], convert_to_tensor=True)
                sim = util.pytorch_cos_sim(text1, text2).item()
                weight += sim * 5

        # 🔥 Boost Actor Similarity in Semantic Comparison
        if pd.notna(row1['cast']) and pd.notna(row2['cast']):
            cast_sim = util.pytorch_cos_sim(
                semantic_model.encode(row1['cast'], convert_to_tensor=True),
                semantic_model.encode(row2['cast'], convert_to_tensor=True)
            ).item()
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

# 🚀 Step 4: Search Function (with Expanded Scope + Path Strength)
def search_movie(query, top_n=10):
    print(f"\n[INFO] Processing search query: '{query}'")
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)

    # Measure similarity to descriptions and rank them
    similarities = {}
    print("[INFO] Calculating similarity...")
    for idx in tqdm(G.nodes, desc="Measuring Similarity"):
        desc = G.nodes[idx].get('description', '')
        if pd.notna(desc):
            desc_embedding = semantic_model.encode(desc, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(query_embedding, desc_embedding).item()
            similarities[idx] = sim

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
