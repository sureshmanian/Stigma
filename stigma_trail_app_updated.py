
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# --- Trail Memory ---
trail_memory = {}

# --- Input Toggle ---
st.title("Stigma: Language Trail Visualizer")
st.markdown("Select a domain and watch the trail adapt.")

example_texts = {
    "📚 Literary (Austen)": """Elizabeth had never felt so embarrassed before. 
The sudden arrival of Mr. Darcy had unsettled her composure, and she could hardly speak a word. 
Jane, ever composed, gave her a reassuring glance, but the tension lingered in the drawing room like a forgotten letter.""",

    "🔬 Science": """Photosynthesis is the process by which green plants convert sunlight into chemical energy. 
This transformation occurs in the chloroplasts, where chlorophyll captures light energy. 
The resulting glucose fuels the plant’s metabolic activities and releases oxygen as a byproduct.""",

    "📰 News": """The government announced a new digital policy on Tuesday, aiming to strengthen cybersecurity infrastructure across the country. 
Experts say the initiative could lead to more robust data protection measures and increased transparency in public systems."""
}

selected = st.selectbox("Choose an input text:", list(example_texts.keys()))
text = example_texts[selected]
st.text_area("Selected Text", value=text, height=150)

# --- Tokenization ---
def tokenize(text):
    return [w.strip(".,;!?").lower() for w in text.split() if w.strip(".,;!?")]

tokens = tokenize(text)

# --- Build or update trail memory ---
for i in range(len(tokens)-1):
    pair = (tokens[i], tokens[i+1])
    trail_memory[pair] = trail_memory.get(pair, 0) + 1.0

# --- Threshold slider ---
threshold = st.slider("Trail Threshold", 0.01, 1.0, 0.05)

# --- Draw graph ---
G = nx.DiGraph()
for (w1, w2), weight in trail_memory.items():
    if weight >= threshold:
        G.add_edge(w1, w2, weight=round(weight, 2))

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8,6))
nx.draw_networkx_nodes(G, pos, node_color='lavender', node_size=1000)
nx.draw_networkx_edges(G, pos, edge_color='purple', width=[G[u][v]['weight'] for u,v in G.edges()])
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v): f"{d['weight']:.2f}" for u,v,d in G.edges(data=True)}, font_color='violet')

st.pyplot(plt)
