
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re

st.set_page_config(layout="wide")
st.title("🐜 Stigma Debug Trail Visualizer")

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def train_pheromone(tokens):
    bi_pher = defaultdict(float)
    tri_pher = defaultdict(float)
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        tri_pher[(w1, w2, w3)] += 1.0
        bi_pher[(w1, w2)] += 1.0
    if len(tokens) >= 2:
        bi_pher[(tokens[-2], tokens[-1])] += 1.0
    return bi_pher, tri_pher

def stringify_keys(d):
    return {str(k): float(v) for k, v in dict(d).items()}

def draw_trail_graph(bi_pher, tri_pher, threshold):
    G = nx.DiGraph()
    edge_count = 0
    for (w1, w2), weight in bi_pher.items():
        if weight >= threshold:
            G.add_edge(w1, w2, weight=weight)
            edge_count += 1
    for (w1, w2, w3), weight in tri_pher.items():
        if weight >= threshold:
            G.add_edge(f"{w1} {w2}", w3, weight=weight)
            edge_count += 1
    if edge_count == 0:
        st.warning("🚫 No trails meet threshold.")
        return
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color=weights,
            edge_cmap=plt.cm.plasma, width=2, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()},
                                 font_color='purple')
    st.pyplot(plt.gcf(), clear_figure=True)

text = st.text_area("Enter a sentence to debug trail memory:", "She was not a bit handsome, but she was very clever.")
threshold = st.slider("Trail strength threshold", 0.0, 5.0, 0.1)

tokens = tokenize(text)
bi_pher, tri_pher = train_pheromone(tokens)

st.markdown("### 🔍 Debug Info")
st.write("Tokens:", tokens)

st.subheader("Bigram Trails:")
st.write(stringify_keys(bi_pher))

st.subheader("Trigram Trails:")
st.write(stringify_keys(tri_pher))

st.markdown("### 📈 Trail Graph")
draw_trail_graph(bi_pher, tri_pher, threshold)
