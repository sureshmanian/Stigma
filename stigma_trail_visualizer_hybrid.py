
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re

st.set_page_config(layout="wide")
st.title("🐜 Stigma ACO Trail Visualizer (Hybrid)")

# Reset
if st.button("🏠 Reset App"):
    st.cache_data.clear()
    st.rerun()

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def train_pheromone(tokens):
    bi_pher = defaultdict(float)
    node_freq = defaultdict(int)
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        bi_pher[(w1, w2)] += 1.0
        node_freq[w1] += 1
    if tokens:
        node_freq[tokens[-1]] += 1
    return bi_pher, node_freq

def draw_graph(bi_pher, node_freq, threshold):
    G = nx.DiGraph()
    max_weight = max(bi_pher.values(), default=1.0)

    # Add nodes with frequency labels
    for node, freq in node_freq.items():
        label = f"{node} (×{freq})" if freq > 1 else node
        G.add_node(node, label=label)

    edge_count = 0
    for (src, tgt), weight in bi_pher.items():
        if weight >= threshold:
            G.add_edge(src, tgt, weight=weight)
            edge_count += 1

    if edge_count == 0:
        st.warning("🚫 No trails meet threshold.")
        return

    pos = nx.spring_layout(G, k=0.9, seed=42)
    edge_widths = [G[u][v]['weight'] / max_weight * 4 for u, v in G.edges()]
    edge_labels = { (u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges() }

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1400, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"), font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='darkblue', arrows=True,
                           arrowstyle='-|>', connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=9, ax=ax)

    ax.set_title("Trail Graph", fontsize=14)
    ax.axis("off")
    st.pyplot(fig)

# UI
text = st.text_area("📝 Enter sentence:", "She was not a bit handsome, but she was very clever.")
threshold = st.slider("Edge threshold", 0.0, 5.0, 0.1)

tokens = tokenize(text)
bi_pher, node_freq = train_pheromone(tokens)

st.markdown("### 🧠 Tokens and Frequency")
st.write(tokens)
st.write(node_freq)

st.markdown("### 🔗 Trail Graph")
draw_graph(bi_pher, node_freq, threshold)
