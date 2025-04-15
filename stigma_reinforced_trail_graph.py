
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import math
import re

st.set_page_config(layout="wide")
st.title("🧠 Stigma Trail Graph (Emergent + Reinforced + Decay)")

# Reset
if st.button("🏠 Reset"):
    st.cache_data.clear()
    st.rerun()

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def compute_entropy(tokens):
    total = len(tokens)
    counts = Counter(tokens)
    entropy = {}
    for tok, count in counts.items():
        p = count / total
        entropy[tok] = -p * math.log2(p)
    return entropy

def update_trails(tokens, bi_pher, tri_pher, entropy, decay=0.98, reinforce=1.0):
    # Apply decay first
    for k in list(bi_pher.keys()):
        bi_pher[k] *= decay
    for k in list(tri_pher.keys()):
        tri_pher[k] *= decay

    # Reinforce current tokens
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        e = entropy.get(w3, 1e-9)
        weight = reinforce / (e + 1e-9)
        bi_pher[(w1, w2)] += weight
        tri_pher[(w1, w2, w3)] += weight

    return bi_pher, tri_pher

def draw_graph(bi_pher, tri_pher, threshold):
    G = nx.DiGraph()
    max_w = max(list(bi_pher.values()) + list(tri_pher.values()) + [1.0])
    for (src, tgt), weight in bi_pher.items():
        if weight >= threshold:
            G.add_edge(src, tgt, weight=weight)
    for (w1, w2, w3), weight in tri_pher.items():
        if weight >= threshold:
            G.add_edge(f"{w1} {w2}", w3, weight=weight)

    if not G.edges:
        st.warning("🚫 No trails meet the threshold.")
        return

    pos = nx.spring_layout(G, seed=42, k=0.9)
    edge_widths = [G[u][v]['weight'] / max_w * 4 for u, v in G.edges()]
    edge_labels = { (u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges() }

    fig, ax = plt.subplots(figsize=(11, 7))
    nx.draw_networkx_nodes(G, pos, node_size=1400, node_color='lightblue', alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='darkblue',
                           arrowstyle='-|>', connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='darkred', ax=ax)
    ax.set_title("Emergent + Reinforced Trail Graph", fontsize=14)
    ax.axis('off')
    st.pyplot(fig)

# Initial memory
if "bi_pher" not in st.session_state:
    st.session_state.bi_pher = defaultdict(float)
    st.session_state.tri_pher = defaultdict(float)

text = st.text_area("Enter new input to stimulate trail evolution:", "She was not a bit handsome, but she was very clever.")
threshold = st.slider("Trail threshold", 0.0, 5.0, 0.1)
tokens = tokenize(text)
entropy = compute_entropy(tokens)

# Update trails with reinforcement + decay
st.session_state.bi_pher, st.session_state.tri_pher = update_trails(
    tokens, st.session_state.bi_pher, st.session_state.tri_pher, entropy, decay=0.98, reinforce=1.0
)

# Show trails
st.markdown("### 🔗 Current Trails")
draw_graph(st.session_state.bi_pher, st.session_state.tri_pher, threshold)
