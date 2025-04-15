
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

bi_pher = {}
tri_pher = {}

def tokenize(text):
    return [w.strip(".,!?;:").lower() for w in text.split() if w.strip(".,!?;:")]

def deposit(tokens, delta_tri=0.5, delta_bi=0.3):
    for i in range(len(tokens) - 2):
        tri_pher[(tokens[i], tokens[i+1], tokens[i+2])] = tri_pher.get((tokens[i], tokens[i+1], tokens[i+2]), 0) + delta_tri
    for i in range(len(tokens) - 1):
        bi_pher[(tokens[i], tokens[i+1])] = bi_pher.get((tokens[i], tokens[i+1]), 0) + delta_bi

def visualize_graph(threshold=0.1):
    G = nx.DiGraph()
    for (a,b), wt in bi_pher.items():
        if wt >= threshold:
            G.add_edge(a, b, weight=round(wt, 2))
    return G

def sample_next(w1, w2=None, k=5):
    if w2:
        candidates = [(w3, v) for (a,b,w3),v in tri_pher.items() if a==w1 and b==w2]
    else:
        candidates = [(b, v) for (a,b),v in bi_pher.items() if a==w1]
    if not candidates:
        return random.choice(list(set(t for _,t in bi_pher)))
    candidates = sorted(candidates, key=lambda x:-x[1])
    return [c for c,_ in candidates[:k]]

st.set_page_config(page_title="Stigma Suite", layout="wide")
st.title("🧠 Stigma: Pheromone-Based Language Engine")

tabs = st.tabs(["Trail Visualizer", "Text Generator", "Reinforcement", "Emergence", "BERT Comparison", "Metrics", "About"])

with tabs[0]:
    st.subheader("Trail Visualizer")
    sample = st.text_area("Enter text to visualize:", "She was not a bit handsome, but she was very clever.")
    tokens = tokenize(sample)
    deposit(tokens)

    threshold = st.slider("Minimum trail strength to display", 0.0, 2.0, 0.1, 0.1)
    G = visualize_graph(threshold=threshold)
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8,6))
        nx.draw(G, pos, with_labels=True, node_size=800, node_color="lightblue", font_size=10,
                width=[G[u][v]['weight'] for u,v in G.edges()], edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v): f"{d['weight']}" for u,v,d in G.edges(data=True)})
        st.pyplot(plt)
    else:
        st.info("No edges above the threshold. Try reducing it.")

with tabs[1]:
    st.subheader("Stigmergic Text Generator")
    seed1 = st.text_input("First word", "she")
    seed2 = st.text_input("Second word (optional)", "was")
    length = st.slider("Words to generate", 5, 50, 15)
    generated = [seed1]
    if seed2: generated.append(seed2)
    for _ in range(length):
        next_words = sample_next(*generated[-2:])
        generated.append(random.choice(next_words))
    st.markdown("**Generated Text:**")
    st.success(" ".join(generated))

with tabs[2]:
    st.subheader("Reinforcement Dynamics")
    phrase = st.text_input("Enter phrase to reinforce", "She was clever")
    reps = st.slider("Repetitions", 1, 50, 10)
    toks = tokenize(phrase)
    for _ in range(reps):
        deposit(toks)
    st.success(f"Reinforced {phrase!r} {reps} times.")

with tabs[3]:
    st.subheader("Emergence Explorer")
    novel = st.text_area("Introduce new phrase", "Though plain, she was witty.")
    deposit(tokenize(novel))
    st.success("New phrase added to memory. Check Trail Visualizer to see impact.")

with tabs[4]:
    st.subheader("⚠️ Coming Soon")
    st.info("BERT attention heatmap comparison will be integrated in next update.")

with tabs[5]:
    st.subheader("Trail Metrics")
    st.write(f"🔢 Total bi-grams: {len(bi_pher)}")
    st.write(f"🔢 Total tri-grams: {len(tri_pher)}")
    st.write(f"🚀 Most reinforced pair: {max(bi_pher.items(), key=lambda x:x[1])}")

with tabs[6]:
    st.markdown("""
### 🧬 What is Stigma?
Stigma is a lightweight, emergent language engine inspired by stigmergy — a biological principle where memory and behavior arise through trails left in the environment.

### 🔍 Features
- Pheromone-style trail memory (bi & tri-gram)
- Interactive text generation and evolution
- Visual, interpretable edge-weight graphs
- Fast, real-time, no pretrained models

### 🚀 Built for:
- Language exploration
- Educational demos
- Alternative NLP research

Made with ❤️ using Streamlit.
""")
