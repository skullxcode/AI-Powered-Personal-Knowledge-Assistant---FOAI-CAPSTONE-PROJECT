import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import os
import plotly.express as px

st.set_page_config(page_title="Chroma DB Visualizer", page_icon="📊", layout="wide")

with st.sidebar:
    st.header("Navigation")
    st.page_link("app.py", label="Back to Assistant", icon="🤖")

st.title("📊 Chroma DB Vector Visualizer")
st.markdown("This app extracts the high-dimensional embeddings from your local Chroma DB and uses PCA (Principal Component Analysis) to reduce them to 3D for visualization.")

DB_PATH = "chroma_db"

if not os.path.exists(DB_PATH):
    st.error(f"Chroma DB folder not found at `{DB_PATH}`. Please run data ingestion first!")
    st.stop()

@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data
def load_and_process_data():
    # Load Chroma
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=get_embedding_function()
    )
    
    # Get all stored items including embeddings
    # We query the _collection directly to get all data
    data = vector_db._collection.get(include=["embeddings", "documents", "metadatas"])
    
    return data

with st.spinner("Loading vectors from Chroma DB..."):
    data = load_and_process_data()

if data.get("embeddings") is None or len(data["embeddings"]) == 0:
    st.warning("No embeddings found in the database. Please ingest some documents first.")
    st.stop()

embeddings = np.array(data["embeddings"])
documents = data["documents"]
metadatas = data["metadatas"]
ids = data["ids"]

st.write(f"**Total Vectors Found:** {len(embeddings)}")
if len(embeddings) > 0:
    st.write(f"**Vector Dimensionality:** {embeddings.shape[1]}")

st.divider()

# Dimensionality reduction
st.subheader("3D Vector Visualization using PCA")
if len(embeddings) >= 3:
    with st.spinner("Applying PCA..."):
        pca = PCA(n_components=3)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Build DataFrame for visualization
        df = pd.DataFrame(reduced_embeddings, columns=["PCA1", "PCA2", "PCA3"])
        df["ID"] = ids
        df["Source"] = [meta.get("source", "Unknown") for meta in metadatas]
        # Adding a shortened document content for hover
        df["Document Snippet"] = [doc[:100] + "..." if len(doc) > 100 else doc for doc in documents]

        st.markdown("The 3D scatter plot below shows how your document chunks cluster together. Closer points have similar semantic meaning! You can rotate, pan, and zoom to explore.")
        
        # Plot using Plotly Express 3D Scatter
        fig = px.scatter_3d(
            df,
            x="PCA1",
            y="PCA2",
            z="PCA3",
            color="Source",
            hover_data=["Document Snippet", "ID"],
            opacity=0.8
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(height=700, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Show Raw PCA Data"):
            st.dataframe(df)
else:
    st.info("You need at least 3 vectors to perform 3D PCA dimensionality reduction.")

st.divider()

st.subheader("Raw Vector Data Sample")
st.markdown("Here is what the raw embeddings and stored metadata look like inside Chroma DB.")

sample_df = pd.DataFrame({
    "ID": ids[:5],
    "Source": [meta.get("source", "Unknown") for meta in metadatas[:5]],
    "Document Snippet": [doc[:50] + "..." for doc in documents[:5]],
    "Raw Vector (First 5 dims)": [str(vec[:5]) + "..." for vec in embeddings[:5]]
})

st.dataframe(sample_df, use_container_width=True)
st.success("Visualization complete!")
