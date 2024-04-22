import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import pacmap
import numpy as np
import plotly.express as px
import pandas as pd

# Load all pdf files from the folder
pdf_folder_path = "Documents"
documents = []
for file in os.listdir(pdf_folder_path):
   if file.endswith('.pdf'):
       pdf_path = os.path.join(pdf_folder_path, file)
       loader = PyPDFLoader(pdf_path)
       documents.extend(loader.load())

# Load and execute embedding model
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Data preprocessing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)
chunked_documents = text_splitter.split_documents(documents)

# Convert tokens to embedding(vectors) and store in FAISS database
store = FAISS.from_documents(chunked_documents, embeddings)

# Embed a user query in the same space
user_query = "How old of the Moon"
query_vector = embeddings.embed_query(user_query)

# Reduce the dimension of embedding from 384D to 2D
embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)

embeddings_2d = [
   list(store.index.reconstruct_n(idx, 1)[0]) for idx in range(len(chunked_documents))
] + [query_vector]

# Fit the data (The index of transformed data corresponds to the index of the original data)
documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")

# Prepare the data into the pandas Dataframe
data = [
   {
       "x": documents_projected[i, 0],
       "y": documents_projected[i, 1],
       # "source": docs_processed[i].metadata["source"].split("/")[1], for single file
       "source": chunked_documents[i].metadata.get("source", "Unknown"),
       "extract": chunked_documents[i].page_content[:100] + "...",
       "symbol": "circle",
       "size_col": 10,
   }
   for i in range(len(chunked_documents))
] + [
   {
       "x": documents_projected[-1, 0],
       "y": documents_projected[-1, 1],
       "source": "User query",
       "extract": user_query,
       "size_col": 100,
       "symbol": "star",
   }
]

df = pd.DataFrame(data)

# Visualize the embedding
fig = px.scatter(
   df,
   x="x",
   y="y",
   color="source",
   hover_data="extract",
   size="size_col",
   symbol="symbol",
   color_discrete_map={"User query": "black"},
   width=1000,
   height=700,
)
fig.update_traces(
   marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
   selector=dict(mode="markers"),
)
fig.update_layout(
   legend_title_text="<b>Chunk source</b>",
   title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
)
fig.show()