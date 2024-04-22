# :robot: Natbot - Retrieval-Augmented Generation (RAG) for chatbots


<p align="center">
  <img width="460" height="460" src="https://github.com/tonkra1234/Final_RAG_chatbot/blob/main/Description/RAG%20chatbot.PNG">
</p>


Retrieval-Augmented Generation (RAG) for chatbots using [LangChain](https://python.langchain.com/docs/get_started/introduction.html) and [Huggingface](https://huggingface.co/) to retrieve the informations from the documents and [Streamlit](https://streamlit.io/) to create friendly user interface. The structure and components will be discussed to clearer understand about RAG chatbot.

### Artificial Intelligence and Chatbot

It’s very surprising that Generative Artificial Intelligence is dramatically interesting. People widespreadly adopted AI and are gradually relying on them. This technology has changed the way we interact with machines. One of the most obvious cases is Generative Question Answering (GQA).  The human-like interaction can be formed by GQA and enhanced by Information Retrieval (IR). The clear examples of IR are Google search, Netflix and Amazon. Imagine a chatbot that can summarize and answer your queries based on the provided information. Nowadays, this imagination is already possible. This article will delve into retrieval-augmented GQA and how to apply it in real-life projects.
    
## Table of contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#Getting-Started)
- [References](#references)

## Introduction

### Generative Question Answering (GQA)

User queries can be answered by using large language models (LLMs) to generate human-like responses. The basic parameters of a straightforward GQA system are a user query and LLMs. The user query and machine response are displayed pass through the user interface. Diagram in Figure xx illustrates that there are three main components of GQA chatbot. Firstly, user input represents the query provided by users. Secondly,  prompt represents the process of dynamically assigning specific tasks and instructions of the selected LLMs. Lastly, It receives the user input after prompting to generate the response based on the given prompt and comprehension of language pattern. However, this process can not perform without LLMs chain. Chains combine the different primitive (user input) and LLMs to construct the sequence of NPL operation.

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) currently combines the Information Retrieval and Generative models to provide a higher level of artificial intelligence application. The Information Retrieval search through databases to find relevant information and Generative models is capable of dynamically generating human-like answers.

### Information Retrieval (IR)

To create an external knowledge base, information retrieval is necessary. Knowledge Base plays a crucial role as a place where the information is stored and fed to the system. The storage of semantically relevant information is known as a vector database. A vector database stores vectors representing the information encoded by machine learning models.  These models known as Embedding can convert and encode passages into vector space by vectorisation technique.

The ‘hallucination’ problem of the LLMs is eliminated because of the additional source knowledge (fed directly into the model) or Information Retrieval (IR). These combinations improve the reliability of the information and capability of chatbots.

## Prerequisites

Before starting develope RAG chatbot, Python >=3.10.0 and these packages should be installed on the system consisting of :

1. langchain
2. langchain-community
3. sentence-transformers 
4. faiss-cpu 
5. ctransformers
6. pypdf
7. streamlit
8. pandas
9. langchain-experimental
10. pacmap
11. plotly 
12. numpy

## Installation

1. Clone this repository to the local machine
```shell
git clone https://github.com/tonkra1234/Final_RAG_chatbot.git
```
2. Install the requirement packages
```shell
pip install -r requirements.txt
```

## Getting Started

After the project is cloned and the packages are installed, the project is ready to start.

1. Set the selected LLM and embedding model in the main.py
   
```shell
embedding_model = "sentence-transformers/all-MiniLM-L6-v2", llm_model_file = 'llama-2-7b-chat.Q5_K_M.gguf'
```

2. Run the RAG chatbot by execute this code on the terminal to run Streamlit
   
```shell
streamlit run main.py
```

## References
* LLMs:
  * [Calculating GPU memory for serving LLMs](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm/)
  * [Building Response Synthesis from Scratch](https://gpt-index.readthedocs.io/en/latest/examples/low_level/response_synthesis.html#)
  * [Attention Sinks in LLMs for endless fluency](https://huggingface.co/blog/tomaarsen/attention-sinks)
  * [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
  * [Introduction to Weight Quantization](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c)
* LLM integration and Modules:
  * [LangChain](https://python.langchain.com/docs/get_started/introduction.html):
    * [MarkdownTextSplitter](https://api.python.langchain.com/en/latest/_modules/langchain/text_splitter.html#MarkdownTextSplitter)
    * [Chroma Integration](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma)
    * [The Problem With LangChain](https://minimaxir.com/2023/07/langchain-problem/#:~:text=The%20problem%20with%20LangChain%20is,don't%20start%20with%20LangChain)
* Embeddings:
  * [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    * This is a `sentence-transformers` model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
* Vector Databases:
  * [Chroma](https://www.trychroma.com/)
  * [Food Discovery with Qdrant](https://qdrant.tech/articles/new-recommendation-api/#)
  * Indexing algorithms:
    * There are many algorithms for building indexes to optimize vector search. Most vector databases implement `Hierarchical Navigable Small World (HNSW)` and/or `Inverted File Index (IVF)`. Here are some great articles explaining them, and the trade-off between `speed`, `memory` and `quality`:
      * [Nearest Neighbor Indexes for Similarity Search](https://www.pinecone.io/learn/series/faiss/vector-indexes/)
      * [Hierarchical Navigable Small World (HNSW)](https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37)
      * [From NVIDIA - Accelerating Vector Search: Using GPU-Powered Indexes with RAPIDS RAFT](https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/)
      * [From NVIDIA - Accelerating Vector Search: Fine-Tuning GPU Index Algorithms](https://developer.nvidia.com/blog/accelerating-vector-search-fine-tuning-gpu-index-algorithms/)
      * > PS: Flat indexes (i.e. no optimisation) can be used to maintain 100% recall and precision, at the expense of speed.
* Retrieval Augmented Generation (RAG):
  * [Rewrite-Retrieve-Read](https://github.com/langchain-ai/langchain/blob/master/cookbook/rewrite.ipynb)
    * > Because the original query can not be always optimal to retrieve for the LLM, especially in the real world, we first prompt an LLM to rewrite the queries, then conduct retrieval-augmented reading.
  * [Rerank](https://txt.cohere.com/rag-chatbot/#implement-reranking)
  * [Conversational awareness](https://langstream.ai/2023/10/13/rag-chatbot-with-conversation/)
  * [Summarization: Improving RAG quality in LLM apps while minimizing vector storage costs](https://www.ninetack.io/post/improving-rag-quality-by-summarization)
* Chatbot Development:
  * [Streamlit](https://discuss.streamlit.io/):
    * [Build a basic LLM chat app](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app)
    * [Layouts and Containers](https://docs.streamlit.io/library/api-reference/layout)
    * [st.chat_message](https://docs.streamlit.io/library/api-reference/chat/st.chat_message)
    * [Add statefulness to apps](https://docs.streamlit.io/library/advanced-features/session-state)
      * [Why session state is not persisting between refresh?](https://discuss.streamlit.io/t/why-session-state-is-not-persisting-between-refresh/32020)
    * [st.cache_resource](https://docs.streamlit.io/library/api-reference/performance/st.cache_resource)
    * [Handling External Command Line Arguments](https://github.com/streamlit/streamlit/issues/337)
  * [(Investigate) FastServe - Serve Llama-cpp with FastAPI](https://github.com/aniketmaurya/fastserve)
  * [(Investigate) Chat Templates to standardise the format](https://huggingface.co/blog/chat-templates)
  * [(Investigate) Ollama](https://github.com/ollama/ollama)
* Text Processing and Cleaning:
  * [clean-text](https://github.com/jfilter/clean-text/tree/main)
* Open Source Repositories:
  * [CTransformers](https://github.com/marella/ctransformers)
  * [GPT4All](https://github.com/nomic-ai/gpt4all)
  * [llama.cpp](https://github.com/ggerganov/llama.cpp)
  * [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
  * [pyllamacpp](https://github.com/abdeladim-s/pyllamacpp)
  * [chroma](https://github.com/chroma-core/chroma)
  * Inspirational repos:
    * [lit-gpt](https://github.com/Lightning-AI/lit-gpt)
    * [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
    * [PrivateDocBot](https://github.com/Abhi5h3k/PrivateDocBot)
    * [Rag_bot - Adaptive Intelligence Chatbot](https://github.com/kylejtobin/rag_bot)

