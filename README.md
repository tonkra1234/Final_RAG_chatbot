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
    git clone https://github.com/tonkra1234/Final_RAG_chatbot.git
2. Install the requirement packages
    pip install -r requirements.txt

## Getting Started

## References


