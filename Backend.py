from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

import os

class chatbot():

    # Set the init variables
    def __init__(self, embedding_model, llm_model_file,promt_template):
        self.embedding_model = embedding_model
        self.llm_model_file = llm_model_file
        self.promt_template = promt_template

    # Vectorisation process
    def vectorisation(self):
        #  Download embedding  'sentence-transformers/all-MiniLM-L6-v2' by HuggingFaceEmbedding.
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
        )

        # Load all pdf files from the folder
        pdf_folder_path = "Documents"
        documents = []
        for file in os.listdir(pdf_folder_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_folder_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())

        # Data preparing process
        # Create the tokens from text data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunked_documents = text_splitter.split_documents(documents)

        # Convert tokens to embedding(vectors) and store in FAISS database
        store = FAISS.from_documents(chunked_documents, embeddings)

        # Write our index to disk.
        store.save_local("./vectorstore")

        # Load the saved FAISS store from the disk.
        store = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

        # Retrieve data from FAISS database
        retriever = store.as_retriever()

        return retriever

    def RAG_system(self, vectorisation):

        # Create the LLM model and set the configuration
        config = {'max_new_tokens': 200, 'repetition_penalty': 1.1, 'context_length': 4096 }
        llm = CTransformers(model = 'TheBloke/Llama-2-7B-Chat-GGUF', model_file = self.llm_model_file, config = config)

        retriever = vectorisation
        # Step 5
        # qa_chain = create_qa_with_sources_chain(llm)

        # Retrieval-Augmented Generation process
        retrieval_qa = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': self.promt_template}
        )

        return retrieval_qa
