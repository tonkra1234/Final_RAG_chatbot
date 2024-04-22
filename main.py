import streamlit as st

from Backend import chatbot
from langchain_core.prompts import PromptTemplate

import time
from datetime import datetime

# Generate the response and source document
def response_generator(question):

    # answer = llm_chain.invoke(question)
    output = retrieval_qa.invoke({
        "question": question,
        "chat_history": []
    })
    results = output['answer']
    sources = output['source_documents'][0].metadata['source']
    return results , sources

# Streamed response emulator
def streaming_text(results):

    result = results.strip()
    for word in result.split():
        yield word + " "
        time.sleep(0.04)

# Prompt template
template = """### [INST] Instruction: Your name is Natbot. 
You will be provided with questions and related data. 
Your task is to find the answers to the questions using the given data. 
The answer should be brief as one paragraph within 200 words. 
If the data doesn't contain the answer to the question, then you must return 'Not enough information.'

{context}

### Question: {question} [/INST]"""

# Create prompt and provide variables
prompts = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Streamlit header section
st.title(":male-technologist: Chat with :blue[NatBot]")
st.caption("""$\qquad$The Nat Chatbot represents a significant advancement in conversational AI technology,
           Powered by Langchain framwork to run to model using( Ctransformers ) and The Bloke's Llama-2-7B-Chat-GGUF model from Hugginface Hub,
           it delivers accurate and engaging responses across various topics.
           User interface ensures a seamless user experience,
           To quickly prototype and deploy a simple web interface for chatbot """
           )
st.header('', divider='rainbow')

# Call call chatbot from Backend
RAG = chatbot(embedding_model = "sentence-transformers/all-MiniLM-L6-v2", llm_model_file = 'llama-2-7b-chat.Q5_K_M.gguf', promt_template = prompts)

# Spinner during vectorisation and retrieval process
if 'key' not in st.session_state:
    with st.spinner(text="Vectorising the dataset ......."):
        # Vectorisation process
        vectorisation_data = RAG.vectorisation()
    with st.spinner(text="Initialize the system ......."):
        # Retrieval process
        retrieval_qa = RAG.RAG_system(vectorisation_data)
    st.session_state['key'] = retrieval_qa

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I'm Natbot, your assistance. How can I assist you?"}
    ]

# Show the history chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Call retrieval
retrieval_qa = st.session_state['key']

# Accept user input
if question:= st.chat_input("Message Natbot ...."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response in chat message container
    with st.spinner(text="In progress..."):
        # Generate response
        result = response_generator(question)
    with st.chat_message("assistant"):
        # Perform streaming response
        streaming_result = streaming_text(result[0])
        response = st.write_stream(streaming_result)
        # Show data source and datetime of execution
        st.markdown("**Retrieve from : " + result[1] + ' at ' + datetime.now().strftime(
            "%I:%M:%S %p") + ' on ' + datetime.now().strftime("%d-%b-%Y") + '**')

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})