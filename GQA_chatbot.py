# pip install ctransformers
# pip install langchain-community
# pip install langchain

from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

#  config and download LLM model.
config = {'max_new_tokens': 256, 'repetition_penalty': 1.1}
llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGUF', model_file='llama-2-7b-chat.Q4_K_M.gguf', config=config)

# Setup prompt template
template = """
You are an AI chatbot for NASA named Natbot.
If you don't know the answer, just apologize and say that you cannot answer.
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Perform the chain between LLM and Prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Start the loop
while True:
    question = input("> ")
    answer = llm_chain.invoke(question)
    print(answer['text'])

