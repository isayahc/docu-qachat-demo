# gradio
import gradio as gr
import random
import time
#boto3 for S3 access
import boto3
from botocore import UNSIGNED
from botocore.client import Config
# access .env file
from dotenv import load_dotenv
from bs4 import BeautifulSoup
# langchain
# web loader
#from langchain.document_loaders import WebBaseLoader
# HF libraries
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
# vectorestore
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# retrieval chain
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
# prompt template
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
# logging
import logging

# load HF Token
config = load_dotenv(".env")


model_id = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.1, "max_new_tokens":1024, "repetition_penalty":1.2})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = HuggingFaceHubEmbeddings()

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
s3.download_file('rad-rag-demos', 'vectorstores/chroma.sqlite3', './chroma_db/chroma.sqlite3')

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
db.get()

retriever = db.as_retriever()
global qa 
qa = RetrievalQA.from_chain_type(llm=model_id, chain_type="stuff", retriever=retriever, verbose=True, return_source_documents=True, chain_type_kwargs={
    "verbose": True,
    "memory": ConversationBufferMemory(
        memory_key="history",
        input_key="question"),
}
    )
#qa = RetrievalQAWithSourcesChain.from_chain_type(llm=model_id, chain_type="stuff", retriever=retriever, return_source_documents=True)

template = """


"""

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    history[-1][1] = ""
    for character in response['result']:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

def infer(question, history):
    #logging.basicConfig()
    #logging.getLogger("langchain.chains.retrieval_qa").setLevel(logging.INFO)
    query =  question
    result = qa({"query": query, "history": history, "question": question})

    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>
    <p style="text-align: center;">Chat with Documentation, <br />
    when everything is ready, you can start asking questions about the docu ;)</p>
</div>
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)      
        chatbot = gr.Chatbot([], elem_id="chatbot")
        clear = gr.Button("Clear")
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
    question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()
