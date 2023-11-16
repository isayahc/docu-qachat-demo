import gradio as gr
import boto3
from dotenv import load_dotenv
from bs4 import BeautifulSoup

config = load_dotenv(".env")

from langchain.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

from langchain.llms import HuggingFaceHub
model_id = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.1, "max_new_tokens":600})

from langchain.embeddings import HuggingFaceHubEmbeddings
embeddings = HuggingFaceHubEmbeddings()

from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

from langchain.prompts import ChatPromptTemplate


#web_links = ["https://www.databricks.com/","https://help.databricks.com","https://docs.databricks.com","https://kb.databricks.com/","http://docs.databricks.com/getting-started/index.html","http://docs.databricks.com/introduction/index.html","http://docs.databricks.com/getting-started/tutorials/index.html","http://docs.databricks.com/machine-learning/index.html","http://docs.databricks.com/sql/index.html"]
#loader = WebBaseLoader(web_links)
#documents = loader.load()

s3 = boto3.client('s3')
s3.download_file('rad-rag-demos', 'vectorstores/chroma.sqlite3', './chroma_db/chroma.sqlite3')

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
db.get()

#texts = text_splitter.split_documents(documents)
#db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever()
global qa 
qa = RetrievalQA.from_chain_type(llm=model_id, chain_type="stuff", retriever=retriever, return_source_documents=True)


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

def infer(question):
    
    query = question
    result = qa({"query": query})

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
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch()