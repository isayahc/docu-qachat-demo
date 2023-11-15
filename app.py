import gradio as gr
from dotenv import load_dotenv
from bs4 import BeautifulSoup

config = load_dotenv(".env")

from langchain.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=10)

from langchain.llms import HuggingFaceHub
model_id = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.1, "max_new_tokens":300})

from langchain.embeddings import HuggingFaceHubEmbeddings
embeddings = HuggingFaceHubEmbeddings()

from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

web_links = ["https://www.databricks.com/","https://help.databricks.com","https://docs.databricks.com","https://kb.databricks.com/","http://docs.databricks.com/getting-started/index.html","http://docs.databricks.com/introduction/index.html","http://docs.databricks.com/getting-started/tutorials/index.html","http://docs.databricks.com/machine-learning/index.html","http://docs.databricks.com/sql/index.html"]
loader = WebBaseLoader(web_links)
documents = loader.load()
     
texts = text_splitter.split_documents(documents)
db = Chroma.from_documents(texts, embeddings)
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
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;)</p>
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