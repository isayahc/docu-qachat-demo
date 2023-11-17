import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.vectorstores.utils import filter_complex_metadata
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
)

# Stage one: read all the docs, split them into chunks.
st = time.time()
print('Loading documents ...')
docs = loader.load()
chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
et = time.time() - st
print(f'Time taken: {et} seconds.')

#Stage two: embed the docs.
# use all-mpnet-base-v2 sentence transformer to convert pieces of text in vectors to store them in the vector store
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
    )
print(f'Loading chunks into vector store ...')
st = time.time()
db = Chroma.from_documents(filter_complex_metadata(chunks), embeddings, persist_directory="/content/chroma_db")
et = time.time() - st
print(f'Time taken: {et} seconds.')