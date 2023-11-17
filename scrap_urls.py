import chromadb
import time
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


#web_links = ["https://docs.ray.io"]
web_links = ["https://www.tum.de/", "https://hm.edu/", "https://www.thi.de/", "https://www.fau.de/", "https://www.tha.de/", "https://www.haw-landshut.de/", "https://www.uni-augsburg.de/", "https://www.uni-bamberg.de/"]

print("starting with loading URLs \n")
udocs = []
for ulink in web_links:
  url = ulink
  print(f"loading the URL {url} \n")
  loader = RecursiveUrlLoader(url=url, max_depth=6, extractor=lambda x: Soup(x, "html.parser").text)
  udocuments = loader.load()
print(f'finished loading URLs, we have scrapped {len(udocuments)} document \n')

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
)
print(f'finished chunking documents, we have scrapped {len(udocuments)} document \n')

# Stage one: read all the docs, split them into chunks.
st = time.time()
print('Loading documents ...')
docs = loader.load()
chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
et = time.time() - st
print(f'Time taken: {et} seconds.')
print(f'finished chunking documents, we have {len(chunks)} chunk \n')

#Stage two: embed the docs.
# use all-mpnet-base-v2 sentence transformer to convert pieces of text in vectors to store them in the vector store
model_name = "sentence-transformers/all-mpnet-base-v2"
#model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
#    model_kwargs=model_kwargs
    )
print(f'Loading chunks into vector store ...')
st = time.time()
db = Chroma.from_documents(filter_complex_metadata(chunks), embeddings, persist_directory="/chroma_db")
et = time.time() - st
print(f'Time taken: {et} seconds.')