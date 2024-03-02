from huggingface_hub import login
from langchain import HuggingFacePipeline
from os import path
import torch

device = "cuda:7" if torch.cuda.is_available() else "cpu"

with open('/homes/ac.rzenka/llm/.token', 'r') as file:
    hf_token = file.read().replace('\n', '')

login(token=hf_token, add_to_git_credential=True)

from langchain.embeddings import HuggingFaceEmbeddings
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':device}
encode_kwargs = {'normalize_embeddings':False}
embeddings = HuggingFaceEmbeddings(
  model_name = modelPath,
  model_kwargs = model_kwargs,
  encode_kwargs=encode_kwargs
)

from langchain.vectorstores import FAISS
if not path.exists("faiss_pdfs"):
  from langchain_community.document_loaders import DirectoryLoader
  loader = DirectoryLoader('/homes/ac.rzenka/llm/llm-workshop/tutorials/04-rag/PDFs', glob="**/*.pdf", show_progress=True)
  documents = loader.load()

  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
  docs = text_splitter.split_documents(documents)

  db = FAISS.from_documents(docs, embeddings)
  db.save_local("faiss_pdfs")
else:
  print("Loading existing FAISS database")
  db = FAISS.load_local("faiss_pdfs", embeddings)

question = "What is RF Fold?"
searchDocs = db.similarity_search(question)
print(searchDocs[0].page_content)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
llm = HuggingFacePipeline(
   pipeline = pipe,
   model_kwargs={"temperature": 0, "max_length": 2048, "return_full_text": False},
)

from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=db.as_retriever(),
  chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain ({ "query" : "What technique proposed in 2023 can be used to predict protein folding?" })
print(result["result"])

article = qa_chain ({ "query" : "Which scientific article should I read to learn about RFdiffusion for protein folding?" })
print(article["result"])

