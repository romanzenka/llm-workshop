from huggingface_hub import login
from langchain import HuggingFacePipeline
from os import path
import torch

device = "cuda:6" if torch.cuda.is_available() else "cpu"

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

print("-------------------------------------")
question = "What is RF Fold?"
print(f"Find closest database entry to '{question}'\n")
searchDocs = db.similarity_search(question)
print(searchDocs[0].page_content)
print("=====================================")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype = torch.float16)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=device, 
                generation_config=generation_config)
llm = HuggingFacePipeline(
   pipeline = pipe,
   model_kwargs={"temperature": 0, "max_length": 2048, "return_full_text": False},
)

from langchain.prompts import PromptTemplate

template = """<s>[INST] <<SYS>>
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible.
{context}
<</SYS>>
{question}
[/INST]"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=db.as_retriever(),
  chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
print("-----------------------")
query = { "query" : "What technique proposed in 2023 can be used to predict protein folding?" };
print(f"Question: {query['query']}")
result = qa_chain (query)
print(result["result"])
print("=======================")

print("-----------------------")
query = { "query" : "Which scientific article should I read to learn about RFdiffusion for protein folding?" }
print(f"Question: {query['query']}")
article = qa_chain (query)
print(article["result"])
print("=======================")

