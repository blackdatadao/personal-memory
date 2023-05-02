# -*- coding: utf-8 -*-

from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
from langchain.document_loaders import UnstructuredURLLoader
# from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader

# from gpt_index import Document, GPTSimpleVectorIndex

from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, SimpleDirectoryReader,GPTTreeIndex
# documents = SimpleDirectoryReader('data').load_data()
# new_index = GPTTreeIndex.from_documents(documents)
# query_engine = new_index.as_query_engine(
#     child_branch_factor=2
# )
# response = query_engine.query("What did the author do growing up?")
# documents = []
# books = [
#     "example_data/a.txt"
# ]
# for file in books:
#     with open(file, "r",encoding='utf-8') as f:
#         documents.append(Document(f.read(), doc_id=file))

# index = GPTSimpleVectorIndex.from_documents(documents)
# index_file_name = "index_my.json"
# # # save to disk
# # index.save_to_disk(index_file_name)
# ----------------
# index = GPTVectorStoreIndex.load_from_dict(index_file_name)
# history = []
# query_str = "为什么AI应用乐观？"



# response = query_engine.query(query_str)
# print(response)

# response = index.query("为什么AI应用乐观？") 
# print(response.response)
#---------------------------
# c=1
# #open index.json
# with open('index.json', 'r', encoding='utf-8') as f:
#     index = f.read()
# #convert index['nodes_dict'] to df
# import pandas as pd
# import json
# nodes_dict = json.loads(index)['index_struct']['nodes_dict']
# df = pd.DataFrame(nodes_dict)
# print(df)

from llama_index import StorageContext, load_index_from_storage

def qury_from_storage_index():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    QA_PROMPT_TMPL = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str},answer should be in Chinese beginning with 冀田认为，重写答案把它扩展到500字.\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)

    response = query_engine.query("流程再造在企业应用AI时候扮演什么角色?")
    print(response)  
    print(1)

def query_txt(query,llm):
    
    loader = TextLoader('example_data/a.txt',encoding='utf-8')
    
    index = VectorstoreIndexCreator().from_loaders([loader])
    query = "什么是通用人工智能？"
    print(query,index.query(query,llm))             

def query_txt_llama_index():
    
    # loader = TextLoader('example_data/a.txt',encoding='utf-8')
    documents = SimpleDirectoryReader('data').load_data()
    # documents = loader.load()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
    query_engine = index.as_query_engine()
    response = query_engine.query("传统企业如何应用AI?")
    print(response)  


# query_txt_llama_index()
qury_from_storage_index()

def query_url(query,llm,urls):
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()[0]['page_content']
    index = GPTSimpleVectorIndex(documents)
    index = VectorstoreIndexCreator().from_loaders([loader])
    print(query,index.query(query,llm))             




def save_db(urls):
    # loader = UnstructuredURLLoader(urls=urls)
    loader = TextLoader('example_data/a.txt',encoding='utf-8')
    documents = loader.load()
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    
    persist_directory='example_data/db3'
    vectordb = Chroma.from_documents(texts, embeddings)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
    query = "AI应用是否乐观?"
    print(qa.run(query))
urls0 = [
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023"
]

urls = [
    "https://www.yicai.com/news/101651452.html"
    # ,"https://www.yicai.com/author/1317.html"
]





# embeddings = OpenAIEmbeddings()
# persist_directory='example_data/db'
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# # qa=RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb)
# qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
# query = "what did russia do？"
# print(qa.run(query))
# retriever = db.as_retriever() 
# below works as a whole


# from langchain.document_loaders import UnstructuredWordDocumentLoader
# loader = UnstructuredWordDocumentLoader("example_data/family.docx")

# from unstructured.partition.auto import partition

# elements = partition("example_data/family.docx")
# print("\n\n".join([str(el) for el in elements]))
# data = loader.load()
    # index = VectorstoreIndexCreator().from_loaders([loader])
    # query = "俄罗斯提出了什么措施？"
    # print(query,index.query(query,llm))

c=1
# # coding=utf-8
# def check_charset(file_path):
#     import chardet
#     with open(file_path, "rb") as f:
#         data = f.read(4)
#         charset = chardet.detect(data)['encoding']
#     return charset
 
# your_path = 你的文件路径
# with open(your_path, encoding=check_charset(your_path)) as f:
#     data = f.read()
#     print(data)
# vectorstore = Chroma("langchain_store", embeddings.embed_query)