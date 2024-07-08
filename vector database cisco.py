


from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate




import pandas as pd

import yaml

from pprint import pprint






OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']




# PDF Loader 




import os

from os import getcwd

getcwd()




from langchain_community.document_loaders import PyPDFDirectoryLoader




loader = PyPDFLoader("cisco_pdfs/Q2 Cisco Report.pdf")

loader1= PyPDFLoader("cisco_pdfs/Q3 Cisco Report.pdf")




# THIS TAKES 5 MINUTES...

documents = loader.load()




CHUNK_SIZE = 1500 #Try not to go over 2000. 




text_splitter = CharacterTextSplitter(

    chunk_size=CHUNK_SIZE, 

    # chunk_overlap=100,

    separator="\n"

)




docs = text_splitter.split_documents(documents)




docs




len(docs) #This is about 2 or 3 documents per page as there are 1723 documents. 




docs[0] #This is first document. 




pprint(dict(docs[5])["page_content"]) #If we give this text to an LLM model, can it read it? GPT-4.0 model can read it.
 




docs[5]




# THIS TAKES 5 MINUTES...

documents1 = loader1.load()




CHUNK_SIZE = 1000 #Try not to go over 2000. 




text_splitter = CharacterTextSplitter(

    chunk_size=CHUNK_SIZE, 

    # chunk_overlap=100,

    separator="\n"

)




docs1 = text_splitter.split_documents(documents1)




docs1




len(docs1) #This is about 2 or 3 documents per page as there are 1723 documents. 




docs1[0] #This is first document. 




pprint(dict(docs1[5])["page_content"]) #If we give this text to an LLM model, can it read it? GPT-4.0 model can read it.
 




docs1[5]




# Vector Database




embedding_function = OpenAIEmbeddings(

    model='text-embedding-ada-002',

    api_key=OPENAI_API_KEY

)




#Use Chroma from documents and provide it docs, directory and embedding function.
 




vectorstore = Chroma.from_documents(

    docs, 

    persist_directory="cisco_earnings_statement_challenge/data/chroma_learning.db",

    embedding=embedding_function

)




vectorstore = Chroma.from_documents(

    docs1, 

    persist_directory="cisco_earnings_statement_challenge/data/chroma_learning.db",

    embedding=embedding_function

)




retriever = vectorstore.as_retriever() #We create retriever from vector store. 




retriever




# RAG LLM Model




#The RAG chain is integrated into the streamlit app. 




template = """Answer the question based only on the following context:

{context}




Question: {question}

"""




prompt = ChatPromptTemplate.from_template(template)




model = ChatOpenAI(

    model = 'gpt-3.5-turbo',

    temperature = 1,

    api_key=OPENAI_API_KEY

)




rag_chain = (

    {"context": retriever, "question": RunnablePassthrough()}

    | prompt

    | model

    | StrOutputParser()

)




result = rag_chain.invoke("What was Cisco total revenues in 2nd quarter?")




pprint(result)




result1 = rag_chain.invoke("What was Cisco total revenues in 3rd quarter?")

pprint(result1)
