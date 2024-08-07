from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain

import streamlit as st
import os
from tempfile import NamedTemporaryFile
import yaml

# Load API Key
import os
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# 1.0 LOAD AND SUMMARIZE FUNCTION
def load_and_summarize(file, use_template = False):
    
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: #We have to store this file somewhere. Named temporary file creates a temporary file that is automatically deleted. If delete = false, the file is not automatically deleted when the python session is closed. 
        tmp.write(file.getvalue()) #This is file name and the value of it. 
        file_path = tmp.name
    
    try: #Try and finally. Try clause attempts to execute a chunk of code. If the code executes, then finally clause will run after try block. 
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        if use_template: #If they want the bullet points. 
            #Bullets
            
            prompt_template = """
            Write a concise summary of the following:
            {text}

            Use 3 to 7 numbered bullet points to describe key points.
            """

            prompt = PromptTemplate(input_variables = ["text"], template=prompt_template)

            llm_chain = LLMChain(prompt=prompt, llm=model)

            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            response = stuff_chain.invoke(docs)
        
        else: 
            # No bullets
            summarizer_chain = load_summarize_chain(llm = model, chain_type="stuff")
            
            response = summarizer_chain.invoke(docs)
        
    finally:
        
        os.remove(file_path)
        
    return response['output_text']


# 2.0 STREAMLIT INTERFACE
st.title("PDF Earnings Call Summarizer")

st.subheader("Upload a PDF Document:")
uploaded_file = st.file_uploader("Choose a file", type="pdf")

if uploaded_file is not None:
    
    use_template = st.checkbox("Use numbered bullet points? (if not a paragraph will be returned)")
    
    if st.button("Summarize Document"):
        with st.spinner("Summarizing..."):
            
            summary = load_and_summarize(uploaded_file, use_template)
            
            st.subheader("Summarization Results:")
            st.markdown(summary)
         
else: 
    
    st.write("No file uploaded. Please upload a PDF file to proceed.")
