import streamlit as st
import pandas as pd
import os
#from apikey import apikey
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import base64  # Import the base64 module

st.set_page_config(layout="wide", page_icon="💬", page_title="HKSTP🤖")

st.markdown(
    f"""
    <h1 style='text-align: center;'> Matching Tools-Companies, Solutions & Investors</h1>
    """,
    unsafe_allow_html=True,
)

# Sidebar inputs
#apikey = st.sidebar.text_input("Please input the API_KEY:", type='password')
# Set the environment variable

api_key = os.environ.get('API_KEY')
os.environ['OPENAI_API_KEY'] = api_key 

# New dropdown for matching options
matching_option = st.sidebar.selectbox("Matching", ["Company Matching", "Funds Matching", "Solution Matching"])
Key_words = st.sidebar.text_input("Please input the key words:", key='key_words_input')
Top_key = st.sidebar.number_input("Please input the top number:", min_value=1,key='top_number_input')


# Load different indexes and dataframes based on matching option
if matching_option == "Company Matching":
    db = FAISS.load_local("index_hkstp_new", OpenAIEmbeddings())
    df = pd.read_csv("hkstp_copmany_with_country.csv")
    default_columns = ['name_EN', 'introduction_EN', 'product_EN', 'website','country_region']
    
    
    
elif matching_option == "Funds Matching":
    db = FAISS.load_local("index_funds", OpenAIEmbeddings())
    df = pd.read_csv("3_investor.csv")
    default_columns = ['INVESTOR', '英文名称', '主要投资领域','管理规模(记录币种)', '品牌介绍','机构官网']
    
    
elif matching_option == "Solution Matching":
    db = FAISS.load_local("index_solutions", OpenAIEmbeddings())
    df = pd.read_csv("2_solutions.csv")
    default_columns = ['Title', 'web_content', 'Link', 'Institute']



# Columns to select for display
all_columns = df.columns.tolist()
columns_to_display = st.sidebar.multiselect("Select columns to display:", all_columns, default=default_columns)

# Initialize retriever
retriever = db.as_retriever(search_kwargs={"k": Top_key})

if Key_words:
    row = []
    docs = retriever.get_relevant_documents(Key_words)
    
    if len(docs) == 0:
        st.write("No relevant documents found.")
    else:
        for i in range(min(Top_key, len(docs))):
            num = docs[i].metadata['row']
            row.append(num)
        
        result = df.loc[row][columns_to_display]
        
        st.write("Here is the result DataFrame:")
        st.write(result)


# Function to download dataframe as CSV
        def download_link(object_to_download, download_filename, download_link_text):
           if isinstance(object_to_download, pd.DataFrame):
              object_to_download = object_to_download.to_csv(index=False)
           b64 = base64.b64encode(object_to_download.encode()).decode()  # Use the base64 module here
           return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

        # Download link for dataframe
        if st.button('Download Data as CSV'):
            tmp_download_link = download_link(result, 'result.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
