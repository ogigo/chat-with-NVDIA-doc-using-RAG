import streamlit as st
from chain import get_qa_chain
from langchain.vectorstores import FAISS
from embedding import instructor_embeddings



st.title("Question and Answer with NVIDIA documentation")


question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])