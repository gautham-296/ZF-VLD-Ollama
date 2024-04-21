import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import chromadb
import os
import time
import argparse
import warnings

warnings.filterwarnings('ignore')

def answer_query(query):
    model = os.environ.get("MODEL", "mistral")
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
    persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    llm = Ollama(model=model, callbacks=[StreamingStdOutCallbackHandler()])

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

    # Get the answer from the chain
    res = qa(query)
    answer = res['result']

    # Display Results
    st.write("Question:", query)
    st.write("Answer:", answer)


def main():
    st.title("Locally Trained LLM App")

    # User Input
    query = st.text_input("Enter a query:")

    if st.button("Ask"):
        if query:
            answer_query(query)


if __name__ == "__main__":
    main()
