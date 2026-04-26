import streamlit as st
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def generate_response(uploaded_file, openai_api_key, query_text):
    """Build a temporary vector store from the uploaded text and answer a query."""
    if uploaded_file is None:
        return "Please upload a .txt file first."

    document_text = uploaded_file.read().decode("utf-8", errors="ignore")
    documents = [document_text]

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.create_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
    )

    return qa_chain.run(query_text)


st.set_page_config(page_title="Ask the Doc App")
st.title("Ask the Doc App")

uploaded_file = st.file_uploader("Upload a text article", type=["txt"])
query_text = st.text_input(
    "Enter your question",
    placeholder="Please provide a short summary.",
    disabled=not uploaded_file,
)

response = None
with st.form("ask_doc_form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        disabled=not (uploaded_file and query_text),
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text),
    )

    if submitted:
        if not openai_api_key.startswith("sk-"):
            st.error("Please enter a valid OpenAI API key.")
        else:
            with st.spinner("Calculating..."):
                response = generate_response(uploaded_file, openai_api_key, query_text)

if response:
    st.info(response)
