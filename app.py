import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

# Set up API key
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
os.environ["GOOGLE_API_KEY"] = GEMINI_KEY

# Streamlit UI
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Set up the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_KEY,
    temperature=0.6,
    max_tokens=600,
    timeout=None,
    max_retries=2,
)

# Process URLs and build FAISS index
if process_url_clicked:
    main_placeholder.text("Loading data from URLs... ✅")
    filtered_urls = [url for url in urls if url.startswith(('http://', 'https://'))]
    loader = WebBaseLoader(filtered_urls)
    data = loader.load()

    main_placeholder.text("Splitting text into chunks... ✅")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_documents(data)

    main_placeholder.text("Generating embeddings... ✅")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    main_placeholder.text("Saving vector store to disk... ✅")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local("my_faiss_index")

# Accept user query
query = main_placeholder.text_input("Ask a question based on the articles:")
if query:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(
        "my_faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    # Use the default "stuff" chain
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    result = chain({"question": query}, return_only_outputs=True)

    # Display result
    st.header("Answer")
    st.write(result["answer"])

    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        for source in sources.split("\n"):
            st.write(source)
