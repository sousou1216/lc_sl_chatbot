# 必要なライブラリの準備
import os

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import tempfile # PDFアップロードの際に必要


# DBの保存先の設定
folder_name = "./.data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# ストリーム表示
class StreamCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens_area = st.empty()
        self.tokens_stream = ""

    def on_llm_new_token(self, token, **kwargs):
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

# UI周り
st.title("QA")
uploaded_file = st.file_uploader("Upload a file after paste OpenAI API key", type="pdf")
    
with st.sidebar:
    user_api_key = st.text_input(
        label="OpenAI API key",
        placeholder="Paste your openAI API key",
        type="password"
    )
    os.environ['OPENAI_API_KEY'] = user_api_key
    select_model = st.selectbox("Model", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview",])
    select_temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1,)
    select_chunk_size = st.slider("Chunk", min_value=0.0, max_value=1000.0, value=300.0, step=10.0,)

# PDFのアップロード後実行
if uploaded_file:
    # 一時ファイルにPDFを書き込みバスを取得
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 読み取り
    loader = PyMuPDFLoader(file_path=tmp_file_path) 
    documents = loader.load() 

    # 分割
    text_splitter = SpacyTextSplitter(
        chunk_size=select_chunk_size,
        pipeline="ja_core_news_sm",
    )

    data = text_splitter.split_documents(documents)

    # ベクトル化とDBへの保存
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )

    database = Chroma(
        persist_directory="./.data",
        embedding_function=embeddings,
    )

    database.add_documents(data)

    # モデルの設定
    chat = ChatOpenAI(
        model=select_model,
        temperature=select_temperature,
        streaming=True,
    )

    # retrieverに変換（検索、プロンプトの構築）
    retriever = database.as_retriever()

    # 会話履歴を初期化
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    # 会話履歴をメモリに保持
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=retriever,
        memory=st.session_state.memory,
    )

    # UI用の会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # UI用の会話履歴を表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # UI周り
    prompt = st.chat_input("Ask something about the file.")

    if prompt:
        # UI用の会話履歴に追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = chain(
                {"question": prompt},
                callbacks=[StreamCallbackHandler()], # ストリーム表示
            )
        
        # UI用の会話履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    # メモリの内容をターミナルで確認
    print(st.session_state.memory)



