

import os
import hashlib
import tempfile
from operator import itemgetter

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="AIVLE School 학습 도우미",
    page_icon="📘",
    layout="wide"
)

st.title("📘 AIVLE School 학습 도우미")
st.caption("AIVLE School 교육과정부터 프로젝트, 취업 정보까지 함께 알아보세요!")


# ==========================================
# 2. 문서 포맷터
# ==========================================

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ==========================================
# 3. RAG 파이프라인
# 어제 작성한 함수 기반
# ==========================================

@st.cache_resource(show_spinner=False)
def rag_pipeline(filepath, file_hash=None):

    # PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Embedding 모델
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 문서별 벡터 저장소 경로 분리
    if file_hash:
        db_dir = os.path.join("uploaded_chroma_db", file_hash)
    else:
        db_dir = "chroma_db"

    # 벡터 저장소 만들기
    vectorstore = Chroma(
        collection_name="aivle_docs",
        embedding_function=embedding,
        persist_directory=db_dir
    )

    # 기존 데이터가 없을 때만 벡터화
    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(chunks)

    # 검색기 구성
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # LLM 선언
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    # 시스템 메시지
    s_msg = """
    당신은 KT AIVLE School의 교육과정, 지원 절차, 교육 운영,
    프로젝트 및 취업 정보를 안내하는 AI 상담 챗봇입니다.

    당신의 역할은 KT AIVLE School에 관심 있는 예비 지원자와
    교육생의 질문에 정확하고 이해하기 쉽게 답변하는 것입니다.

    [답변 원칙]
    1. 반드시 제공된 [참고 문서]에 근거하여 답변하세요.
    2. 참고 문서에 없는 정보는 임의로 생성하거나 추측하지 마세요.
    3. 문서에서 답변을 찾을 수 없다면
       "제공된 자료에서는 해당 내용을 확인할 수 없습니다."라고 답변하세요.
    4. 공식 교육 운영 정보와 개인 교육생의 후기를 명확히 구분하세요.
    5. 교육생의 실명이나 개인 식별 정보는 답변에 포함하지 마세요.
    6. 질문에 직접적으로 답하고 필요한 경우 세부 내용을 설명하세요.

    [기수 및 시점 구분]
    1. 교육 일정, 지원 자격, 지원금, 출결 규정 등은
       문서에 명시된 적용 기수와 시점을 기준으로 답변하세요.
    2. 10기 정보를 다른 기수에도 동일하게 적용된다고 단정하지 마세요.
    3. 예정된 일정과 실제 완료된 활동을 구분하세요.
    4. 최신 정보가 필요한 경우 공식 홈페이지 확인을 안내하세요.

    [교육후기 및 취업후기]
    1. 교육생의 경험은 개별 사례로 소개하세요.
    2. 개인의 경험을 전체 교육생의 공통된 결과로 일반화하지 마세요.
    3. 취업 기업, 직무, 프로젝트 성과는 문서에 명시된 사실만 사용하세요.
    4. 취업 사례를 취업 보장이나 취업률로 해석하지 마세요.

    [학습 도우미 역할]
    1. 사용자가 학습에 어려움을 느낀다면 친절하게 설명하세요.
    2. 학습을 포기하지 않고 적극적으로 도전할 수 있도록 격려하세요.
    3. 교육과정이나 프로젝트를 설명할 때
       필요한 경우 학습 방향을 함께 제안하세요.
    4. 사용자의 질문을 존중하고 긍정적인 학습 분위기를 조성하세요.
    5. 단, 문서에 없는 교육 운영 사실을 만들어내지 마세요.

    [답변 형식]
    1. 모든 답변은 자연스러운 한국어로 작성하세요.
    2. 핵심 내용을 먼저 설명하고 필요한 경우 항목별로 정리하세요.
    3. 여러 트랙이나 제도를 비교할 때는 표 또는 목록을 활용하세요.
    4. 전문 용어는 예비 지원자도 이해할 수 있도록 설명하세요.
    5. 문서에 없는 출처나 페이지 번호를 만들어내지 마세요.

    [참고 문서]
    {context}
    """

    h_msg = "{input}"

    messages = [
        ("system", s_msg),
        ("human", h_msg)
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    # RAG 체인 구성
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ==========================================
# 4. 기존 백서 RAG 체인 불러오기
# ==========================================

@st.cache_resource(show_spinner=False)
def load_default_chain():

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name="aivle_docs",
        embedding_function=embedding,
        persist_directory="chroma_db"
    )

    if vectorstore._collection.count() == 0:
        raise ValueError("기본 백서 벡터 DB가 비어 있습니다.")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # 기존 DB는 재벡터화하지 않고 검색기만 연결
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    s_msg = """
    당신은 KT AIVLE School 학습 도우미입니다.

    반드시 참고 문서에 근거하여 정확하게 답변하세요.
    문서에 없는 내용은 추측하지 마세요.
    확인할 수 없는 내용은 자료에서 확인할 수 없다고 안내하세요.
    교육생이 적극적으로 학습하고 도전하도록 격려하세요.
    공식 정보와 개인 후기를 구분하고,
    기수별 교육 정보를 일반화하지 마세요.
    교육생의 개인정보는 공개하지 마세요.

    [참고 문서]
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", s_msg),
        ("human", "{input}")
    ])

    chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ==========================================
# 5. 세션 상태 초기화
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "document_name" not in st.session_state:
    st.session_state.document_name = "AIVLE School 백서"


# ==========================================
# 6. 사이드바
# ==========================================

with st.sidebar:

    st.image(
        "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png",
        width=160
    )

    st.header("📂 문서 관리")

    uploaded_file = st.file_uploader(
        "PDF 문서를 업로드하세요.",
        type=["pdf"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        # 동일 문서 확인용 해시
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if st.session_state.document_hash != file_hash:

            with st.spinner("문서를 분석하고 있습니다..."):

                # 임시 PDF 파일 생성
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                try:
                    # 기존 rag_pipeline 사용
                    new_chain = rag_pipeline(
                        tmp_path,
                        file_hash
                    )

                    st.session_state.chain = new_chain
                    st.session_state.document_hash = file_hash
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.messages = []

                finally:
                    os.remove(tmp_path)

            st.success("문서 분석이 완료되었습니다!")

    st.divider()

    st.subheader("📄 현재 문서")
    st.info(st.session_state.document_name)

    if st.button("📘 기본 백서로 돌아가기", use_container_width=True):

        st.session_state.chain = None
        st.session_state.document_hash = None
        st.session_state.document_name = "AIVLE School 백서"
        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.subheader("💡 추천 질문")

    recommended_questions = [
        "AIVLE School의 교육과정을 설명해 주세요.",
        "미니 프로젝트는 어떻게 진행되나요?",
        "DX 트랙에서는 무엇을 배우나요?",
        "AIVLE School의 취업 지원은 어떤 것이 있나요?"
    ]

    selected_question = None

    for question in recommended_questions:
        if st.button(question, use_container_width=True):
            selected_question = question

    st.divider()

    if st.button("🗑️ 대화 삭제", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# 7. RAG 체인 준비
# ==========================================

try:

    if st.session_state.chain is None:
        chain = load_default_chain()
    else:
        chain = st.session_state.chain

except Exception as e:
    st.error(f"RAG 시스템을 불러오지 못했습니다: {e}")
    st.stop()


# ==========================================
# 8. 채팅 인터페이스
# ==========================================

st.subheader("💬 AI 학습 상담")

st.write(
    "AIVLE School에 대해 궁금한 점을 질문해 보세요. "
    "업로드한 문서를 기반으로 답변해 드립니다."
)

# 이전 대화 출력
for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        role = "user"
    else:
        role = "assistant"

    with st.chat_message(role):
        st.markdown(message.content)


# ==========================================
# 9. 사용자 질문 및 AI 응답
# ==========================================

user_input = st.chat_input("질문을 입력하세요.")

# 추천 질문 선택 시 자동 입력
if selected_question:
    user_input = selected_question

if user_input:

    # 사용자 메시지 저장
    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답
    with st.chat_message("assistant"):

        try:
            response_stream = chain.stream(
                {"input": user_input}
            )

            full_response = st.write_stream(response_stream)

            # AI 응답 저장
            st.session_state.messages.append(
                AIMessage(content=full_response)
            )

        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")




