
import os, base64, json, requests
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="음성 AI 서비스", page_icon="🤖", layout="wide")
st.title("음성 인식 AI 서비스")
st.write("음성으로 질문하시면 답변해 드립니다.")

# 역할 지정
sys_role = """
당신은 다양한 업무를 처리하는 AI 서비스입니다.
개인 비서처럼 친절하게 응대합니다.
"""

# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")

# 버튼을 눌렀을 때만 실행
if submit:
    # 업로드 된 파일이 있으면 디코딩
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 녹음이 없으면 녹음 요청
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    # 시스템 메세지
    system_content = sys_role
    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"
    input_messages = [{"role": "system", "content": system_content}]

    # STT: 음성 → 텍스트
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_value, "audio/wav"))

    # 사용자 메시지
    input_messages.append({"role": "user", "content": transcript.text})

    # LLM 호출 및 응답 듣기
    with st.spinner("답변을 준비하고 있습니다..."):
        # LLM 호출
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages
        )
        answer = response.output_text

        # 응답 출력
        st.subheader("AI 답변")
        st.write(answer)

    # ============================================================
# 1. 공통 기능: 세션 상태 관리
# ============================================================

# 기존 코드의 messages 초기화는 그대로 사용

# 음성 인식 결과 저장
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# AI 답변 저장
if "answer" not in st.session_state:
    st.session_state.answer = ""

# 업로드한 텍스트 저장
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""

# 회의록 요약 결과 저장
if "meeting_summary" not in st.session_state:
    st.session_state.meeting_summary = ""

    
# ============================================================
# 2. 공통 기능: 기본 질문과 응답 (LLM)
# ============================================================

def ask_ai(question, text=""):

    # 기존 코드의 시스템 메시지 구성 방식 사용
    system_content = sys_role

    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"

    input_messages = [
        {"role": "system", "content": system_content}
    ]

    input_messages.append({
        "role": "user",
        "content": question
    })

    # 기존 코드와 동일한 LLM 호출
    response = client.responses.create(
        model="gpt-4o-mini",
        input=input_messages
    )

    answer = response.output_text

    return answer


# ============================================================
# 3. 공통 기능: 텍스트 파일 읽기
# ============================================================

def read_text_file(uploaded_file):

    text = ""

    # 기존 코드의 디코딩 방식 사용
    if uploaded_file is not None:
        text = uploaded_file.getvalue().decode("utf-8")

    return text


# ============================================================
# 4. 공통 기능: 음성 인식 (STT)
# ============================================================

def speech_to_text(audio_value):

    if audio_value is None:
        return ""

    # 기존 코드와 동일한 Whisper 모델 사용
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=(
            "audio.wav",
            audio_value.getvalue(),
            "audio/wav"
        )
    )

    return transcript.text


# ============================================================
# 5. 공통 기능: AI 답변 출력
# ============================================================

def display_ai_answer(answer):

    # 기존 Streamlit 출력 방식 사용
    st.subheader("AI 답변")
    st.write(answer)


# ============================================================
# 6. 공통 기능: 대화 기록 관리
# ============================================================

def save_chat_history(question, answer):

    # 기존 코드의 messages 세션 상태 활용
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# ============================================================
# 1. 기업용 UI 스타일 설정
# ============================================================

st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background-color: #F8FAFC;
        color: #172B4D;
    }

    /* 메인 화면 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1.5rem;
        max-width: 1100px;
    }

    /* 사이드바 */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }

    /* 제목 */
    h1, h2, h3 {
        color: #172B4D;
        font-weight: 650;
        letter-spacing: -0.5px;
    }

    /* 버튼 */
    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* 입력 영역 */
    .stTextInput input,
    .stTextArea textarea {
        border-radius: 8px;
    }

    /* 카드 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. 공통 세션 상태 초기화
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""

if "meeting_summary" not in st.session_state:
    st.session_state.meeting_summary = ""


# ============================================================
# 3. 사이드바: 기능 선택
# ============================================================

with st.sidebar:

    st.markdown("## WORKSPACE AI")
    st.caption("Enterprise AI Assistant")

    st.divider()

    st.markdown("##### WORKSPACE")

    selected_menu = st.radio(
        "기능 선택",
        [
            "AI 채팅",
            "음성 AI",
            "문서 분석",
            "대화 기록"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("AI 기반 업무 지원 서비스")
    st.caption("Powered by OpenAI")


# ============================================================
# 4. 메인 화면: AI 채팅
# ============================================================

if selected_menu == "AI 채팅":

    st.title("AI 채팅")
    st.caption("업무에 필요한 질문을 입력하고 AI의 도움을 받으세요.")

    st.divider()

    # 기존 대화 표시
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 채팅 입력
    question = st.chat_input("업무 관련 질문을 입력하세요.")

    if question:

        # 사용자 질문 표시
        with st.chat_message("user"):
            st.write(question)

        # AI 답변
        with st.chat_message("assistant"):

            with st.spinner("답변을 생성하고 있습니다..."):

                try:
                    answer = ask_ai(question)

                    st.write(answer)

                    # 세션에 저장
                    st.session_state.answer = answer
                    save_chat_history(question, answer)

                except Exception as e:
                    st.error(f"AI 응답 오류: {e}")


# ============================================================
# 5. 메인 화면: 음성 AI
# ============================================================

elif selected_menu == "음성 AI":

    st.title("음성 AI")
    st.caption("음성으로 질문하고 AI의 답변을 확인하세요.")

    st.divider()

    with st.container(border=True):

        st.subheader("음성 입력")

        with st.form("voice_form"):

            voice_audio = st.audio_input(
                "음성을 녹음해주세요.",
                key="enterprise_voice"
            )

            voice_submit = st.form_submit_button(
                "음성 분석 시작",
                use_container_width=True
            )

    if voice_submit:

        if voice_audio is None:
            st.warning("음성을 먼저 녹음해주세요.")

        else:
            try:
                # STT
                with st.spinner("음성을 인식하고 있습니다..."):
                    question = speech_to_text(voice_audio)

                st.session_state.transcript = question

                # 인식 결과
                with st.container(border=True):
                    st.subheader("음성 인식 결과")
                    st.write(question)

                # AI 응답
                if question.strip():

                    with st.spinner("답변을 생성하고 있습니다..."):
                        answer = ask_ai(question)

                    st.session_state.answer = answer
                    save_chat_history(question, answer)

                    with st.container(border=True):
                        st.subheader("AI 답변")
                        st.write(answer)

                else:
                    st.warning("음성을 인식하지 못했습니다.")

            except Exception as e:
                st.error(f"음성 처리 오류: {e}")


# ============================================================
# 6. 메인 화면: 문서 분석
# ============================================================

elif selected_menu == "문서 분석":

    st.title("문서 분석")
    st.caption("업무 문서를 업로드하고 관련 내용을 질문하세요.")

    st.divider()

    with st.container(border=True):

        st.subheader("문서 업로드 및 질문")

        with st.form("document_form"):

            document_file = st.file_uploader(
                "TXT 문서를 업로드하세요.",
                type=["txt"],
                key="enterprise_document"
            )

            document_question = st.text_area(
                "문서에 대해 질문하세요.",
                placeholder="예: 이 회의의 주요 결정사항을 정리해줘."
            )

            document_submit = st.form_submit_button(
                "문서 분석 시작",
                use_container_width=True
            )

    if document_submit:

        if document_file is None:
            st.warning("문서를 업로드해주세요.")

        elif not document_question.strip():
            st.warning("질문을 입력해주세요.")

        else:
            try:
                # 문서 읽기
                document_text = read_text_file(document_file)

                st.session_state.meeting_text = document_text

                # LLM 호출
                with st.spinner("문서를 분석하고 있습니다..."):

                    answer = ask_ai(
                        question=document_question,
                        text=document_text
                    )

                st.session_state.answer = answer
                save_chat_history(document_question, answer)

                # 결과 표시
                with st.container(border=True):

                    st.subheader("분석 결과")
                    st.write(answer)

            except Exception as e:
                st.error(f"문서 분석 오류: {e}")


# ============================================================
# 7. 메인 화면: 대화 기록
# ============================================================

elif selected_menu == "대화 기록":

    st.title("대화 기록")
    st.caption("현재 세션에서 진행한 AI 대화를 확인하세요.")

    st.divider()

    if st.session_state.messages:

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.write(message["content"])

        st.divider()

        if st.button(
            "대화 기록 초기화",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.session_state.answer = ""
            st.session_state.transcript = ""
            st.rerun()

    else:
        st.info("아직 저장된 대화 기록이 없습니다.")
