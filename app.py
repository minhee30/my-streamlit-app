
# ============================================================
# 1. 라이브러리 및 OpenAI 클라이언트 설정
# ============================================================

import os
import base64
import json
import requests
import streamlit as st
from openai import OpenAI

client = OpenAI()


# ============================================================
# 2. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="에이블 | AI 업무 비서",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. 기업용 UI 스타일 설정
# ============================================================

st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background-color: #F8FAFC;
        color: #172B4D;
    }

    /* 메인 콘텐츠 */
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
# 4. AI 역할(System Prompt) 설정
# ============================================================

sys_role = """
당신은 '에이블(AIVLE)'이라는 이름의 AI 업무 비서입니다.

다양한 업무를 처리하는 AI 서비스이며,
회사원들의 업무 효율성을 높이는 것을 목표로 합니다.

사용자의 질문에 친절하고 정확하게 답변합니다.

업로드된 문서가 있다면 해당 내용을 참고하여 답변합니다.
문서에 없는 정보는 임의로 만들어내지 않습니다.

답변은 명확하고 전문적인 비즈니스 언어로 작성합니다.
"""


# ============================================================
# 5. 공통 기능: 세션 상태 관리
# ============================================================

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []

# AI 답변
if "answer" not in st.session_state:
    st.session_state.answer = ""

# 음성 인식 결과
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# 업로드한 문서
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""

# 회의록 요약 결과
if "meeting_summary" not in st.session_state:
    st.session_state.meeting_summary = ""


# ============================================================
# 6. 공통 기능: 기본 질문과 응답 (LLM)
# ============================================================

def ask_ai(question, text=""):

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

    response = client.responses.create(
        model="gpt-4o-mini",
        input=input_messages
    )

    return response.output_text


# ============================================================
# 7. 공통 기능: 텍스트 파일 읽기
# ============================================================

def read_text_file(uploaded_file):

    if uploaded_file is None:
        return ""

    return uploaded_file.getvalue().decode("utf-8-sig")


# ============================================================
# 8. 공통 기능: 음성 인식 (STT)
# ============================================================

def speech_to_text(audio_value):

    if audio_value is None:
        return ""

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
# 9. 공통 기능: AI 답변 출력
# ============================================================

def display_ai_answer(answer):

    st.subheader("AI 답변")
    st.write(answer)


# ============================================================
# 10. 공통 기능: 대화 기록 저장
# ============================================================

def save_chat_history(question, answer):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# ============================================================
# 11. 사이드바: 서비스명 및 기능 선택
# ============================================================

with st.sidebar:

    # 서비스명
    st.markdown("## ✦ 에이블")
    st.caption("AIVLE | Enterprise AI Assistant")

    st.divider()

    st.markdown("##### WORKSPACE")

    # 기능 선택
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
# 12. 메인 화면: AI 채팅
# ============================================================

if selected_menu == "AI 채팅":

    st.title("✦ AI 채팅")
    st.caption("업무에 필요한 질문을 입력하고 AI의 도움을 받으세요.")

    st.divider()

    # 첫 방문 안내
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("#### 에이블 AI")
            st.write("안녕하세요. 오늘 어떤 업무를 도와드릴까요?")

    # 기존 대화 표시
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 사용자 입력
    question = st.chat_input("업무 관련 질문을 입력하세요.")

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("답변을 생성하고 있습니다..."):

                try:
                    answer = ask_ai(question)

                    st.write(answer)

                    st.session_state.answer = answer
                    save_chat_history(question, answer)

                except Exception as e:
                    st.error(f"AI 응답 오류: {e}")


# ============================================================
# 13. 메인 화면: 음성 AI
# ============================================================

elif selected_menu == "음성 AI":

    st.title("🎙️ 음성 AI")
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
                # 13-1. 음성 인식
                with st.spinner("음성을 인식하고 있습니다..."):
                    question = speech_to_text(voice_audio)

                st.session_state.transcript = question

                # 13-2. 음성 인식 결과
                with st.container(border=True):
                    st.subheader("음성 인식 결과")
                    st.write(question)

                # 13-3. AI 답변
                if question.strip():

                    with st.spinner("답변을 생성하고 있습니다..."):
                        answer = ask_ai(question)

                    st.session_state.answer = answer
                    save_chat_history(question, answer)

                    with st.container(border=True):
                        display_ai_answer(answer)

                else:
                    st.warning("음성을 인식하지 못했습니다.")

            except Exception as e:
                st.error(f"음성 처리 오류: {e}")


# ============================================================
# 14. 메인 화면: 문서 분석
# ============================================================

elif selected_menu == "문서 분석":

    st.title("📄 문서 분석")
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
                # 14-1. 문서 읽기
                document_text = read_text_file(document_file)

                st.session_state.meeting_text = document_text

                # 14-2. LLM 호출
                with st.spinner("문서를 분석하고 있습니다..."):

                    answer = ask_ai(
                        question=document_question,
                        text=document_text
                    )

                st.session_state.answer = answer
                save_chat_history(document_question, answer)

                # 14-3. 분석 결과
                with st.container(border=True):

                    st.subheader("분석 결과")
                    st.write(answer)

            except Exception as e:
                st.error(f"문서 분석 오류: {e}")


# ============================================================
# 15. 메인 화면: 대화 기록
# ============================================================

elif selected_menu == "대화 기록":

    st.title("🕘 대화 기록")
    st.caption("현재 세션에서 진행한 AI 대화를 확인하세요.")

    st.divider()

    if st.session_state.messages:

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.write(message["content"])

        st.divider()

        # 대화 기록 초기화
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


# ============================================================
# 16. 추가 기능 연결 영역
# ============================================================

# 앞으로 별도 .py 파일에서 구현한 기능을 연결합니다.

# 16-1. 이메일 발송
# from email_service import send_email

# 16-2. 회의록 요약
# from meeting_service import summarize_meeting

# 16-3. 음성 요청 기반 회의록 요약
# speech_to_text() + summarize_meeting()

# 16-4. 텍스트 → 음성 변환 (TTS)
# from audio_service import text_to_speech

# 16-5. Function Calling
# from tools import execute_tool

# 16-6. 캘린더 일정 관리
# from calendar_service import create_event
