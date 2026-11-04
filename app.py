
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
# 7. 추가 기능 연결 영역
# ============================================================

# 7-1. 이메일 발송
# 개별 기능 구현 파일 확인 후 연결

# 7-2. 회의록 요약
# 개별 기능 구현 파일 확인 후 연결

# 7-3. 음성 요청 기반 회의록 요약
# STT + 회의록 요약 기능 연결

# 7-4. 요약 결과 음성 출력 (TTS)
# 개별 기능 구현 파일 확인 후 연결

# 7-5. 음성 질문과 응답
# speech_to_text() + ask_ai() 활용

# 7-6. Function Calling
# 개별 기능 구현 파일 확인 후 연결
