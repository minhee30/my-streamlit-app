
import streamlit as st

from assistant import ask_ai, summarize_meeting
from audio_utils import speech_to_text, text_to_speech
from email_utils import send_email


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="에이블 | AI 업무 비서",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. 기업용 디자인
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }

    h1, h2, h3 {
        color: #172B4D;
        letter-spacing: -0.5px;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. 공통 상태 및 보조 함수
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""

if "meeting_summary" not in st.session_state:
    st.session_state.meeting_summary = ""


def read_text_file(uploaded_file):
    if uploaded_file is None:
        return ""
    return uploaded_file.getvalue().decode("utf-8-sig")


def save_chat_history(question, answer):
    st.session_state.messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ])


# ============================================================
# 4. 사이드바
# ============================================================

with st.sidebar:
    st.markdown("## ✦ 에이블")
    st.caption("AIVLE | Enterprise AI Assistant")

    st.divider()
    st.markdown("##### WORKSPACE")

    selected_menu = st.radio(
        "기능 선택",
        [
            "✦ AI 채팅",
            "🎙️ 음성 AI",
            "📄 문서 분석",
            "📝 회의록 요약",
            "✉️ 이메일 발송",
            "🕘 대화 기록"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("AI 기반 업무 지원 서비스")


# ============================================================
# 5. AI 채팅
# ============================================================

if selected_menu == "✦ AI 채팅":

    st.title("✦ AI 채팅")
    st.caption("업무에 필요한 질문을 입력하고 AI의 도움을 받으세요.")
    st.divider()

    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("#### 에이블 AI")
            st.write("안녕하세요. 오늘 어떤 업무를 도와드릴까요?")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("업무 관련 질문을 입력하세요.")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("답변을 생성하고 있습니다..."):
                    answer = answer = ask_ai(question=question,history=st.session_state.messages)

                st.write(answer)
                st.session_state.answer = answer
                save_chat_history(question, answer)

            except Exception as e:
                st.error(f"AI 응답 오류: {e}")


# ============================================================
# 6. 음성 AI
# ============================================================

elif selected_menu == "🎙️ 음성 AI":

    st.title("🎙️ 음성 AI")
    st.caption("음성으로 질문하고 AI 답변을 확인하세요.")
    st.divider()

    with st.container(border=True):
        with st.form("voice_form"):
            voice_audio = st.audio_input("음성을 녹음해주세요.")
            voice_submit = st.form_submit_button(
                "음성 분석 시작",
                use_container_width=True
            )

    if voice_submit:
        if voice_audio is None:
            st.warning("음성을 먼저 녹음해주세요.")
        else:
            try:
                with st.spinner("음성을 인식하고 있습니다..."):
                    question = speech_to_text(voice_audio)

                st.session_state.transcript = question

                with st.container(border=True):
                    st.subheader("음성 인식 결과")
                    st.write(question)

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
# 7. 문서 분석
# ============================================================

elif selected_menu == "📄 문서 분석":

    st.title("📄 문서 분석")
    st.caption("TXT 문서를 업로드하고 내용을 질문하세요.")
    st.divider()

    with st.container(border=True):
        with st.form("document_form"):
            document_file = st.file_uploader(
                "문서 업로드",
                type=["txt"]
            )
            document_question = st.text_area(
                "질문 입력",
                placeholder="예: 이 문서의 핵심 내용을 요약해줘."
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
                document_text = read_text_file(document_file)
                st.session_state.meeting_text = document_text

                with st.spinner("문서를 분석하고 있습니다..."):
                    answer = ask_ai(
                        document_question,
                        text=document_text
                    )

                st.session_state.answer = answer
                save_chat_history(document_question, answer)

                with st.container(border=True):
                    st.subheader("분석 결과")
                    st.write(answer)

            except Exception as e:
                st.error(f"문서 분석 오류: {e}")


# ============================================================
# 8. 회의록 요약 및 TTS
# ============================================================

elif selected_menu == "📝 회의록 요약":

    st.title("📝 회의록 요약")
    st.caption("회의 내용을 분석하고 담당 업무를 정리합니다.")
    st.divider()

    with st.container(border=True):
        with st.form("meeting_form"):
            employee_name = st.text_input(
                "담당 업무를 확인할 직원 이름 (선택)"
            )

            meeting_date = st.date_input(
                "회의 날짜",
                value=None
            )

            meeting_file = st.file_uploader(
                "회의록 TXT 업로드",
                type=["txt"]
            )

            meeting_input = st.text_area(
                "회의 내용을 직접 입력할 수도 있습니다.",
                height=160
            )

            meeting_audio = st.audio_input(
                "또는 음성으로 요약을 요청하세요."
            )

            meeting_submit = st.form_submit_button(
                "회의록 요약 시작",
                use_container_width=True
            )

    if meeting_submit:
        try:
            meeting_text = meeting_input.strip()

            if meeting_file is not None:
                meeting_text = read_text_file(meeting_file)

            if not meeting_text:
                st.warning("회의 내용을 입력하거나 파일을 업로드해주세요.")
            else:
                st.session_state.meeting_text = meeting_text

                instruction = ""

                if meeting_audio is not None:
                    with st.spinner("음성 요청을 인식하고 있습니다..."):
                        instruction = speech_to_text(meeting_audio)

                    st.info(f"음성 요청: {instruction}")

                with st.spinner("회의록을 요약하고 있습니다..."):
                    summary = summarize_meeting(
                        meeting_text,
                        employee_name=employee_name or "김 대리",
                        meeting_date=meeting_date.isoformat() if meeting_date else None
                        )

                    # 음성 요청에서 추가 지시가 있다면 반영
                    if instruction.strip():
                        summary = ask_ai(
                            question=(
                                f"다음 회의록 요약에 대해 사용자의 요청을 "
                                f"반영해 다시 정리하세요.\n"
                                f"요청: {instruction}"
                            ),
                            text=summary
                        )

                st.session_state.meeting_summary = summary

        except Exception as e:
            st.error(f"회의록 처리 오류: {e}")

    if st.session_state.meeting_summary:
        with st.container(border=True):
            st.subheader("회의록 요약 결과")
            st.write(st.session_state.meeting_summary)

            st.download_button(
                "요약 결과 다운로드",
                data=st.session_state.meeting_summary,
                file_name="meeting_summary.txt",
                mime="text/plain"
            )

        if st.button("🔊 요약 내용 음성으로 듣기"):
            try:
                with st.spinner("음성을 생성하고 있습니다..."):
                    audio_bytes = text_to_speech(
                        st.session_state.meeting_summary
                    )

                st.audio(audio_bytes, format="audio/mp3")

            except Exception as e:
                st.error(f"TTS 오류: {e}")


# ============================================================
# 9. 이메일 발송
# ============================================================

elif selected_menu == "✉️ 이메일 발송":

    st.title("✉️ 이메일 발송")
    st.caption("수신자와 내용을 확인한 뒤 이메일을 발송하세요.")
    st.divider()

    with st.container(border=True):
        with st.form("email_form"):
            email_to = st.text_input(
                "받는 사람",
                placeholder="example@company.com"
            )

            email_subject = st.text_input("제목")

            email_body = st.text_area(
                "본문",
                height=220
            )

            email_submit = st.form_submit_button(
                "이메일 발송",
                use_container_width=True
            )

    if email_submit:
        if not all([
            email_to.strip(),
            email_subject.strip(),
            email_body.strip()
        ]):
            st.warning("수신자, 제목, 본문을 모두 입력해주세요.")
        else:
            with st.spinner("이메일을 발송하고 있습니다..."):
                result = send_email(
                    email_to.strip(),
                    email_subject,
                    email_body
                )

            if result["status"] == "success":
                st.success("이메일이 발송되었습니다.")
            else:
                st.error(
                    f"이메일 발송 실패: {result.get('error', '알 수 없는 오류')}"
                )


# ============================================================
# 10. 대화 기록
# ============================================================

elif selected_menu == "🕘 대화 기록":

    st.title("🕘 대화 기록")
    st.caption("현재 세션의 AI 대화를 확인하세요.")
    st.divider()

    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

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
