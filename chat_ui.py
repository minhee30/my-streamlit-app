"""회의록 후속 질문을 위한 멀티턴 대화 UI."""
import streamlit as st
from openai import OpenAI
from app_state import save_response
from command_service import MissingInputError, run_command

def render_chat(meeting_content: str, sys_role: str,
                audio_query=None, audio_submitted: bool = False) -> None:
    st.subheader("💬 회의록에 이어 질문하기")
    st.caption("회의록과 앞선 질문·답변을 기억하며 대화합니다. 대화 기록은 현재 세션에만 보관됩니다.")

    if st.button("회의록 대화 초기화", key="clear_meeting_chat"):
        st.session_state.meeting_messages = []
        st.rerun()

    for message in st.session_state.meeting_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "예: 김 대리의 업무 중 이번 주 마감인 것은?",
        key="meeting_chat_input",
    )
    if not prompt and not audio_submitted:
        if not st.session_state.meeting_messages:
            st.info("회의록을 업로드한 뒤 추가 질문을 입력해 보세요.")
        return
    if audio_submitted and audio_query is None:
        st.warning("음성을 먼저 녹음해 주세요.")
        return

    with st.spinner("회의록과 이전 대화를 참고해 답변 중입니다..."):
        try:
            result = run_command(
                client=OpenAI(timeout=60.0),
                text_query=prompt or "",
                audio_query=audio_query if audio_submitted else None,
                meeting_content=meeting_content,
                sys_role=sys_role,
                messages=st.session_state.meeting_messages,
                meeting_summary=st.session_state.meeting_summary,
            )
        except MissingInputError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:
            st.error(f"질문 처리 오류: {exc}")
            return

    st.session_state.meeting_messages.extend([
        {"role": "user", "content": result.user_prompt},
        {"role": "assistant", "content": result.answer},
    ])
    with st.chat_message("user"):
        st.write(result.user_prompt)
    with st.chat_message("assistant"):
        st.markdown(result.answer)
        if result.audio_bytes:
            st.audio(result.audio_bytes, format="audio/mp3")
        if result.audio_error:
            st.caption(f"음성 답변 생성 오류: {result.audio_error}")
