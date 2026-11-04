"""사용자 텍스트/음성 질문을 회의록과 멀티턴 맥락으로 처리합니다."""
from dataclasses import dataclass
from assistant import run_assistant
from audio_utils import transcribe_audio, text_to_speech

class MissingInputError(ValueError):
    pass

@dataclass
class CommandResult:
    user_prompt: str
    answer: str
    audio_bytes: bytes | None
    audio_error: str | None = None

def run_command(client, text_query: str, audio_query, meeting_content: str,
                sys_role: str, messages: list, meeting_summary: str = "") -> CommandResult:
    user_prompt = (transcribe_audio(client, audio_query)
                   if audio_query is not None else text_query.strip())
    if not user_prompt or not user_prompt.strip():
        raise MissingInputError("텍스트 입력 또는 음성 녹음이 필요합니다.")
    answer = run_assistant(
        client=client,
        sys_role=sys_role,
        conversation_history=messages,
        current_prompt=user_prompt,
        meeting_text=meeting_content,
        meeting_summary=meeting_summary,
    )
    try:
        audio_bytes = text_to_speech(client, answer)
        audio_error = None
    except Exception as exc:
        audio_bytes = None
        audio_error = str(exc)
    return CommandResult(user_prompt, answer, audio_bytes, audio_error)
