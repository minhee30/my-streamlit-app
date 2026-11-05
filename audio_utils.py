# (출처: 6 - 미션 4. STT / 6 - 미션 5. TTS / 7 - 2. 음성 처리 로직)
import array
import io
import re
import wave

from openai import OpenAI

# 정확도가 높은 모델부터 시도하고, 결과가 한국어가 아니면 다음 모델로 다시 시도합니다.
STT_MODELS = ("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1")

# 인식 힌트: 무엇에 관한 말인지 알려 주면 엉뚱한 단어로 바뀌는 일이 줄어듭니다.
# (예시 문장을 넣으면 무음일 때 그 문장을 그대로 돌려주는 일이 있어 설명만 넣었습니다.)
STT_HINTS = {
    "command": "한국어로 말하는 AI 개인비서 에이블에게 하는 질문이나 명령입니다. 회의, 일정, 할 일, 이메일에 관한 내용입니다.",
    "schedule": "한국어로 말하는 회사 업무 일정입니다. 날짜, 요일, 시간과 회의, 미팅, 발표, 마감, 제출 같은 단어가 나옵니다.",
    "todo": "한국어로 말하는 회사 업무 할 일입니다. 자료 준비, 보고서 작성, 메일 보내기 같은 업무 표현이 나옵니다.",
}

_HAN = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")   # 한자·일본어 문자
_HANGUL = re.compile(r"[가-힣]")


def _audio_bytes(audio_file) -> bytes:
    """st.audio_input 결과(UploadedFile), 파일 객체, bytes 모두 bytes로 바꿉니다."""
    if isinstance(audio_file, (bytes, bytearray)):
        return bytes(audio_file)
    if hasattr(audio_file, "getvalue"):
        return audio_file.getvalue()
    if hasattr(audio_file, "seek"):
        audio_file.seek(0)
    return audio_file.read()


def _file_tuple(data: bytes):
    """녹음 파일의 실제 형식에 맞는 이름과 MIME 타입을 붙입니다."""
    if data[:4] == b"RIFF":
        return ("audio.wav", data, "audio/wav")
    if data[:4] == b"\x1aE\xdf\xa3":
        return ("audio.webm", data, "audio/webm")
    if data[:4] == b"OggS":
        return ("audio.ogg", data, "audio/ogg")
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return ("audio.mp3", data, "audio/mpeg")
    if data[4:8] == b"ftyp":
        return ("audio.m4a", data, "audio/mp4")
    return ("audio.wav", data, "audio/wav")


def _check_wav(data: bytes) -> None:
    """너무 짧거나 거의 무음인 녹음을 걸러 냅니다.
    무음을 인식시키면 모델이 중국어나 엉뚱한 문장을 지어내는 경우가 많습니다."""
    if data[:4] != b"RIFF":
        return
    try:
        with wave.open(io.BytesIO(data)) as w:
            seconds = w.getnframes() / float(w.getframerate())
            width = w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except Exception:
        return
    if seconds < 0.7:
        raise ValueError("녹음이 너무 짧습니다. 마이크 버튼을 누르고 1초 이상 말한 뒤 멈춰 주세요.")
    if width == 2 and frames:
        samples = array.array("h", frames[: len(frames) - len(frames) % 2])
        peak = max((abs(s) for s in samples[::4]), default=0)
        if peak < 500:   # 최대 32767 중 매우 작은 소리
            raise ValueError("목소리가 거의 들리지 않습니다. 마이크 권한·입력 장치를 확인하고 조금 더 크게 말해 주세요.")


def looks_korean(text: str) -> bool:
    """한자·일본어가 없고, 글자 중 한글이 절반 이상이면 정상적인 한국어 인식으로 봅니다.
    ('PPT', 'KPI' 같은 영어 단어가 조금 섞이는 것은 허용합니다.)"""
    if _HAN.search(text):
        return False
    letters = re.findall(r"[^\W\d_]", text)
    return bool(letters) and len(_HANGUL.findall(text)) / len(letters) >= 0.5


def transcribe_audio(client: OpenAI, audio_file, mode: str = "command") -> str:
    """
    사용자의 음성 입력(audio_input 또는 오디오 파일)을 한국어 텍스트로 변환(STT)합니다.
    mode: 'command'(사이드바 질문) / 'schedule'(상세 일정) / 'todo'(To-do)
    """
    data = _audio_bytes(audio_file)
    if not data or len(data) < 1000:
        raise ValueError("녹음된 소리가 없습니다. 다시 녹음해 주세요.")
    _check_wav(data)

    hint = STT_HINTS.get(mode, STT_HINTS["command"])
    fallback, last_error, got_text = "", None, False
    for model in STT_MODELS:
        try:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=_file_tuple(data),
                language="ko",      # 한국어로 고정
                prompt=hint,        # 업무 상황 힌트
                temperature=0,      # 지어내지 않도록 가장 보수적으로
            )
        except Exception as exc:
            last_error = exc
            continue
        text = (transcript.text or "").strip()
        got_text = got_text or bool(text)
        if looks_korean(text):
            return text
        if text and not _HAN.search(text) and not fallback:
            fallback = text   # 한자 없는 영어 문장 등은 예비로 보관
    if fallback:
        return fallback
    if last_error is not None and not got_text:
        raise RuntimeError(f"음성 인식에 실패했습니다: {last_error}")
    raise ValueError("한국어로 인식하지 못했습니다. 조용한 곳에서 또렷하게 다시 말씀해 주세요.")


def text_to_speech(client: OpenAI, text: str) -> bytes:
    """
    LLM의 텍스트 응답을 음성 바이너리 데이터(MP3)로 변환(TTS)합니다.
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text[:1000]  # 브라우저 재생 지연 방지를 위해 최대 1000자 제한
    )
    return response.content
