
from openai import OpenAI

client = OpenAI(timeout=60.0)


def speech_to_text(audio_value):
    """음성 파일을 텍스트로 변환"""

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


def text_to_speech(text):
    """텍스트를 MP3 음성으로 변환"""

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=text
    )

    return response.content
