
from openai import OpenAI

client = OpenAI(timeout=60.0)

SYSTEM_ROLE = """
당신은 기업용 AI 업무 비서 '에이블'입니다.
사용자의 업무를 친절하고 정확하게 지원합니다.
답변은 한국어로 명확하고 전문적으로 작성합니다.
제공된 자료에 없는 사실은 임의로 만들어내지 않습니다.
"""


def ask_ai(question, text=""):
    """일반 질문 또는 문서 기반 질문"""

    system_content = SYSTEM_ROLE

    if text:
        system_content += f"\n\n참고 문서:\n{text}"

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
    )

    return response.output_text


def summarize_meeting(meeting_text, employee_name=""):
    """미팅 요약 및 담당 업무 추출"""

    role = """
    당신은 AI 업무 비서 에이블입니다.
    미팅 내용을 읽고 다음을 정리하세요.

    1. 미팅 핵심 요약
    2. 주요 결정사항
    3. 담당자별 업무, 기한, 관련자
    4. 지정된 직원에게 배정된 업무

    확인되지 않은 사실은 만들어내지 마세요.
    지정된 직원의 업무가 없으면
    '배정된 업무가 없습니다'라고 알려주세요.
    """

    prompt = f"""
    확인할 직원: {employee_name or '전체 참석자'}

    미팅 내용:
    {meeting_text}
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ]
    )

    return response.output_text
