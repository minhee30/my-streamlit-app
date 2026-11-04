
import os
import smtplib
from email.mime.text import MIMEText


def send_email(to, subject, body):
    """Gmail SMTP 이메일 발송"""

    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender or not password:
        return {
            "status": "failed",
            "error": "Gmail 계정 또는 앱 비밀번호가 설정되지 않았습니다."
        }

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, [to], msg.as_string())

        return {"status": "success"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


# OpenAI Function Calling용 도구 정의
EMAIL_TOOLS = [
    {
        "type": "function",
        "name": "send_email",
        "description": "지정된 수신자에게 이메일을 발송합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "이메일 수신자 주소"
                },
                "subject": {
                    "type": "string",
                    "description": "이메일 제목"
                },
                "body": {
                    "type": "string",
                    "description": "이메일 본문"
                }
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False
        },
        "strict": True
    }
]
