"""에이블 캐릭터 색상(오렌지·민트)으로 만든 화면 테마와 공통 디자인 요소입니다."""
import base64
import html
import os
from functools import lru_cache

import streamlit as st

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _asset(name):
    path = os.path.join(_ASSETS, name)
    return path if os.path.exists(path) else None


# 캐릭터 이미지 (assets 폴더에 없으면 이모지로 대신 표시)
ABLE_IMAGE = _asset("able.jpg")
ABLE_AVATAR_PATH = _asset("able_avatar.jpg")
ABLE_AVATAR = ABLE_AVATAR_PATH or "🤖"

# 캐릭터에서 뽑은 색상
COLORS = {
    "orange": "#F28A2E",       # 헤드셋
    "orange_soft": "#FFF3E6",
    "mint": "#19C3B1",         # AIVLE 배지·음성 파형
    "teal": "#0E8F86",
    "mint_soft": "#E8FAF7",
    "cloud": "#FFFFFF",
    "mist": "#F4F3F8",         # 배경 회색
    "ink": "#1F2A37",
    "muted": "#6B7280",
    "line": "#E4E7EE",
}

_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

:root {
  --able-orange:#F28A2E; --able-orange-soft:#FFF3E6; --able-mint:#19C3B1; --able-teal:#0E8F86;
  --able-mint-soft:#E8FAF7; --able-ink:#1F2A37; --able-muted:#6B7280; --able-line:#E4E7EE;
}
html, body, .stApp, [data-testid="stSidebar"], button, input, textarea {
  font-family:'Pretendard', -apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif !important;
}
.stApp {
  background:
    radial-gradient(900px 420px at 0% -5%, rgba(25,195,177,.10), transparent 60%),
    radial-gradient(800px 380px at 100% 0%, rgba(242,138,46,.09), transparent 55%),
    #F4F3F8;
}
h1, h2, h3, h4 { color:var(--able-ink); letter-spacing:-0.02em; font-weight:800 !important; }
[data-testid="stHeader"] { background:transparent; }

/* 사이드바 */
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#FFFFFF 0%,#F1FBF9 60%,#FFF7EF 100%);
  border-right:1px solid var(--able-line);
}
.able-side-name { text-align:center; font-weight:800; font-size:1.25rem; color:var(--able-ink); margin-top:-.3rem; }
.able-side-name span { color:var(--able-orange); }
.able-side-tag { text-align:center; font-size:.78rem; color:var(--able-teal); letter-spacing:.12em; margin-bottom:.4rem; }

/* 상단 소개 배너 */
.able-hero {
  display:flex; align-items:center; gap:28px; padding:22px 30px; margin:0 0 18px 0;
  background:linear-gradient(115deg,#FFFFFF 0%,#EEFBF8 55%,#FFF3E6 100%);
  border:1px solid var(--able-line); border-radius:24px;
  box-shadow:0 10px 30px rgba(14,143,134,.08), 0 2px 6px rgba(31,42,55,.04);
}
.able-hero img { width:150px; height:auto; border-radius:20px; flex-shrink:0; }
.able-kicker { font-size:.78rem; font-weight:700; letter-spacing:.18em; color:var(--able-teal); }
.able-title { font-size:2rem; font-weight:800; color:var(--able-ink); line-height:1.25; margin:.15rem 0 .35rem; }
.able-title b { color:var(--able-orange); }
.able-sub { color:var(--able-muted); font-size:.95rem; }
.able-chips { margin-top:.7rem; display:flex; flex-wrap:wrap; gap:8px; }
.able-chip { padding:4px 12px; border-radius:999px; font-size:.8rem; font-weight:600;
  background:#fff; border:1px solid var(--able-line); color:var(--able-ink); }
.able-chip.mint { background:var(--able-mint-soft); border-color:#BDEFE8; color:var(--able-teal); }
.able-chip.orange { background:var(--able-orange-soft); border-color:#FBD6B3; color:#C2620F; }
@media (max-width:700px){ .able-hero{flex-direction:column; text-align:center;} .able-chips{justify-content:center;} }

/* 카드 (st.container(key='card_...')) */
[class*="st-key-card_"] {
  background:rgba(255,255,255,.92) !important; border:1px solid var(--able-line) !important;
  border-radius:20px !important; padding:18px 20px !important;
  box-shadow:0 8px 24px rgba(31,42,55,.05);
}
[class*="st-key-card_"] h3 { font-size:1.2rem !important; }

/* 캘린더 전체 테두리 */
[class*="st-key-calendar_frame"] {
  border:2px solid #CFE9E5 !important; border-radius:18px !important; padding:10px !important; background:#FFFFFF !important;
}
/* 날짜 칸 */
[class*="st-key-calcell_"] {
  border:1px solid var(--able-line) !important; border-radius:12px !important; background:#FFFFFF !important;
  transition:box-shadow .15s ease, border-color .15s ease;
}
[class*="st-key-calcell_"]:hover { border-color:var(--able-mint) !important; box-shadow:0 4px 14px rgba(25,195,177,.15); }
[class*="st-key-calcell_today"] { border:2px solid var(--able-mint) !important; background:var(--able-mint-soft) !important; }
[class*="st-key-calcell_empty"] { background:#F8F8FB !important; border-style:dashed !important; }
[class*="st-key-calcell_"] button { min-height:30px; padding:0 4px; font-weight:700; }

/* 버튼 */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius:12px; font-weight:600; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
  background:linear-gradient(135deg,#F6A04D 0%,#F28A2E 100%); border:none; color:#fff;
  box-shadow:0 4px 12px rgba(242,138,46,.30);
}
.stButton > button[kind="secondary"]:hover, .stFormSubmitButton > button:hover { border-color:var(--able-mint); color:var(--able-teal); }
.stDownloadButton > button { border:1px solid #BDEFE8; color:var(--able-teal); background:var(--able-mint-soft); }

/* 입력창·녹음기 */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { border-radius:12px; }
[data-testid="stAudioInput"] { border-radius:14px; }
[data-testid="stForm"] { border:1px solid var(--able-line); border-radius:16px; background:rgba(255,255,255,.7); }

/* 대화 말풍선 */
[data-testid="stChatMessage"] { border-radius:16px; background:#FFFFFF; border:1px solid var(--able-line); }

/* 일정 종류 배지 */
.able-badge { color:#fff; border-radius:6px; padding:1px 8px; font-size:.74rem; font-weight:700; margin-right:6px; }
.able-event { font-size:.76rem; line-height:1.3; padding:2px 5px; margin-bottom:3px; border-radius:6px; word-break:break-all; }
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


@lru_cache(maxsize=4)
def _data_uri(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def render_sidebar_brand() -> None:
    if ABLE_IMAGE:
        st.image(ABLE_IMAGE, width="stretch")
    st.markdown("<div class='able-side-tag'>AI PERSONAL ASSISTANT</div>"
                "<div class='able-side-name'>에이블 <span>ABLE</span></div>", unsafe_allow_html=True)


def render_hero() -> None:
    img = f"<img src='{_data_uri(ABLE_IMAGE)}' alt='에이블'/>" if ABLE_IMAGE else ""
    st.markdown(
        f"""<div class='able-hero'>{img}<div>
        <div class='able-kicker'>AI PERSONAL ASSISTANT · ABLE</div>
        <div class='able-title'>에이블과 함께하는 <b>일정 & 업무 관리</b></div>
        <div class='able-sub'>회의록을 올리면 일정이 캘린더에 바로 등록되고, 회의 요약을 음성으로 들려 드려요.</div>
        <div class='able-chips'>
          <span class='able-chip mint'>🎙️ 음성 명령</span>
          <span class='able-chip orange'>📅 회의록 자동 등록</span>
          <span class='able-chip'>🔊 요약 음성 재생</span>
          <span class='able-chip'>✉️ 이메일 전송</span>
        </div></div></div>""",
        unsafe_allow_html=True)


def badge(label: str, color: str) -> str:
    return f"<span class='able-badge' style='background:{color}'>{html.escape(label)}</span>"


def card(key: str):
    """테두리·그림자가 있는 카드 영역"""
    return st.container(key=f"card_{key}")
