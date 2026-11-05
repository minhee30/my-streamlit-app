"""캘린더(회의/마감 색 구분), 상세 일정·To-do 음성 등록, 추천 To-do, 한글 PNG 저장."""
import calendar
import hashlib
import html
import io
import os
import tempfile
import urllib.request
from datetime import date, time
from functools import lru_cache
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from app_state import korea_today
from audio_utils import transcribe_audio
from command_service import parse_spoken_item
from meeting_planner import SCHEDULE_KINDS, add_schedule, add_todo, focus_date, kind_of
from ui_theme import ABLE_AVATAR_PATH, COLORS, badge

WEEKDAYS = ['월', '화', '수', '목', '금', '토', '일']
KIND_OPTIONS = {'회의': 'meeting', '마감기한': 'deadline', '일반': 'general'}
CELL_HEIGHT = 165   # 화면 캘린더 날짜 칸 높이(px). 더 길게 하려면 이 값을 늘리세요.
SAT_COLOR, SUN_COLOR = COLORS['teal'], '#E0483E'


# ---------------- 한글 폰트 (PNG 한글 깨짐 해결) ----------------
# 1순위: 저장소의 fonts/NanumGothic.ttf  2순위: packages.txt로 설치되는 fonts-nanum
# 3순위: 그 밖의 CJK 폰트  4순위: Google Fonts에서 나눔고딕을 한 번 내려받아 임시 폴더에 보관
_FONT_CANDIDATES = (
    str(Path(__file__).with_name('fonts') / 'NanumGothic.ttf'),
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
)
_FONT_URL = 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf'


@lru_cache(maxsize=1)
def _font_path():
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    target = Path(tempfile.gettempdir()) / 'NanumGothic-Regular.ttf'
    if not target.exists():
        try:
            with urllib.request.urlopen(_FONT_URL, timeout=15) as resp:
                target.write_bytes(resp.read())
        except Exception:
            return None
    return str(target)


@lru_cache(maxsize=32)
def _font(size):
    path = _font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _fit(draw, text, font, max_width):
    """칸 너비를 넘는 글자는 잘리지 않도록 '…'로 줄입니다."""
    text = str(text)
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + '…', font=font) > max_width:
        text = text[:-1]
    return text + '…'


def _wrap(draw, text, font, max_width):
    """긴 할 일은 이미지 밖으로 잘리지 않도록 여러 줄로 나눕니다."""
    lines, current = [], ''
    for ch in str(text):
        if current and draw.textlength(current + ch, font=font) > max_width:
            lines.append(current)
            current = ch.lstrip()
        else:
            current += ch
    lines.append(current)
    return lines


def _paste_avatar(img, x, y, size):
    """PNG 오른쪽 위에 에이블 캐릭터를 둥글게 넣습니다."""
    if not ABLE_AVATAR_PATH:
        return
    try:
        avatar = Image.open(ABLE_AVATAR_PATH).convert('RGB').resize((size, size))
        mask = Image.new('L', (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 4, fill=255)
        img.paste(avatar, (x, y), mask)
    except Exception:
        pass


def _to_png(image):
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()


def calendar_png(year, month, events):
    weeks = calendar.monthcalendar(year, month)
    cw, ch, top, margin = 160, 165, 120, 20
    width = cw * 7 + margin * 2
    img = Image.new('RGB', (width, top + len(weeks) * ch + 75), '#FFFFFF')
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, width, 8), fill=COLORS['mint'])
    d.text((margin, 24), 'ABLE 캘린더', font=_font(30), fill=COLORS['ink'])
    d.text((margin + 190, 30), f'{year}년 {month}월', font=_font(24), fill=COLORS['orange'])
    _paste_avatar(img, width - margin - 70, 18, 70)
    for j, wd in enumerate(WEEKDAYS):
        color = SUN_COLOR if j == 6 else SAT_COLOR if j == 5 else '#475569'
        d.text((margin + j * cw + 10, 88), wd, font=_font(18), fill=color)
    small = _font(14)
    today = korea_today()
    for i, week in enumerate(weeks):
        for j, day in enumerate(week):
            x, y = margin + j * cw, top + i * ch
            if not day:
                d.rectangle((x, y, x + cw - 1, y + ch - 1), fill='#F8F8FB', outline=COLORS['line'])
                continue
            is_today = date(year, month, day) == today
            d.rectangle((x, y, x + cw - 1, y + ch - 1), fill=COLORS['mint_soft'] if is_today else '#FFFFFF',
                        outline=COLORS['mint'] if is_today else COLORS['line'], width=2 if is_today else 1)
            d.text((x + 8, y + 6), str(day), font=_font(18),
                   fill=SUN_COLOR if j == 6 else SAT_COLOR if j == 5 else COLORS['ink'])
            day_string = f'{year}-{month:02d}-{day:02d}'
            matches = sorted((e for e in events if e['date'] == day_string), key=lambda e: e.get('time', ''))
            for k, e in enumerate(matches[:4]):
                info = SCHEDULE_KINDS[kind_of(e)]
                ty = y + 36 + k * 28
                d.rounded_rectangle((x + 6, ty - 1, x + cw - 7, ty + 21), radius=5, fill=info['soft'])
                d.rectangle((x + 6, ty - 1, x + 9, ty + 21), fill=info['color'])
                label = (e.get('time', '') + ' ' + e['schedule']).strip()
                d.text((x + 14, ty + 2), _fit(d, label, small, cw - 26), font=small, fill=info['color'])
            if len(matches) > 4:
                d.text((x + 8, y + ch - 20), f'외 {len(matches) - 4}개', font=_font(12), fill=COLORS['muted'])
    # 캘린더 전체 테두리
    d.rounded_rectangle((margin - 4, top - 4, margin + cw * 7 + 3, top + len(weeks) * ch + 3),
                        radius=10, outline=COLORS['teal'], width=3)
    # 범례
    ly, lx = top + len(weeks) * ch + 26, margin
    for info in SCHEDULE_KINDS.values():
        d.rounded_rectangle((lx, ly + 3, lx + 16, ly + 19), radius=4, fill=info['color'])
        d.text((lx + 22, ly), info['label'], font=_font(17), fill='#334155')
        lx += 46 + int(d.textlength(info['label'], font=_font(17)))
    return _to_png(img)


def todo_png(todos):
    width, pad, line_h = 950, 34, 30
    font, title_font = _font(18), _font(30)
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    text_x = pad + 36
    rows = []
    for item in todos:
        label = item['task'] + (f" (담당: {item['assignee']})" if item.get('assignee') else '')
        rows.append((item['done'], _wrap(measure, label, font, width - text_x - pad)))
    height = max(220, 125 + sum(len(lines) * line_h + 14 for _, lines in rows) + pad)
    img = Image.new('RGB', (width, height), '#FFFFFF')
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, width, 8), fill=COLORS['orange'])
    d.text((pad, 26), 'My To-do List', font=title_font, fill=COLORS['ink'])
    done_count = sum(1 for done, _ in rows if done)
    d.text((pad, 70), f'완료 {done_count}/{len(rows)}', font=_font(16), fill=COLORS['teal'])
    _paste_avatar(img, width - pad - 64, 20, 64)
    y = 115
    if not rows:
        d.text((pad, y), '등록된 할 일이 없습니다.', font=font, fill=COLORS['muted'])
    for done, lines in rows:
        # 체크박스를 글자 대신 도형으로 그려 기호가 잘리거나 깨지지 않게 합니다.
        d.rounded_rectangle((pad, y + 4, pad + 22, y + 26), radius=5, width=2,
                            outline=COLORS['teal'] if done else '#94A3B8',
                            fill=COLORS['mint'] if done else '#FFFFFF')
        if done:
            d.line([(pad + 5, y + 15), (pad + 10, y + 20), (pad + 18, y + 9)], fill='#FFFFFF', width=3)
        for i, line in enumerate(lines):
            d.text((text_x, y + i * line_h), line, font=font, fill='#94A3B8' if done else COLORS['ink'])
        y += len(lines) * line_h + 14
    return _to_png(img)


def _safe_download(container, label, builder, file_name):
    try:
        data = builder()
    except Exception as exc:
        container.caption(f'PNG 생성 오류: {exc}')
        return
    # on_click='ignore' : 다운로드 클릭 시 앱을 다시 실행하지 않습니다.
    container.download_button(label, data, file_name=file_name, mime='image/png', on_click='ignore')


# ---------------- 음성 입력(STT) 공통 ----------------
# 1) 녹음  2) 한국어로 인식(무음·중국어 결과 걸러 냄)  3) 인식된 문장을 보여 주고 고칠 수 있게 함  4) 등록
def _voice_input(label, key, mode, get_client):
    """녹음된 음성을 인식해 확인 대기 목록에 넣습니다. 확인 대기 중이면 녹음기를 숨깁니다."""
    if key in st.session_state.voice_pending:
        return
    versions = st.session_state.voice_versions
    audio = st.audio_input(label, key=f'{key}_{versions.get(key, 0)}')
    if audio is None:
        return
    data = audio.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest in st.session_state.processed_voice:
        return
    st.session_state.processed_voice.add(digest)
    versions[key] = versions.get(key, 0) + 1   # 다음 실행에서 녹음 위젯을 비웁니다.
    try:
        with st.spinner('음성을 한국어로 인식하는 중...'):
            text = transcribe_audio(get_client(), data, mode=mode)
        st.session_state.voice_pending[key] = text
    except Exception as exc:
        st.session_state.voice_notice[key] = ('warn', f'음성 인식 오류: {exc}')
    st.rerun()


def _voice_confirm(key, placeholder):
    """인식 결과를 보여 주고, 사용자가 고친 문장을 돌려줍니다. 아직 확인 전이면 None."""
    pending = st.session_state.voice_pending
    if key not in pending:
        return None
    with st.container(border=True):
        st.markdown(f"<span style='color:{COLORS['teal']};font-weight:700'>🎙️ 이렇게 들었어요</span> "
                    "— 틀린 글자는 고친 뒤 등록하세요.", unsafe_allow_html=True)
        digest = hashlib.md5(pending[key].encode('utf-8')).hexdigest()[:8]
        text = st.text_input('인식된 내용', value=pending[key], key=f'{key}_edit_{digest}',
                             placeholder=placeholder, label_visibility='collapsed')
        ok_col, retry_col = st.columns(2)
        ok = ok_col.button('등록', key=f'{key}_ok', type='primary', use_container_width=True)
        retry = retry_col.button('다시 녹음', key=f'{key}_retry', use_container_width=True)
    if retry:
        pending.pop(key, None)
        st.rerun()
    if ok:
        pending.pop(key, None)
        text = text.strip()
        if not text:
            st.session_state.voice_notice[key] = ('warn', '등록할 내용이 비어 있습니다.')
            st.rerun()
        return text
    return None


def _show_notice(key):
    notice = st.session_state.voice_notice.pop(key, None)
    if notice:
        (st.success if notice[0] == 'ok' else st.warning)(notice[1])


def _kind_badge(kind):
    info = SCHEDULE_KINDS[kind]
    return badge(info['label'], info['color'])


def _recommended(title):
    if '회의' in title:
        return ['회의 안건 정리', '회의 자료 준비', '참석자 확인']
    if '발표' in title:
        return ['발표 자료 작성', '발표 대본 준비', '발표 리허설']
    if '보고서' in title:
        return ['자료 조사', '보고서 초안 작성', '최종 검토']
    return [f'{title} 준비 사항 확인']


# ---------------- 캘린더 ----------------
def render_calendar():
    year, month = st.session_state.calendar_year, st.session_state.calendar_month
    title, a, b, c, d = st.columns([2.2, 0.5, 1.6, 0.5, 1.4], vertical_alignment='center')
    title.subheader('📅 일정 캘린더')
    if a.button('◀', key='prev_month', use_container_width=True):
        month -= 1
        if month < 1:
            month, year = 12, year - 1
        st.session_state.calendar_year, st.session_state.calendar_month = year, month
        st.rerun()
    b.markdown(f"<div style='text-align:center;font-size:1.35rem;font-weight:800;color:{COLORS['ink']}'>"
               f"{year}년 <span style='color:{COLORS['orange']}'>{month}월</span></div>", unsafe_allow_html=True)
    if c.button('▶', key='next_month', use_container_width=True):
        month += 1
        if month > 12:
            month, year = 1, year + 1
        st.session_state.calendar_year, st.session_state.calendar_month = year, month
        st.rerun()
    events = [dict(e) for e in st.session_state.schedules]
    _safe_download(d, '📥 달력 PNG', lambda: calendar_png(year, month, events), f'calendar_{year}_{month:02d}.png')

    st.markdown(' '.join(_kind_badge(k) for k in SCHEDULE_KINDS)
                + f"<span style='color:{COLORS['muted']};font-size:.8rem;margin-left:6px'>"
                  "· 날짜를 누르면 아래에서 상세 일정을 볼 수 있어요</span>", unsafe_allow_html=True)

    today = korea_today()
    # 캘린더 전체를 테두리 상자로 감싸고, 날짜 칸마다 테두리와 고정 높이를 줍니다.
    with st.container(key='calendar_frame'):
        for j, (col, wd) in enumerate(zip(st.columns(7, gap='small'), WEEKDAYS)):
            color = SUN_COLOR if j == 6 else SAT_COLOR if j == 5 else COLORS['muted']
            col.markdown(f"<div style='text-align:center;font-weight:800;color:{color}'>{wd}</div>",
                         unsafe_allow_html=True)
        for w, week in enumerate(calendar.monthcalendar(year, month)):
            for j, (col, day) in enumerate(zip(st.columns(7, gap='small'), week)):
                if not day:
                    col.container(key=f'calcell_empty_{year}_{month}_{w}_{j}', height=CELL_HEIGHT)
                    continue
                current = date(year, month, day)
                key = f"calcell_{'today_' if current == today else ''}{current.isoformat()}"
                matches = sorted((e for e in st.session_state.schedules if e['date'] == current.isoformat()),
                                 key=lambda e: e.get('time', ''))
                with col.container(key=key, height=CELL_HEIGHT):
                    if st.button(str(day), key=f'date_{current}', use_container_width=True,
                                 type='primary' if current == st.session_state.selected_date else 'tertiary'):
                        st.session_state.selected_date = current
                        st.rerun()
                    for item in matches[:4]:
                        info = SCHEDULE_KINDS[kind_of(item)]
                        text = html.escape((item.get('time', '') + ' ' + item['schedule']).strip())
                        st.markdown(
                            f"<div class='able-event' style='color:{info['color']};background:{info['soft']};"
                            f"border-left:3px solid {info['color']}'>{text}</div>",
                            unsafe_allow_html=True)
                    if len(matches) > 4:
                        st.caption(f'외 {len(matches) - 4}개')


# ---------------- 상세 일정 + 추천 To-do ----------------
def render_details(get_client):
    selected = st.session_state.selected_date
    left, right = st.columns([1.1, 1], gap='large')
    with left:
        st.subheader(f'📝 {selected.month}월 {selected.day}일 상세 일정')
        _show_notice('voice_schedule')
        matches = sorted((x for x in st.session_state.schedules if x['date'] == selected.isoformat()),
                         key=lambda e: e.get('time', ''))
        if not matches:
            st.info('선택한 날짜의 일정이 없습니다.')
        for item in list(matches):
            a, b = st.columns([5, 1], vertical_alignment='center')
            text = html.escape((item.get('time', '') + ' ' + item['schedule']).strip())
            a.markdown(_kind_badge(kind_of(item)) + text, unsafe_allow_html=True)
            if b.button('삭제', key=f'del_schedule_{item["id"]}'):
                st.session_state.schedules.remove(item)
                st.rerun()

        # 음성으로 일정 등록 → 인식된 문장을 확인·수정한 뒤 등록
        _voice_input('🎙️ 음성으로 일정 등록', 'voice_schedule', 'schedule', get_client)
        spoken = _voice_confirm('voice_schedule', '예: 오후 3시 팀 회의')
        if spoken:
            try:
                parsed = parse_spoken_item(get_client(), spoken, 'schedule', selected.isoformat())
                title = str(parsed.get('title') or spoken).strip()
                day = parsed.get('date') or selected.isoformat()
                added = add_schedule(title, day, parsed.get('time', ''), kind=parsed.get('kind', 'general'))
                if added:
                    focus_date(date.fromisoformat(day))
                    st.session_state.calendar_recommendations = _recommended(title)
                    notice = ('ok', f"{day} '{title}' 일정을 등록했습니다.")
                else:
                    notice = ('warn', '이미 등록된 일정입니다.')
            except Exception as exc:
                notice = ('warn', f'음성 일정 등록 실패: {exc}')
            st.session_state.voice_notice['voice_schedule'] = notice
            st.rerun()

        has_time = st.checkbox('시간 지정', key='has_time')
        with st.form('schedule_form', clear_on_submit=True):
            title = st.text_input('일정 내용')
            kind_label = st.radio('일정 종류', list(KIND_OPTIONS), horizontal=True)
            when = st.time_input('일정 시간', value=time(9, 0)) if has_time else None
            submitted = st.form_submit_button('선택한 날짜에 등록', type='primary')
        if submitted:
            try:
                if add_schedule(title, selected.isoformat(), when.strftime('%H:%M') if when else '',
                                kind=KIND_OPTIONS[kind_label]):
                    st.session_state.calendar_recommendations = _recommended(title)
                    st.rerun()
                st.info('이미 등록된 일정입니다.')
            except ValueError as exc:
                st.warning(str(exc))

    with right:
        st.subheader('💡 추천 To-do List')
        analysis = st.session_state.get('meeting_analysis') or {}
        recs = [{'title': x['title'], 'assignee': x.get('assignee', ''), 'parent': x.get('parent', '')}
                for x in analysis.get('todos', [])]
        recs += [{'title': x, 'assignee': '', 'parent': ''} for x in st.session_state.calendar_recommendations]
        existing = {x['task'] for x in st.session_state.todos}
        seen, visible = set(), []
        for item in recs:
            if item['title'] not in existing and item['title'] not in seen:
                seen.add(item['title'])
                visible.append(item)
        if not visible:
            st.caption('회의록 분석 또는 일정 등록 후 추천이 나타납니다.')
        for item in visible:
            a, b = st.columns([5, 1], vertical_alignment='center')
            a.write('• ' + item['title'])
            key = hashlib.md5(item['title'].encode('utf-8')).hexdigest()[:12]
            if b.button('추가', key=f'add_rec_{key}'):
                add_todo(item['title'], item['assignee'], item['parent'])
                st.rerun()


# ---------------- My To-do ----------------
def render_todos(get_client):
    st.subheader('✅ My To-do List')
    _show_notice('voice_todo')

    # 음성으로 할 일 추가 → 인식된 문장을 확인·수정한 뒤 등록
    _voice_input('🎙️ 음성으로 할 일 추가', 'voice_todo', 'todo', get_client)
    spoken = _voice_confirm('voice_todo', '예: 견적서 보내기')
    if spoken:
        try:
            parsed = parse_spoken_item(get_client(), spoken, 'todo')
            title = str(parsed.get('title') or spoken).strip()
            notice = ('ok', f"'{title}'를 추가했습니다.") if add_todo(title) else ('warn', '이미 등록된 할 일입니다.')
        except Exception as exc:
            notice = ('warn', f'음성 할 일 추가 실패: {exc}')
        st.session_state.voice_notice['voice_todo'] = notice
        st.rerun()

    with st.form('todo_form', clear_on_submit=True):
        title = st.text_input('새로운 할 일')
        submit = st.form_submit_button('업무 추가', type='primary')
    if submit:
        try:
            if add_todo(title):
                st.rerun()
        except ValueError as exc:
            st.warning(str(exc))

    total = len(st.session_state.todos)
    done = sum(x['done'] for x in st.session_state.todos)
    st.caption(f'완료 {done}/{total}')
    for item in list(st.session_state.todos):
        # 좁은 오른쪽 영역에서도 삭제 버튼이 잘리지 않도록 여백 없는 버튼을 사용합니다.
        a, b = st.columns([0.82, 0.18], vertical_alignment='center')
        item['done'] = a.checkbox(item['task'], value=item['done'], key=f'todo_{item["id"]}')
        if b.button('✕', key=f'del_todo_{item["id"]}', type='tertiary', help='삭제'):
            st.session_state.todos.remove(item)
            st.rerun()
    todos = [dict(x) for x in st.session_state.todos]
    _safe_download(st, '📥 To-do PNG', lambda: todo_png(todos), 'todos.png')
