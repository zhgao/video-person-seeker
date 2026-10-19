#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_parser.py —— 中文自然语言查询解析（监控找人 v2）

把"这两天谁在摘菜"这类口语解析成结构化查询：
    {"date_from": date, "date_to": date, "mode": "bend" | "person"}

- 日期关键词：今天 / 昨天 / 前天 / 这两天（近两天/最近两天）/ 这周（本周）
- 显式日期：2026-10-18、10-18、10/18、10月18日（单个日期=当天全天）；
  也支持"10月18日到10月20日"这类范围。
- 动作关键词：出现 摘菜/摘/弯腰/干活 → mode='bend'（弯腰检测）；
  只有 有人/谁/人 → mode='person'（人物检测）。

独立模块，方便单测。注意：解析用的是"今天"是哪一天，
单测时请传入固定的 now 参数。
"""
import re
from datetime import date, datetime, timedelta

# ---- 动作关键词：命中任意一个即判定为 bend 模式 ----
BEND_KEYWORDS = ("摘菜", "弯腰", "干活", "摘")

# ---- 日期关键词 ----
_TWO_DAYS = ("这两天", "近两天", "最近两天")
_THIS_WEEK = ("这周", "本周")


def _parse_explicit_dates(text, today):
    """从文本里找显式日期，返回按出现顺序的 date 列表。"""
    found = []

    def add(y, m, d):
        try:
            found.append(date(y, m, d))
        except ValueError:
            pass  # 非法日期（如 2月30日）直接忽略

    # 1) 2026-10-18 / 2026/10/18
    for m in re.finditer(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text):
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # 2) 10月18日 / 10月18号
    for m in re.finditer(r"(\d{1,2})月(\d{1,2})[日号]?", text):
        mm, dd = int(m.group(1)), int(m.group(2))
        try:
            dt = date(today.year, mm, dd)
        except ValueError:
            continue
        if dt > today:  # 未来日期 → 去年
            dt = date(today.year - 1, mm, dd)
        found.append(dt)

    # 3) 10-18 / 10/18（排除已被"2026-10-18"匹配过的位置）
    taken = [m.span() for m in
             re.finditer(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text)]
    taken += [m.span() for m in re.finditer(r"\d{1,2}月\d{1,2}[日号]?", text)]

    def overlaps(span):
        return any(span[0] < t[1] and span[1] > t[0] for t in taken)

    for m in re.finditer(r"(?<!\d)(\d{1,2})[-/](\d{1,2})(?!\d)", text):
        if overlaps(m.span()):
            continue
        mm, dd = int(m.group(1)), int(m.group(2))
        try:
            dt = date(today.year, mm, dd)
        except ValueError:
            continue
        if dt > today:
            dt = date(today.year - 1, mm, dd)
        found.append(dt)

    return found


def parse_query(text, now=None):
    """解析自然语言查询。

    返回 {"date_from": date, "date_to": date, "mode": "bend"|"person"}。
    now: 参考"今天"（datetime/date），默认当前时间；单测时传入固定值。
    """
    text = (text or "").strip()
    today = (now.date() if isinstance(now, datetime)
             else now if isinstance(now, date) else date.today())

    # ---- 1) 显式日期优先 ----
    dates = _parse_explicit_dates(text, today)
    if dates:
        date_from = min(dates)
        date_to = max(dates)
    else:
        # ---- 2) 日期关键词 ----
        if any(k in text for k in _TWO_DAYS):
            date_from, date_to = today - timedelta(days=1), today
        elif any(k in text for k in _THIS_WEEK):
            monday = today - timedelta(days=today.weekday())
            date_from, date_to = monday, today
        elif "前天" in text:
            date_from = date_to = today - timedelta(days=2)
        elif "昨天" in text:
            date_from = date_to = today - timedelta(days=1)
        elif "今天" in text:
            date_from = date_to = today
        else:
            # 没提时间 → 默认查今天（单文件场景下也合理）
            date_from = date_to = today

    # ---- 3) 动作模式 ----
    mode = "bend" if any(k in text for k in BEND_KEYWORDS) else "person"

    return {"date_from": date_from, "date_to": date_to, "mode": mode}


if __name__ == "__main__":
    # 手动试跑：python query_parser.py "这两天谁在摘菜"
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "这两天谁在摘菜"
    print(parse_query(q))
