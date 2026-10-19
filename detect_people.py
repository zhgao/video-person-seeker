#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控视频找人 detect_people.py（v2）

v1（保持不变）：扫描单个录好的监控视频，用 YOLO 检测"人"出现的时间段，
生成可点击跳转的 HTML 报告。
    python detect_people.py 监控录像.mp4

v2（新增）：自然语言查询 + 多文件批量 + 弯腰（摘菜）启发式检测。
    python detect_people.py 视频文件夹 --query "这两天谁在摘菜"

依赖: pip install -r requirements.txt
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime

import cv2

from query_parser import parse_query

VIDEO_EXTS = (".mp4", ".avi", ".webm", ".mov", ".mkv", ".flv", ".mpg", ".mpeg", ".ts")

# COCO 关键点序号：5/6=左右肩，11/12=左右髋
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12


def fmt_time(sec):
    """秒数 -> HH:MM:SS"""
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_ffmpeg():
    return shutil.which("ffmpeg")


def sanitize_filename(name):
    """查询词 → 安全文件名：保留中文/字母数字/横线下划线，其余换成下划线。"""
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name.strip())
    return name[:60] if name else "查询结果"


# ---------------------------------------------------------------- v1（原样保留）
def run_v1(args):
    """v1 单文件人物检测：逻辑与 v1.0 完全一致，保证老命令照常工作。"""
    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        print(f"错误：找不到视频文件：{args.video}")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.abspath(args.out) if args.out else \
        os.path.join(os.path.dirname(video_path), f"{stem}_找人结果")
    os.makedirs(out_dir, exist_ok=True)

    # 延迟导入：--help 等不需要模型
    from ultralytics import YOLO
    print("正在加载 YOLO 模型（首次运行会自动下载 yolov8n.pt，约 6MB）…")
    model = YOLO("yolov8n.pt")  # nano 版，最快

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：打不开视频：{args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0
    step = max(1, int(round(args.every * fps)))

    print(f"视频：{os.path.basename(video_path)}  "
          f"时长 {fmt_time(duration)}  帧率 {fps:.1f}")
    print(f"每 {args.every} 秒抽一帧检测（置信度阈值 {args.conf}）…")

    hits = []          # 检测到人的时间点（秒）
    boxes_at = {}      # 时间点 -> 检测框（用于缩略图）
    idx = 0
    checked = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        results = model.predict(frame, verbose=False, conf=args.conf)
        people = [b for b in results[0].boxes
                  if int(b.cls[0]) == 0]  # class 0 = person
        if people:
            hits.append(t)
            boxes_at[t] = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                            int(b.xyxy[0][2]), int(b.xyxy[0][3]))
                           for b in people]
        checked += 1
        if total_frames:
            pct = min(100.0, idx / total_frames * 100)
            print(f"\r  进度 {pct:5.1f}%  已检测 {checked} 帧，发现 {len(hits)} 个有人时刻",
                  end="", flush=True)
        idx += step
    print()
    cap.release()

    # 合并连续时刻为片段：间隔 <= gap 则合并
    segments = []
    for t in hits:
        if segments and t - segments[-1][1] <= args.gap + 1e-6:
            segments[-1][1] = t
        else:
            segments.append([t, t])
    # 片段尾部延长一个抽帧间隔（覆盖最后一帧所在的窗口），不超过总时长
    segments = [[s, min(e + args.every, duration)] for s, e in segments]

    print(f"共发现 {len(segments)} 个有人片段。")

    # 保存缩略图（片段起始帧，画出检测框）
    cap = cv2.VideoCapture(video_path)
    thumbs = []
    for i, (s, e) in enumerate(segments):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
        ok, frame = cap.read()
        thumb_name = f"thumb_{i + 1:03d}.jpg"
        if ok:
            for (x1, y1, x2, y2) in boxes_at.get(s, []):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            h, w = frame.shape[:2]
            nw = 320
            frame = cv2.resize(frame, (nw, int(h * nw / w)))
            cv2.imwrite(os.path.join(out_dir, thumb_name),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumbs.append(thumb_name)
    cap.release()

    # segments.json
    seg_path = os.path.join(out_dir, "segments.json")
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump({"video": os.path.basename(video_path),
                   "duration": round(duration, 2),
                   "every": args.every, "conf": args.conf,
                   "segments": [{"start": round(s, 2), "end": round(e, 2)}
                                for s, e in segments]},
                  f, ensure_ascii=False, indent=2)
    print(f"已保存：{seg_path}")

    # --reel：剪辑集锦
    reel_name = None
    if args.reel:
        ff = find_ffmpeg()
        if not ff:
            print("提示：未找到 ffmpeg，跳过集锦剪辑（只生成报告）。")
        elif not segments:
            print("提示：没有检测到片段，跳过集锦剪辑。")
        else:
            reel_name = f"{stem}_有人集锦.mp4"
            parts = []
            print("正在剪辑集锦…")
            for i, (s, e) in enumerate(segments):
                part = os.path.join(out_dir, f"_part_{i:03d}.mp4")
                subprocess.run([ff, "-y", "-v", "error",
                                "-ss", f"{s:.2f}", "-t", f"{e - s:.2f}",
                                "-i", video_path, "-c", "copy", part],
                               check=True)
                parts.append(part)
            lst = os.path.join(out_dir, "_parts.txt")
            with open(lst, "w", encoding="utf-8") as f:
                for p in parts:
                    f.write(f"file '{p}'\n")
            subprocess.run([ff, "-y", "-v", "error", "-f", "concat",
                            "-safe", "0", "-i", lst, "-c", "copy",
                            os.path.join(out_dir, reel_name)], check=True)
            for p in parts:
                os.remove(p)
            os.remove(lst)
            print(f"已保存集锦：{os.path.join(out_dir, reel_name)}")

    # report.html（纯离线，file:// 直接打开）
    rel_video = os.path.relpath(video_path, out_dir).replace(os.sep, "/")
    rows = []
    for i, ((s, e), th) in enumerate(zip(segments, thumbs)):
        rows.append(
            f'<div class="ch" data-t="{s:.2f}" data-v="{rel_video}">'
            f'<img src="{th}" alt="片段{i + 1}">'
            f'<div class="meta"><b>片段 {i + 1}</b>'
            f'<span>{fmt_time(s)} → {fmt_time(e)}（{e - s:.0f} 秒）</span>'
            f'<em>点击跳转 ▶</em></div></div>')
    body = ("\n".join(rows) if rows
            else '<p class="empty">整段视频没有检测到人。</p>')
    reel_html = (f'<p class="reel">集锦视频：<a href="{reel_name}">{reel_name}</a></p>'
                 if reel_name else "")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>监控视频找人 - {stem}</title>
<style>
body{{font-family:"Microsoft YaHei",PingFang SC,system-ui,sans-serif;
max-width:960px;margin:0 auto;padding:16px;background:#f5f6f8;color:#222}}
h1{{font-size:20px}}video{{width:100%;background:#000;border-radius:8px}}
.info{{color:#666;font-size:14px}}.ch{{display:flex;gap:12px;align-items:center;
background:#fff;border-radius:8px;padding:8px;margin:8px 0;cursor:pointer;
box-shadow:0 1px 3px rgba(0,0,0,.08)}}.ch:hover{{background:#eef4ff}}
.ch img{{width:160px;border-radius:4px}}.ch .meta b{{display:block;font-size:16px}}
.ch .meta span{{color:#666;font-size:14px}}.ch .meta em{{display:block;
color:#1a73e8;font-style:normal;font-size:13px;margin-top:4px}}
.empty{{color:#999;padding:24px;text-align:center}}.reel a{{color:#1a73e8}}
</style></head><body>
<h1>🎥 监控视频找人</h1>
<p class="info">视频：{stem} ｜ 时长 {fmt_time(duration)} ｜ 发现 {len(segments)} 个有人片段</p>
<video id="v" controls preload="metadata" src="{rel_video}"></video>
{reel_html}
<h2>有人片段（点击跳转）</h2>
{body}
<script>
var v = document.getElementById('v');
document.querySelectorAll('.ch').forEach(function(el){{
  el.addEventListener('click', function(){{
    var src = el.getAttribute('data-v');
    if (v.getAttribute('src') !== src) {{ v.setAttribute('src', src); }}
    v.currentTime = parseFloat(el.getAttribute('data-t'));
    v.play();
  }});
}});
</script>
</body></html>"""
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成报告：{html_path}")
    print("完成！用浏览器打开 report.html，点击任意片段即可跳转播放。")


# ---------------------------------------------------------------- v2：弯腰启发式
def trunk_lean_angle(sh_xy, hip_xy):
    """躯干前倾角（度）。

    sh_xy / hip_xy：肩中心、髋中心坐标 (x, y)，图像坐标系（y 向下）。
    站立挺直 ≈ 0°，向前弯腰到水平 ≈ 90°。
    返回 None 表示几何无效（比如肩在髋下方，可能是关键点误检）。
    """
    tx, ty = sh_xy[0] - hip_xy[0], sh_xy[1] - hip_xy[1]
    if ty >= -1e-6:  # 肩没有在髋上方 → 不可信
        return None
    return math.degrees(math.atan2(abs(tx), -ty))


def analyze_frame_bend(pose_model, frame, conf, bend_angle, kp_conf=0.3):
    """对一帧跑姿态估计，返回 [(box, angle, bending)]。

    box: (x1,y1,x2,y2) 人物框；angle: 前倾角（None=算不出）；
    bending: 是否判定为弯腰。
    """
    results = pose_model.predict(frame, verbose=False, conf=conf)
    r = results[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return []
    kps = r.keypoints.xy.cpu().numpy()      # (N,17,2)
    kpc = r.keypoints.conf.cpu().numpy()    # (N,17)
    boxes = r.boxes.xyxy.cpu().numpy()      # (N,4)
    out = []
    for i in range(len(kps)):
        # 肩/髋四个关键点置信度都要达标，否则这个人不判
        need = (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP)
        if any(kpc[i][k] < kp_conf for k in need):
            continue
        sh = ((kps[i][KP_L_SHOULDER] + kps[i][KP_R_SHOULDER]) / 2).tolist()
        hip = ((kps[i][KP_L_HIP] + kps[i][KP_R_HIP]) / 2).tolist()
        angle = trunk_lean_angle(sh, hip)
        bending = angle is not None and angle >= bend_angle
        out.append((tuple(map(int, boxes[i].tolist())), angle, bending))
    return out


def load_pose_model():
    """加载 yolov8n-pose（ultralytics 自带，无需新依赖）。
    失败时返回 None，调用方降级为纯人物检测。"""
    from ultralytics import YOLO
    try:
        print("正在加载姿态模型 yolov8n-pose（首次运行会自动下载，约 6MB）…")
        return YOLO("yolov8n-pose.pt")
    except Exception as e:  # noqa: BLE001 —— 任何加载失败都降级
        print(f"提示：姿态模型加载失败（{e}），已降级为纯人物检测。")
        return None


# ---------------------------------------------------------------- v2：文件日期
def parse_file_datetime(path):
    """从文件名解析日期时间，失败则用文件 mtime。

    支持：20261018_093000 / 20261018093000 / 2026-10-18 /
         10月18日 / 10-18 等常见监控文件名格式。
    """
    name = os.path.basename(path)
    pats = [
        r"(\d{4})(\d{2})(\d{2})[^\d]?(\d{2})(\d{2})(\d{2})",  # 20261018_093000
        r"(\d{4})-(\d{1,2})-(\d{1,2})",                       # 2026-10-18
        r"(\d{1,2})月(\d{1,2})[日号]?",                       # 10月18日
        r"(?<!\d)(\d{1,2})-(\d{1,2})(?!\d)",                  # 10-18
    ]
    for p in pats:
        m = re.search(p, name)
        if not m:
            continue
        g = m.groups()
        try:
            if len(g) == 6:
                return datetime(*map(int, g))
            if len(g) == 3 and len(g[0]) == 4:
                return datetime(int(g[0]), int(g[1]), int(g[2]))
            # 无年份：按当年算，若为未来则视为去年
            dt = datetime(datetime.now().year, int(g[-2]), int(g[-1]))
            if dt > datetime.now():
                dt = dt.replace(year=dt.year - 1)
            return dt
        except ValueError:
            continue
    # 兜底：文件修改时间
    return datetime.fromtimestamp(os.path.getmtime(path))


def list_video_files(dir_path):
    """列出目录下的视频文件（按文件名排序）。"""
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))
             if f.lower().endswith(VIDEO_EXTS)
             and os.path.isfile(os.path.join(dir_path, f))]
    return files


# ---------------------------------------------------------------- v2：扫描
def scan_video_v2(video_path, every, conf, mode, bend_angle, bend_secs, gap,
                  det_model=None, pose_model=None):
    """v2 扫描单个视频。

    mode='person'：人物检测；mode='bend'：人物+弯腰启发式（需 pose_model）。
    返回 segments 列表，每项 {"start","end","thumb_t","box","angle"}。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告：打不开视频，已跳过：{video_path}")
        return [], 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0
    step = max(1, int(round(every * fps)))

    hits = []  # (t, box, angle)
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if mode == "bend" and pose_model is not None:
            for box, angle, bending in analyze_frame_bend(
                    pose_model, frame, conf, bend_angle):
                if bending:
                    hits.append((t, box, angle))
                    break  # 一帧只记一次
        else:
            results = det_model.predict(frame, verbose=False, conf=conf)
            people = [b for b in results[0].boxes if int(b.cls[0]) == 0]
            if people:
                # 取最大的人物框（离镜头最近，最可能看清"谁"）
                b = max(people, key=lambda x: float(
                    (x.xyxy[0][2] - x.xyxy[0][0]) *
                    (x.xyxy[0][3] - x.xyxy[0][1])))
                box = tuple(map(int, b.xyxy[0].tolist()))
                hits.append((t, box, None))
        idx += step
    cap.release()

    # 合并为片段
    segments = []
    for t, box, angle in hits:
        if segments and t - segments[-1]["end"] <= gap + 1e-6:
            segments[-1]["end"] = t
        else:
            segments.append({"start": t, "end": t, "thumb_t": t,
                             "box": box, "angle": angle})
    segments = [{**s, "end": min(s["end"] + every, duration)} for s in segments]
    # 弯腰模式：过滤掉持续不足 bend_secs 的短片段（启发式要求"连续"弯腰）
    if mode == "bend":
        segments = [s for s in segments
                    if s["end"] - s["start"] >= bend_secs - 1e-6]
    return segments, duration


def save_thumb_crop(video_path, t, box, out_path, width=480):
    """保存人物裁剪缩略图（让人眼能辨认"谁"），失败返回 False。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    # 向外扩展 25% padding，保证头脚不被裁掉
    pw, ph = (x2 - x1) * 0.25, (y2 - y1) * 0.25
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(w, int(x2 + pw)), min(h, int(y2 + ph))
    if x2 - x1 < 60 or y2 - y1 < 60:  # 框太小就用整帧
        crop, ox, oy = frame, 0, 0
    else:
        crop, ox, oy = frame[y1:y2, x1:x2], x1, y1
    # 在裁剪图上画出检测框
    cv2.rectangle(crop, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), (0, 255, 0), 2)
    ch, cw = crop.shape[:2]
    crop = cv2.resize(crop, (width, int(ch * width / cw)))
    cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def extract_merged(ff, jobs, out_path):
    """用 ffmpeg 把多个 (video_path, start, end) 片段 concat 合并为一个 mp4。

    jobs: [(video_path, s, e), ...]。返回 True/False。"""
    if not jobs:
        return False
    tmpdir = os.path.dirname(out_path)
    parts, ok = [], True
    try:
        for i, (vp, s, e) in enumerate(jobs):
            part = os.path.join(tmpdir, f"_xpart_{i:03d}.mp4")
            r = subprocess.run([ff, "-y", "-v", "error",
                                "-ss", f"{s:.2f}", "-t", f"{max(e - s, 0.5):.2f}",
                                "-i", vp, "-c", "copy", part])
            if r.returncode != 0 or not os.path.isfile(part):
                ok = False
                break
            parts.append(part)
        if ok:
            lst = os.path.join(tmpdir, "_xparts.txt")
            with open(lst, "w", encoding="utf-8") as f:
                for p in parts:
                    f.write(f"file '{p}'\n")
            r = subprocess.run([ff, "-y", "-v", "error", "-f", "concat",
                                "-safe", "0", "-i", lst, "-c", "copy", out_path])
            os.remove(lst)
            ok = r.returncode == 0 and os.path.isfile(out_path)
    finally:
        for p in parts:
            if os.path.isfile(p):
                os.remove(p)
    return ok


# ---------------------------------------------------------------- v2：报告
def write_report_v2(out_dir, query_text, q, groups, extract_name):
    """多文件/查询模式的报告：按日期分组，章节点击跳转（可跨文件）。"""
    total = sum(len(g["segments"]) for g in groups)
    mode_label = "弯腰（疑似摘菜）检测" if q["mode"] == "bend" else "人物检测"
    head = (f"<p class='info'>查询：{query_text} ｜ "
            f"日期 {q['date_from']} ~ {q['date_to']} ｜ {mode_label} ｜ "
            f"共 {len(groups)} 天有命中、{total} 个片段</p>"
            if query_text else
            f"<p class='info'>{mode_label} ｜ 共 {len(groups)} 天有命中、"
            f"{total} 个片段</p>")
    extract_html = (f"<p class='reel'>提取视频：<a href='{extract_name}'>"
                    f"{extract_name}</a></p>" if extract_name else "")
    sections = []
    for g in groups:
        rows = []
        for i, s in enumerate(g["segments"]):
            label = (f"弯腰 {s['angle']:.0f}°，" if s.get("angle") else "")
            rows.append(
                f"<div class='ch' data-t='{s['start']:.2f}' "
                f"data-v='{s['video_rel']}'>"
                f"<img src='{s['thumb']}' alt='片段{i + 1}'>"
                f"<div class='meta'><b>{label}{fmt_time(s['start'])} → "
                f"{fmt_time(s['end'])}（{s['end'] - s['start']:.0f} 秒）</b>"
                f"<span>{g['file']}</span>"
                f"<em>点击跳转 ▶</em></div></div>")
        sections.append(f"<h2>📅 {g['date']}（{len(g['segments'])} 个片段）</h2>"
                        + "\n".join(rows))
    body = "\n".join(sections) if sections else \
        "<p class='empty'>该查询没有命中任何片段。</p>"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>监控视频找人 v2</title>
<style>
body{{font-family:"Microsoft YaHei",PingFang SC,system-ui,sans-serif;
max-width:960px;margin:0 auto;padding:16px;background:#f5f6f8;color:#222}}
h1{{font-size:20px}}h2{{font-size:17px;margin-top:24px}}
video{{width:100%;background:#000;border-radius:8px}}
.info{{color:#666;font-size:14px}}.ch{{display:flex;gap:12px;align-items:center;
background:#fff;border-radius:8px;padding:8px;margin:8px 0;cursor:pointer;
box-shadow:0 1px 3px rgba(0,0,0,.08)}}.ch:hover{{background:#eef4ff}}
.ch img{{width:200px;border-radius:4px}}.ch .meta b{{display:block;font-size:16px}}
.ch .meta span{{color:#666;font-size:13px}}.ch .meta em{{display:block;
color:#1a73e8;font-style:normal;font-size:13px;margin-top:4px}}
.empty{{color:#999;padding:24px;text-align:center}}.reel a{{color:#1a73e8}}
.warn{{background:#fff8e1;border-radius:8px;padding:10px;font-size:13px;color:#7a5c00}}
</style></head><body>
<h1>🎥 监控视频找人 v2</h1>
{head}
{extract_html}
<p class="warn">说明：本工具不做人脸识别，"谁"请通过缩略图人工辨认；
弯腰判定为启发式（肩-髋前倾角），会有误报漏报，仅供筛查。</p>
<video id="v" controls preload="metadata"></video>
{body}
<script>
var v = document.getElementById('v');
document.querySelectorAll('.ch').forEach(function(el){{
  el.addEventListener('click', function(){{
    var src = el.getAttribute('data-v');
    if (v.getAttribute('src') !== src) {{ v.setAttribute('src', src); }}
    v.currentTime = parseFloat(el.getAttribute('data-t'));
    v.play();
  }});
}});
</script>
</body></html>"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def run_v2(args, is_dir):
    """v2：目录批量 / 单文件+查询。"""
    from ultralytics import YOLO

    # ---- 查询解析 ----
    q = None
    if args.query:
        q = parse_query(args.query)
        print(f"查询解析：{args.query} → "
              f"{q['date_from']} ~ {q['date_to']}，"
              f"模式={'弯腰检测' if q['mode'] == 'bend' else '人物检测'}")
    mode = args.mode if args.mode != "auto" else (q["mode"] if q else "person")

    # ---- 目标文件 ----
    target = os.path.abspath(args.video)
    if is_dir:
        files = list_video_files(target)
        stem = os.path.basename(target.rstrip(os.sep)) or "视频文件夹"
    else:
        files = [target]
        stem = os.path.splitext(os.path.basename(target))[0]

    # ---- 日期过滤 ----
    if q:
        kept = [f for f in files
                if q["date_from"] <= parse_file_datetime(f).date() <= q["date_to"]]
        dropped = len(files) - len(kept)
        if dropped:
            print(f"日期过滤：{len(files)} 个文件 → {len(kept)} 个在范围内"
                  f"（{dropped} 个超出 {q['date_from']}~{q['date_to']}，已跳过）")
        files = kept
    if not files:
        print("没有需要扫描的视频文件。")
        sys.exit(0)

    out_dir = os.path.abspath(args.out) if args.out else \
        os.path.join(target if is_dir else os.path.dirname(target),
                     f"{stem}_找人结果")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 加载模型 ----
    print("正在加载 YOLO 模型（首次运行会自动下载 yolov8n.pt，约 6MB）…")
    det_model = YOLO("yolov8n.pt")
    pose_model = None
    if mode == "bend":
        pose_model = load_pose_model()
        if pose_model is None:
            mode = "person"  # 优雅降级

    # ---- 逐文件扫描 ----
    groups = []  # 按日期分组
    for vf in files:
        fdate = parse_file_datetime(vf).date().isoformat()
        print(f"\n扫描：{os.path.basename(vf)}（{fdate}）…")
        segments, duration = scan_video_v2(
            vf, args.every, args.conf, mode,
            args.bend_angle, args.bend_secs, args.gap, det_model, pose_model)
        print(f"  → {len(segments)} 个片段")
        if not segments:
            continue
        vrel = os.path.relpath(vf, out_dir).replace(os.sep, "/")
        # 缩略图名带上文件标识，避免同日期多文件重名覆盖
        fstem = sanitize_filename(os.path.splitext(os.path.basename(vf))[0])
        for i, s in enumerate(segments):
            th = f"thumb_{fdate}_{fstem}_{i + 1:02d}.jpg"
            save_thumb_crop(vf, s["thumb_t"], s["box"],
                            os.path.join(out_dir, th))
            s["thumb"], s["video_rel"] = th, vrel
        groups.append({"date": fdate, "file": os.path.basename(vf),
                       "segments": [{"start": round(s["start"], 2),
                                     "end": round(s["end"], 2),
                                     "thumb": s["thumb"],
                                     "video_rel": s["video_rel"],
                                     "angle": (round(s["angle"], 1)
                                               if s["angle"] else None)}
                                    for s in segments]})
    groups.sort(key=lambda g: g["date"])

    # ---- segments.json ----
    with open(os.path.join(out_dir, "segments.json"), "w",
              encoding="utf-8") as f:
        json.dump({"query": args.query, "mode": mode,
                   "date_from": str(q["date_from"]) if q else None,
                   "date_to": str(q["date_to"]) if q else None,
                   "groups": groups}, f, ensure_ascii=False, indent=2)

    # ---- 提取合并（--query 默认开启，可用 --no-extract 关闭）----
    extract_name = None
    ff = find_ffmpeg()
    want_extract = (not args.no_extract) and (args.query or args.reel)
    if want_extract:
        if not ff:
            print("提示：未找到 ffmpeg，跳过视频提取。")
        else:
            total_seg = sum(len(g["segments"]) for g in groups)
            if total_seg:
                base = sanitize_filename(
                    args.query) if args.query else f"{stem}_有人集锦"
                extract_name = f"{base}_提取.mp4"
                jobs = []
                for g in groups:
                    for s in g["segments"]:
                        # video_rel 是相对 out_dir 的路径，还原绝对路径
                        vabs = os.path.normpath(
                            os.path.join(out_dir, s["video_rel"]))
                        jobs.append((vabs, s["start"], s["end"]))
                print(f"正在提取合并 {total_seg} 个片段 → {extract_name}…")
                if extract_merged(ff, jobs,
                                  os.path.join(out_dir, extract_name)):
                    print(f"已保存提取视频：{os.path.join(out_dir, extract_name)}")
                else:
                    print("提取失败（ffmpeg 出错），报告不受影响。")
                    extract_name = None

    write_report_v2(out_dir, args.query, q or
                    {"date_from": "全部", "date_to": "全部", "mode": mode},
                    groups, extract_name)
    print(f"\n完成！报告：{os.path.join(out_dir, 'report.html')}")
    print("用浏览器打开 report.html，点击任意片段即可跳转播放。")


def main():
    ap = argparse.ArgumentParser(
        description="监控视频找人 v2：扫描录像，定位人物/弯腰（摘菜）片段，"
                    "支持自然语言查询与多文件批量。"
    )
    ap.add_argument("video", help="视频文件，或视频文件夹（自动切换目录模式）")
    ap.add_argument("--every", type=float, default=1.0,
                    help="每隔多少秒抽一帧检测（默认 1.0；越小越准越慢）")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="检测置信度阈值（默认 0.5；调低可减少漏检，调高可减少误检）")
    ap.add_argument("--gap", type=float, default=3.0,
                    help="两次检测间隔超过多少秒视为两个片段（默认 3.0）")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认：视频旁/文件夹内“<名>_找人结果”）")
    ap.add_argument("--reel", action="store_true",
                    help="（v1 单文件）顺带剪出只含有人片段的集锦视频（需 ffmpeg）")
    # ---- v2 新增 ----
    ap.add_argument("--dir", action="store_true",
                    help="强制按目录模式处理第一个参数")
    ap.add_argument("--query", default=None,
                    help='自然语言查询，如 "这两天谁在摘菜"；目录模式下按日期过滤并提取')
    ap.add_argument("--mode", default="auto", choices=("auto", "person", "bend"),
                    help="检测模式：auto=跟随查询（默认），person=人物，bend=弯腰")
    ap.add_argument("--bend-angle", type=float, default=35.0,
                    help="弯腰判定：躯干前倾角阈值（度，默认 35）")
    ap.add_argument("--bend-secs", type=float, default=3.0,
                    help="弯腰判定：持续秒数阈值（默认 3.0）")
    ap.add_argument("--no-extract", action="store_true",
                    help="关闭 --query 默认的片段提取合并")
    args = ap.parse_args()

    target = os.path.abspath(args.video)
    is_dir = args.dir or os.path.isdir(target)

    # v1 兼容：单文件且无 --query → 走原逻辑
    if not is_dir and not args.query and args.mode == "auto":
        run_v1(args)
    else:
        if not is_dir and not os.path.isfile(target):
            print(f"错误：找不到文件或目录：{args.video}")
            sys.exit(1)
        run_v2(args, is_dir)


if __name__ == "__main__":
    main()
