#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控视频找人 detect_people.py
扫描一段录好的监控视频，用 YOLO 检测“人”出现的时间段，
生成可点击跳转的 HTML 报告（report.html）。

用法:
    python detect_people.py 监控录像.mp4
    python detect_people.py 监控录像.mp4 --every 0.5 --conf 0.4 --out 结果目录 --reel

依赖: pip install -r requirements.txt
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2


def fmt_time(sec):
    """秒数 -> HH:MM:SS"""
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_ffmpeg():
    return shutil.which("ffmpeg")


def main():
    ap = argparse.ArgumentParser(
        description="监控视频找人：扫描录像，定位有人出现的时间段，生成可跳转报告。"
    )
    ap.add_argument("video", help="监控视频文件路径（mp4/avi/webm 等）")
    ap.add_argument("--every", type=float, default=1.0,
                    help="每隔多少秒抽一帧检测（默认 1.0；越小越准越慢）")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="检测置信度阈值（默认 0.5；调低可减少漏检，调高可减少误检）")
    ap.add_argument("--gap", type=float, default=3.0,
                    help="两次检测间隔超过多少秒视为两个片段（默认 3.0）")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认：视频同目录下“<文件名>_找人结果”）")
    ap.add_argument("--reel", action="store_true",
                    help="顺带剪出只含有人片段的集锦视频（需要系统安装 ffmpeg）")
    args = ap.parse_args()

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
            h, w = frame.shape[:2]
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
            f'<div class="ch" data-t="{s:.2f}">'
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


if __name__ == "__main__":
    main()
