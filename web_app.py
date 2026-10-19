#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控视频找人 v2.1 网页版
本地 FastAPI 服务：在浏览器里填文件夹路径、输入查询，点按钮就开检。
不上传任何文件到网上——浏览器和检测服务跑在同一台机器上。

启动：
    python web_app.py                  # 自动打开 http://127.0.0.1:8000
    python web_app.py --port 8123      # 换端口
    python web_app.py --no-browser     # 不自动打开浏览器（服务器/测试用）

依赖：pip install -r requirements.txt（含 fastapi、uvicorn）
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

import uvicorn

# ---- 基本路径 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECT_PY = os.path.join(BASE_DIR, "detect_people.py")
# 与 detect_people.py 的目录扫描保持一致（不递归，只看顶层）
VIDEO_EXTS = (".mp4", ".avi", ".webm", ".mov", ".mkv", ".flv",
              ".mpg", ".mpeg", ".ts")

app = FastAPI(title="监控视频找人·网页版")

# 任务表：job_id -> dict，只保存在内存里，重启服务即清空
jobs = {}
jobs_lock = threading.Lock()

PROG_RE = re.compile(r"进度\s*([\d.]+)\s*%")
SCAN_RE = re.compile(r"扫描：(.+?)（")  # v2 目录模式：\n扫描：文件名（日期）…
URL_RE = re.compile(r"((?:src|href|data-v)=['\"])([^'\"]+)(['\"])")

# ---------------------------------------------------------------- 页面
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>监控视频找人 · 网页版</title>
<style>
body{font-family:"Microsoft YaHei",PingFang SC,system-ui,sans-serif;max-width:900px;
margin:0 auto;padding:20px;background:#f5f6f8;color:#222}
h1{font-size:22px}h1 small{color:#888;font-weight:normal;font-size:14px}
.card{background:#fff;border-radius:10px;padding:16px;margin:14px 0;
box-shadow:0 1px 4px rgba(0,0,0,.08)}
label{display:block;margin:10px 0 4px;color:#555;font-size:14px}
input{width:100%;box-sizing:border-box;padding:10px;border:1px solid #ccc;
border-radius:6px;font-size:15px}
button{background:#1a73e8;color:#fff;border:none;border-radius:6px;
padding:12px 28px;font-size:16px;cursor:pointer;margin-top:14px}
button:disabled{background:#9ec1f5;cursor:wait}
.bar{height:14px;background:#e8eaed;border-radius:7px;overflow:hidden;margin:10px 0}
.bar>div{height:100%;width:0;background:#1a73e8;transition:width .3s}
#log{background:#222;color:#cfd8dc;border-radius:6px;padding:10px;font-size:12px;
max-height:220px;overflow:auto;white-space:pre-wrap}
#report{width:100%;height:640px;border:1px solid #ddd;border-radius:6px;background:#fff}
.hint{color:#888;font-size:13px;margin-top:8px}
#dl{display:inline-block;margin:10px 0;color:#1a73e8;font-size:15px}
.err{color:#c62828}
</style></head>
<body>
<h1>🎥 监控视频找人 <small>网页版 v2.1（本机运行，不上传）</small></h1>
<div class="card">
  <label>视频文件夹（填<b>本机</b>路径，例如 Windows 填 <code>D:\\录像</code>，
  macOS/Linux 填 <code>/home/xxx/录像</code>；也可以直接填单个视频文件路径）</label>
  <input id="folder" placeholder="D:\\录像" onblur="refreshEstimate()">
  <label>查询（想找什么）</label>
  <input id="query" value="这两天谁在摘菜">
  <label style="margin-top:12px"><input id="fast" type="checkbox" style="width:auto">
  ⚡ 极速模式（速度翻倍、精度略降，大文件/长时间录像推荐勾选）</label>
  <div class="hint" id="estimate">在上方填写文件夹路径后，这里会显示预估耗时。</div>
  <button id="start" onclick="startJob()">开始检测</button>
  <div class="hint">检测在后台运行，页面会实时显示进度；中途可以最小化浏览器，不影响。
  首次运行需下载 AI 模型（约十几 MB），请耐心等待。</div>
</div>
<div class="card" id="progCard" hidden>
  <h2>检测进度</h2>
  <div class="bar"><div id="bar"></div></div>
  <div id="status">准备中…</div>
  <pre id="log"></pre>
</div>
<div class="card" id="resCard" hidden>
  <h2>检测结果</h2>
  <div><a id="dl" hidden>⬇ 下载提取视频</a></div>
  <iframe id="report" title="检测报告"></iframe>
</div>
<script>
let timer=null;
async function refreshEstimate(){
  const folder=document.getElementById('folder').value.trim();
  const box=document.getElementById('estimate');
  if(!folder){box.textContent='在上方填写文件夹路径后，这里会显示预估耗时。';return;}
  box.textContent='正在估算…';
  try{
    const r=await fetch('/api/estimate',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({folder:folder})});
    const d=await r.json();
    if(!r.ok)throw new Error(d.detail||('估算失败('+r.status+')'));
    const mins=Math.round(d.total_seconds/60);
    box.textContent='共 '+d.file_count+' 个视频，总时长约 '+mins+' 分钟；'
      +'普通模式预计 '+d.eta_normal+'，极速模式预计 '+d.eta_fast+'。';
  }catch(e){box.textContent='估算失败：'+e.message+'（不影响检测）';}
}
async function startJob(){
  const folder=document.getElementById('folder').value.trim();
  const query=document.getElementById('query').value.trim()||'这两天谁在摘菜';
  const fast=document.getElementById('fast').checked;
  if(!folder){alert('请先填写视频文件夹路径');return;}
  const btn=document.getElementById('start');btn.disabled=true;
  document.getElementById('progCard').hidden=false;
  document.getElementById('resCard').hidden=true;
  setStatus('任务创建中…');setLog('');
  try{
    const r=await fetch('/api/jobs',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({folder:folder,query:query,fast:fast})});
    const d=await r.json();
    if(!r.ok)throw new Error(d.detail||('创建任务失败('+r.status+')'));
    poll(d.id);
  }catch(e){setStatus('❌ '+e.message,true);btn.disabled=false;}
}
function setStatus(t,err){const s=document.getElementById('status');
  s.textContent=t;if(err)s.className='err';else s.className='';}
function setLog(t){document.getElementById('log').textContent=t;}
async function poll(id){
  clearInterval(timer);
  timer=setInterval(async ()=>{
    try{
      const r=await fetch('/api/jobs/'+id);const j=await r.json();
      document.getElementById('bar').style.width=j.progress+'%';
      let where='';
      if(j.total_files>1){
        where='第 '+Math.min(j.files_done+1,j.total_files)+'/'+j.total_files+' 个文件'
          +(j.current_file?('：'+j.current_file):'')+'；';
      }else if(j.file_pct>0){
        where='本文件 '+j.file_pct+'%；';
      }
      setStatus(j.status==='done'?'✅ 完成'
        :(j.status==='error'?'❌ 出错，请看日志':'⏳ 检测中… '+where+j.progress+'%'));
      setLog(j.log_tail);
      if(j.status==='done'||j.status==='error'){
        clearInterval(timer);document.getElementById('start').disabled=false;
        if(j.status==='done'&&j.has_report){
          document.getElementById('resCard').hidden=false;
          document.getElementById('report').src='/api/jobs/'+id+'/report';
          const dl=document.getElementById('dl');
          if(j.extract){dl.hidden=false;dl.href='/api/jobs/'+id+'/download';}
          else{dl.hidden=true;}
          document.getElementById('resCard').scrollIntoView();
        }
      }
    }catch(e){setStatus('❌ 轮询失败：'+e.message,true);
      clearInterval(timer);document.getElementById('start').disabled=false;}
  },2000);
}
</script>
</body></html>
"""


# ---------------------------------------------------------------- 任务逻辑
def _count_videos(path):
    """统计待扫视频数（与 detect_people.py 的目录扫描口径一致），用于进度条分母。"""
    if os.path.isfile(path):
        return 1
    try:
        return sum(1 for f in os.listdir(path)
                   if f.lower().endswith(VIDEO_EXTS)
                   and os.path.isfile(os.path.join(path, f)))
    except OSError:
        return 0


def _append_log(job, line):
    job["log"].append(line)
    if len(job["log"]) > 300:  # 只保留最近 300 行，防止内存膨胀
        del job["log"][:len(job["log"]) - 300]


def _run_job(jid):
    """后台线程：调 subprocess 跑 detect_people.py，逐行解析进度。"""
    job = jobs[jid]
    cmd = [sys.executable, DETECT_PY, job["folder"],
           "--query", job["query"], "--out", job["out_dir"]]
    if job.get("fast"):
        cmd += ["--every", "2"]  # 极速模式：速度翻倍、精度略降
    try:
        # cwd 必须设为脚本目录：模型文件 yolov8n.pt 用的是相对路径
        p = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             bufsize=1, errors="replace")
    except Exception as e:  # noqa: BLE001 - 给用户看的中文提示
        job["status"] = "error"
        _append_log(job, f"启动检测失败：{e}")
        return
    for line in p.stdout:
        line = line.rstrip("\n")
        _append_log(job, line)
        m = PROG_RE.search(line)  # v1 单文件模式的 "进度 12.3%"
        if m:
            try:
                job["progress"] = max(job["progress"], float(m.group(1)))
                job["file_pct"] = float(m.group(1))
            except ValueError:
                pass
        sm = SCAN_RE.search(line)  # v2 目录模式：开始扫一个文件
        if sm:
            job["current_file"] = sm.group(1)
            job["file_pct"] = 0.0
        if re.search(r"→\s*\d+\s*个片段", line):  # 该文件扫完
            job["files_done"] += 1
            if job["total_files"]:
                job["progress"] = max(job["progress"],
                                      job["files_done"] / job["total_files"] * 100)
        if "完成！" in line:
            job["progress"] = 100.0
    rc = p.wait()
    if rc == 0:
        job["status"] = "done"
        job["progress"] = 100.0
        try:  # 找出提取视频（有片段时才会生成）
            for f in sorted(os.listdir(job["out_dir"])):
                if f.endswith("提取.mp4") or f.endswith("有人集锦.mp4"):
                    job["extract"] = f
                    break
        except OSError:
            pass
    else:
        job["status"] = "error"
        _append_log(job, f"检测进程异常退出（exit={rc}），请查看上方日志。")


def _new_job(folder, query, fast=False):
    jid = uuid.uuid4().hex[:12]
    out_dir = tempfile.mkdtemp(prefix="personseek_job_")  # 每个任务独立输出目录
    job = {"id": jid, "folder": folder, "query": query, "fast": fast,
           "out_dir": out_dir,
           "status": "running", "progress": 0.0, "log": [], "extract": None,
           "total_files": _count_videos(folder), "files_done": 0,
           "current_file": "", "file_pct": 0.0,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with jobs_lock:
        jobs[jid] = job
    threading.Thread(target=_run_job, args=(jid,), daemon=True).start()
    return job


def _get_job(jid):
    with jobs_lock:
        job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "任务不存在（服务重启后任务会清空）")
    return job


# ---------------------------------------------------------------- 接口
@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/api/jobs")
def create_job(body: dict):
    folder = (body.get("folder") or "").strip().strip('"').strip("'")
    query = (body.get("query") or "").strip() or "这两天谁在摘菜"
    fast = bool(body.get("fast", False))  # 极速模式：--every 2
    if not folder:
        raise HTTPException(400, "请填写视频文件夹路径")
    if not os.path.exists(folder):
        raise HTTPException(400, f"找不到该路径：{folder}")
    job = _new_job(folder, query, fast)
    return {"id": job["id"]}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    """供前端轮询：状态、进度、日志尾部、是否有报告/提取视频。"""
    job = _get_job(jid)
    return {
        "id": jid,
        "status": job["status"],
        "progress": round(job["progress"], 1),
        "log_tail": "\n".join(job["log"][-30:]),
        "extract": job["extract"],
        "has_report": os.path.isfile(os.path.join(job["out_dir"], "report.html")),
        "query": job["query"],
        "fast": job["fast"],
        "files_done": job["files_done"],
        "total_files": job["total_files"],
        "current_file": job["current_file"],
        "file_pct": round(job["file_pct"], 1),
    }


def _probe_duration(path):
    """快速读取视频时长（秒），只读文件头，不解码。"""
    import cv2  # 延迟导入，保持服务启动速度
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frames > 0:
            return frames / fps
    except Exception:  # noqa: BLE001 - 读不到就按 0 算
        pass
    return 0.0


def _fmt_eta(sec):
    sec = max(1, int(sec))
    if sec < 90:
        return f"约 {sec} 秒"
    return f"约 {sec / 60:.0f} 分钟"


@app.post("/api/estimate")
def estimate(body: dict):
    """开始前预估耗时：统计待扫视频总时长，按每秒处理约 8 帧估算。
    文案一律带"仅供参考"——实际速度随机器性能、画面复杂度浮动，
    首次运行还要下载模型（约十几 MB）。"""
    folder = (body.get("folder") or "").strip().strip('"').strip("'")
    if not folder or not os.path.exists(folder):
        raise HTTPException(400, f"找不到该路径：{folder}")
    if os.path.isdir(folder):
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(VIDEO_EXTS)
                 and os.path.isfile(os.path.join(folder, f))]
    else:
        files = [folder]
    total = sum(_probe_duration(f) for f in files)

    def _eta(every):
        return _fmt_eta(total / every / 8) + "（仅供参考）"

    return {"file_count": len(files),
            "total_seconds": round(total),
            "eta_normal": _eta(1.0),
            "eta_fast": _eta(2.0)}


def _rewrite_report(html, base):
    """把报告里的相对资源路径（缩略图、视频、提取视频链接）改写为本服务的接口地址，
    使报告能内嵌在 iframe 里正常显示。绝对/外链地址原样保留。"""

    def _rep(m):
        pre, url, post = m.group(1), m.group(2), m.group(3)
        if url.startswith(("http://", "https://", "/", "#", "data:")):
            return m.group(0)
        return f"{pre}{base}/{url.lstrip('./')}{post}"

    return URL_RE.sub(_rep, html)


@app.get("/api/jobs/{jid}/report", response_class=HTMLResponse)
def job_report(jid: str):
    """直接复用 detect_people.py 生成的 report.html 内容（改写资源路径后内嵌展示）。"""
    job = _get_job(jid)
    rp = os.path.join(job["out_dir"], "report.html")
    if not os.path.isfile(rp):
        raise HTTPException(404, "报告尚未生成")
    with open(rp, encoding="utf-8") as f:
        html = f.read()
    return _rewrite_report(html, f"/api/jobs/{jid}/files")


@app.get("/api/jobs/{jid}/files/{fpath:path}")
def job_file(jid: str, fpath: str):
    """提供报告内引用的文件：缩略图、提取视频，以及 data-v 指向的原始视频。
    说明：服务只监听 127.0.0.1（本机），data-v 的 ../ 路径会解析到用户自己的
    视频文件，这是内嵌播放必需的；不要把本服务暴露到公网。"""
    job = _get_job(jid)
    out = job["out_dir"]
    cand = os.path.normpath(os.path.join(out, fpath))
    if os.path.isfile(cand):
        return FileResponse(cand)
    if os.path.isabs(fpath) and os.path.isfile(fpath):
        return FileResponse(fpath)
    raise HTTPException(404, "文件不存在")


@app.get("/api/jobs/{jid}/download")
def job_download(jid: str):
    """下载提取视频（查询命中的片段合并版）。"""
    job = _get_job(jid)
    if not job["extract"]:
        raise HTTPException(404, "本次任务没有可下载的提取视频（可能没有命中片段）")
    fp = os.path.join(job["out_dir"], job["extract"])
    return FileResponse(fp, filename=job["extract"])


# ---------------------------------------------------------------- 启动
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="监控视频找人·网页版（本地运行）")
    ap.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()
    try:
        url = f"http://127.0.0.1:{args.port}"
        if not args.no_browser:
            # 延迟一点再打开，等 uvicorn 就绪
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        print(f"监控视频找人·网页版已启动：{url}")
        print("浏览器会自动打开；按 Ctrl+C 退出服务。")
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    except Exception as e:  # noqa: BLE001 - 给用户看的中文提示
        print(f"启动失败：{e}")
        print("请确认：已执行 pip install -r requirements.txt；端口未被占用。")
        sys.exit(1)
