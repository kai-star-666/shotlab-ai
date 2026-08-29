"""Web 页面主题与不依赖 Streamlit 的上传校验。"""

from __future__ import annotations

from pathlib import Path


MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}


def validate_upload(filename: str, size: int) -> str | None:
    """返回面向用户的上传错误；合法时返回 ``None``。"""
    if size <= 0:
        return "视频是空文件，请重新选择。"
    if Path(filename).suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        return "暂时只支持 MP4、MOV、AVI；优先上传 MP4/H.264。"
    if size > MAX_UPLOAD_BYTES:
        return f"视频超过 {MAX_UPLOAD_MB} MB，请先裁剪或压缩后再上传。"
    return None


APP_CSS = r"""
<style>
:root {
  --shot-bg: #071017;
  --shot-panel: rgba(14, 26, 36, .88);
  --shot-panel-strong: #101e29;
  --shot-line: rgba(151, 170, 185, .18);
  --shot-text: #f5f8fa;
  --shot-muted: #9cb0bf;
  --shot-orange: #ff6b22;
  --shot-orange-2: #ff934d;
  --shot-teal: #35d0ba;
}

html { scroll-behavior: smooth; }
.stApp {
  background:
    radial-gradient(circle at 88% -4%, rgba(255,107,34,.19), transparent 32rem),
    radial-gradient(circle at 2% 38%, rgba(53,208,186,.10), transparent 30rem),
    var(--shot-bg);
  color: var(--shot-text);
}
[data-testid="stHeader"] { background: rgba(7,16,23,.72); backdrop-filter: blur(14px); }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1180px; padding: 2.2rem 2rem 5rem; }
h1, h2, h3 { letter-spacing: -.025em; }
h2 { margin-top: 2.7rem !important; font-size: 1.55rem !important; }
p, li, label { line-height: 1.65; }

.shot-hero {
  position: relative;
  overflow: hidden;
  padding: 3rem;
  margin: .2rem 0 1.25rem;
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 28px;
  background: linear-gradient(135deg, rgba(17,31,43,.98), rgba(9,18,26,.94));
  box-shadow: 0 30px 80px rgba(0,0,0,.28);
}
.shot-hero:after {
  content: "";
  position: absolute;
  right: -70px;
  top: -92px;
  width: 290px;
  height: 290px;
  border-radius: 50%;
  border: 42px solid rgba(255,107,34,.10);
  box-shadow: 0 0 0 30px rgba(255,107,34,.035);
}
.shot-brand { display:flex; align-items:center; gap:.7rem; color:#dce6ec; font-weight:800; letter-spacing:.14em; font-size:.78rem; }
.shot-brand-mark { width:34px; height:34px; display:grid; place-items:center; border-radius:11px; color:#0a1218; background:linear-gradient(145deg,var(--shot-orange-2),var(--shot-orange)); box-shadow:0 8px 24px rgba(255,107,34,.3); }
.shot-badge { display:inline-flex; margin-top:2rem; padding:.36rem .7rem; border:1px solid rgba(53,208,186,.32); border-radius:999px; color:#78ead9; background:rgba(53,208,186,.08); font-size:.74rem; font-weight:700; letter-spacing:.08em; }
.shot-hero h1 { max-width:760px; margin:.8rem 0 .75rem; font-size:clamp(2.15rem,5vw,4.35rem); line-height:1.02; color:white !important; }
.shot-hero h1 span { color:var(--shot-orange-2) !important; }
.shot-hero p { max-width:690px; margin:0; color:#b7c6d1; font-size:1.05rem; }
.shot-chips { display:flex; flex-wrap:wrap; gap:.65rem; margin-top:1.6rem; }
.shot-chip { padding:.5rem .78rem; border:1px solid var(--shot-line); border-radius:999px; color:#d8e1e7; background:rgba(255,255,255,.035); font-size:.82rem; }

.shot-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:.85rem; margin:1rem 0 1.5rem; }
.shot-step { min-height:104px; padding:1.05rem 1.1rem; border:1px solid var(--shot-line); border-radius:18px; background:rgba(14,26,36,.66); }
.shot-step strong { display:block; color:white; margin-bottom:.26rem; }
.shot-step span { color:var(--shot-muted); font-size:.86rem; }
.shot-num { color:var(--shot-orange-2) !important; font-weight:800; letter-spacing:.08em; }

[data-testid="stFileUploader"] { padding:1.15rem; border:1px solid rgba(255,107,34,.28); border-radius:22px; background:linear-gradient(145deg,rgba(255,107,34,.07),rgba(16,30,41,.86)); }
[data-testid="stFileUploaderDropzone"] { min-height:145px; border:1px dashed rgba(255,147,77,.55); border-radius:16px; background:rgba(5,12,18,.42); }
[data-testid="stFileUploaderDropzone"] button { border-color:rgba(255,147,77,.42); }

.stButton > button[kind="primary"], .stDownloadButton > button {
  min-height:3.25rem;
  border:0;
  border-radius:14px;
  font-weight:800;
  background:linear-gradient(100deg,var(--shot-orange),var(--shot-orange-2));
  color:white;
  box-shadow:0 12px 30px rgba(255,107,34,.20);
  transition:transform .16s ease, box-shadow .16s ease;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 15px 34px rgba(255,107,34,.3); color:white; }

[data-testid="stMetric"] { min-height:112px; padding:1rem 1.05rem; border:1px solid var(--shot-line); border-radius:17px; background:var(--shot-panel); }
[data-testid="stMetricLabel"] { color:var(--shot-muted); }
[data-testid="stMetricValue"] { color:#fff; font-size:1.58rem; }
[data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--shot-line) !important; border-radius:17px !important; }
[data-testid="stAlert"] { border-radius:15px; }
.science-note { border-left:4px solid var(--shot-orange); background:rgba(255,107,34,.08); padding:13px 16px; border-radius:9px; color:#d5e0e7; }
.shot-privacy { margin:.65rem 0 1.3rem; padding:.85rem 1rem; border:1px solid rgba(53,208,186,.2); border-radius:14px; background:rgba(53,208,186,.055); color:#abc1ca; font-size:.84rem; }
.shot-section-kicker { color:var(--shot-orange-2); font-size:.75rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:-.8rem; }

@media (max-width: 768px) {
  .block-container { padding:1rem .9rem 5.5rem; }
  .shot-hero { padding:1.55rem 1.2rem 1.65rem; border-radius:22px; }
  .shot-hero:after { width:180px; height:180px; right:-84px; top:-55px; border-width:28px; }
  .shot-hero h1 { font-size:2.3rem; max-width:92%; }
  .shot-hero p { font-size:.94rem; }
  .shot-badge { margin-top:1.45rem; }
  .shot-chips { gap:.45rem; }
  .shot-chip { font-size:.75rem; padding:.42rem .62rem; }
  .shot-steps { grid-template-columns:1fr; gap:.65rem; }
  .shot-step { min-height:0; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.65rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:calc(50% - .4rem) !important; flex:1 1 calc(50% - .4rem) !important; }
  [data-testid="stMetric"] { min-height:98px; padding:.8rem; }
  [data-testid="stMetricValue"] { font-size:1.24rem; }
  [data-testid="stFileUploader"] { padding:.7rem; }
  h2 { font-size:1.35rem !important; }
}

@media (max-width: 430px) {
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:100% !important; flex-basis:100% !important; }
  .shot-brand { letter-spacing:.1em; }
  .shot-hero h1 { font-size:2rem; }
}
</style>
"""


HERO_HTML = """
<section class="shot-hero">
  <div class="shot-brand"><span class="shot-brand-mark">S</span> SHOTLAB AI</div>
  <div class="shot-badge">V0.4 · WEB BETA</div>
  <h1>把每一次投篮，变成<span>看得见的数据</span></h1>
  <p>上传一段单次或连续投篮视频，自动切分动作、重建人体骨架，并输出关键角度、出手候选、动作问题与可执行训练建议。</p>
  <div class="shot-chips">
    <span class="shot-chip">连续投篮切分</span>
    <span class="shot-chip">二维姿态分析</span>
    <span class="shot-chip">出手点辅助</span>
    <span class="shot-chip">训练建议</span>
  </div>
</section>
"""


STEPS_HTML = """
<section class="shot-steps">
  <div class="shot-step"><span class="shot-num">01 · 拍摄</span><strong>固定机位，全身入镜</strong><span>优先投篮手一侧，人物占画面约 45%–70%。</span></div>
  <div class="shot-step"><span class="shot-num">02 · 上传</span><strong>单球或连续投篮均可</strong><span>优先 MP4/H.264，建议 720p 以上、画面清晰。</span></div>
  <div class="shot-step"><span class="shot-num">03 · 分析</span><strong>获得视频、数据和训练处方</strong><span>结果基于单摄像头二维估计，会明确标注数据边界。</span></div>
</section>
"""
