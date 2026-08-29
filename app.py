"""AI Basketball Shot Analyzer V0.4 Web Beta。"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import tempfile
import traceback

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pose_analyzer import AnalysisError, AnalysisResult, PHASE_LABELS, analyze_video, read_video_info
from report_generator import LIMITATION, generate_report
from web_ui import APP_CSS, HERO_HTML, MAX_UPLOAD_MB, STEPS_HTML, validate_upload

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "pose_landmarker_lite.task"
HAND_MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
BALL_MODEL_PATH = PROJECT_ROOT / "efficientdet_lite0.tflite"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
REQUIRED_ANALYZER_PARAMETERS = {"hand_model_path", "ball_model_path"}

st.set_page_config(
    page_title="ShotLab AI｜篮球投篮动作分析",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)
st.markdown(STEPS_HTML, unsafe_allow_html=True)


def _metric(value: float, unit: str = "°") -> str:
    return f"{value:.1f}{unit}" if math.isfinite(value) else "数据不足"


def _seconds(value: float) -> str:
    return f"{value:.2f} s" if math.isfinite(value) else "数据不足"


def _show_video_info(uploaded_file) -> None:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(uploaded_file.getbuffer())
        temporary_path = Path(temporary.name)
    try:
        info = read_video_info(temporary_path)
        columns = st.columns(5)
        columns[0].metric("文件名", uploaded_file.name)
        columns[1].metric("FPS", f"{info.fps:.2f}")
        columns[2].metric("时长", f"{info.duration:.2f}s")
        columns[3].metric("总帧数", info.frame_count)
        columns[4].metric("分辨率", f"{info.width}×{info.height}")
    except AnalysisError as error:
        st.warning(str(error))
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_analysis(uploaded_file, shooting_hand: str) -> AnalysisResult:
    suffix = Path(uploaded_file.name).suffix.lower()
    progress_bar = st.progress(0.0, text="准备分析环境")
    status = st.empty()

    def update(value: float, message: str) -> None:
        progress_bar.progress(value, text=message)
        status.caption(message)

    with tempfile.TemporaryDirectory(prefix="shot_analyzer_") as temporary_dir:
        input_path = Path(temporary_dir) / f"input{suffix}"
        input_path.write_bytes(uploaded_file.getbuffer())
        result = analyze_video(
            input_path, MODEL_PATH, shooting_hand=shooting_hand, output_root=OUTPUT_ROOT,
            progress=update,
            hand_model_path=HAND_MODEL_PATH if HAND_MODEL_PATH.exists() else None,
            ball_model_path=BALL_MODEL_PATH if BALL_MODEL_PATH.exists() else None,
        )
        result.info = result.info.__class__(uploaded_file.name, result.info.fps, result.info.frame_count, result.info.width, result.info.height, result.info.duration)
    report_path = generate_report(result)
    st.session_state["analysis_result"] = result
    st.session_state["report_path"] = report_path
    progress_bar.progress(1.0, text="分析、视频与报告已生成")
    status.empty()
    return result


def _render_timeline(shot, fps: float) -> None:
    ordered = sorted((shot.timeline_frames or {}).items(), key=lambda item: item[1])
    times = [frame / fps for _, frame in ordered]
    labels = [PHASE_LABELS.get(name, name) for name, _ in ordered]
    figure = go.Figure(go.Scatter(
        x=times, y=[1] * len(times), mode="lines+markers+text",
        line=dict(color="#ff5b57", width=5), marker=dict(size=13, color="#ffd166"),
        text=[f"{label}<br>{time:.2f}s" for label, time in zip(labels, times)],
        textposition="top center", hovertemplate="%{text}<extra></extra>",
    ))
    figure.update_layout(height=250, margin=dict(l=20, r=20, t=65, b=25), xaxis_title="视频时间（秒）", yaxis=dict(visible=False), showlegend=False)
    st.plotly_chart(figure, width="stretch")


def _render_result(result: AnalysisResult, report_path: Path) -> None:
    valid_ratio = result.valid_frames / max(len(result.data), 1)
    release_confidence = result.shots[0].metrics.get("release_confidence", math.nan)
    confidence_text = f"{release_confidence:.0%}" if math.isfinite(release_confidence) else "规则估计"
    st.success(f"分析完成：自动切分 {len(result.shots)} 次投篮，人体姿态、火柴人和报告已生成。")

    st.header("1. 投篮分析总览")
    overview = st.columns(6)
    overview[0].metric("视频", result.info.filename)
    overview[1].metric("时长", f"{result.info.duration:.2f}s")
    overview[2].metric("FPS", f"{result.info.fps:.1f}")
    overview[3].metric("投篮手", result.shooting_hand)
    overview[4].metric("有效 Pose", f"{valid_ratio:.1%}")
    overview[5].metric("出手候选置信", confidence_text)
    st.caption(f"结果目录：{result.output_dir}")

    quality = result.capture_quality
    quality_columns = st.columns((1, 3))
    quality_columns[0].metric("拍摄质量", f"{quality.score}/100", quality.level)
    with quality_columns[1]:
        for issue in quality.issues:
            st.write(f"• {issue}")
        with st.expander("机位与拍摄修正"):
            for correction in quality.corrections:
                st.write(f"• {correction}")
    for note in result.model_notes:
        st.info(note)

    shot_options = {f"第 {shot.shot_id} 球 · 出手候选 {shot.window.release_frame / result.info.fps:.2f}s": shot for shot in result.shots}
    selected = shot_options[st.selectbox("选择逐球分析", list(shot_options))]

    st.header("2. 动作稳定性指标")
    st.caption("评分只用于当前系统内部比较，不代表职业技术等级。单次主要反映动作平顺度，多球才能评价重复性。")
    score_columns = st.columns(4)
    for column, (name, score) in zip(score_columns, (selected.stability_scores or {}).items()):
        column.metric(name, f"{score}/100" if score is not None else "数据不足")

    st.header("3. 实际投篮数据 vs 参考范围")
    st.dataframe(selected.angle_comparison, hide_index=True, width="stretch")
    st.markdown('<div class="science-note">参考范围只用于动作观察，不代表所有球员必须达到完全相同的数值。身高、臂长、投篮距离、节奏与机位都会改变二维测量。</div>', unsafe_allow_html=True)
    st.caption("依据：投篮距离和熟练度会改变关节运动学，因此本系统不设置唯一‘标准角度’。")
    st.markdown("[距离变化的投篮运动学研究（PubMed）](https://pubmed.ncbi.nlm.nih.gov/38314460/) · [熟练者/非熟练者二分与三分投篮研究（PMC）](https://pmc.ncbi.nlm.nih.gov/articles/PMC8822900/)")

    st.header("4. 关键动作帧")
    phase_order = ("ready", "lowest", "set_point", "release_candidate", "followthrough")
    frame_columns = st.columns(5)
    for column, phase in zip(frame_columns, phase_order):
        with column:
            path = selected.keyframes.get(phase)
            if path and path.exists():
                frame = (selected.timeline_frames or {}).get(phase, 0)
                st.image(str(path), caption=f"{PHASE_LABELS[phase]} · {frame / result.info.fps:.2f}s")

    st.header("5. 三种动作视频")
    video_columns = st.columns(3)
    with video_columns[0]:
        st.markdown("**Original Video**")
        st.video(str(result.original_video))
    with video_columns[1]:
        st.markdown("**Pose Overlay Video**")
        st.video(str(result.processed_video))
    with video_columns[2]:
        st.markdown("**Stick Figure Video**")
        st.video(str(result.stick_figure_video))
    with st.expander("篮球轨迹/出手点辅助视频"):
        st.video(str(result.trajectory_video))

    st.header("6. 角度—时间曲线")
    long_data = result.data.melt(id_vars="time", value_vars=["elbow_angle", "knee_angle", "hip_angle", "trunk_angle"], var_name="指标", value_name="角度")
    long_data["指标"] = long_data["指标"].map({"elbow_angle": "肘角", "knee_angle": "膝角", "hip_angle": "髋角", "trunk_angle": "躯干倾角"})
    figure = px.line(long_data, x="time", y="角度", facet_row="指标", color="指标", height=780)
    for phase in ("lowest", "release_candidate", "followthrough"):
        frame = (selected.timeline_frames or {}).get(phase)
        if frame is not None:
            figure.add_vline(x=frame / result.info.fps, line_dash="dash", line_color="#ff5b57", opacity=0.65)
    figure.update_layout(showlegend=False, margin=dict(l=30, r=30, t=30, b=30))
    figure.update_xaxes(title="时间（秒）")
    st.plotly_chart(figure, width="stretch")

    st.header("7. 投篮动作时间轴")
    _render_timeline(selected, result.info.fps)
    st.dataframe([{" 时序指标": name, "我的数据": _seconds(value)} for name, value in (selected.timing or {}).items()], hide_index=True, width="stretch")
    lag = (selected.timing or {}).get("上下肢快速伸展起点时间差", math.nan)
    if math.isfinite(lag):
        st.caption(f"动力链规则观察：肘部快速伸展起点相对膝部 {lag:+.2f}s；这是关节运动时序，不是肌肉发力测量。")

    st.header("8. 当前主要投篮问题")
    problems = selected.problems or []
    if not problems:
        st.info("当前可靠数据中未发现足以输出的明显问题。这不代表动作完美，建议使用同机位多球对比。")
    for index, problem in enumerate(problems, 1):
        with st.container(border=True):
            st.markdown(f"**{index}. {problem.title}**")
            st.write(f"**数据依据：** {problem.evidence}")
            st.write(f"**解释：** {problem.explanation}")

    st.header("9. 动作调整建议")
    for index, item in enumerate(selected.action_suggestions or [], 1):
        st.write(f"{index}. {item}")

    st.header("10. 专项训练建议")
    for index, drill in enumerate(selected.training_drills or [], 1):
        with st.container(border=True):
            st.markdown(f"**训练 {index}：{drill.name}**")
            st.write(f"**重点：** {drill.focus}")
            st.write(f"**方法：** {drill.method}")
            st.write(f"**建议量：** {drill.dosage}")

    st.subheader("手型观察")
    st.info(selected.hand_analysis)
    if len(result.shots) >= 2:
        st.subheader("连续投篮重复性")
        cols = st.columns(3)
        cols[0].metric("出手肘角标准差", _metric(result.consistency.get("elbow_release_std", math.nan)))
        cols[1].metric("最低点膝角标准差", _metric(result.consistency.get("knee_lowest_std", math.nan)))
        cols[2].metric("翻腕变化标准差", _metric(result.consistency.get("wrist_snap_change_std", math.nan)))

    st.header("11. 科学性与技术限制")
    for category, items in result.science_boundaries.items():
        with st.expander(category):
            for item in items:
                st.write(f"• {item}")
    st.warning(LIMITATION)

    st.subheader("下载结果")
    downloads = st.columns(5)
    downloads[0].download_button("下载 HTML 报告", report_path.read_bytes(), file_name="shot_analysis_report.html", mime="text/html")
    downloads[1].download_button("下载逐帧 CSV", result.csv_path.read_bytes(), file_name="pose_data.csv", mime="text/csv")
    downloads[2].download_button("下载投篮汇总 CSV", result.shot_summary_path.read_bytes(), file_name="shot_summary.csv", mime="text/csv")
    downloads[3].download_button("下载 Pose 视频", result.processed_video.read_bytes(), file_name="pose_overlay_video.mp4", mime="video/mp4")
    downloads[4].download_button("下载火柴人视频", result.stick_figure_video.read_bytes(), file_name="stick_figure_video.mp4", mime="video/mp4")
    if result.ball_trajectory_path and result.ball_trajectory_path.exists():
        st.download_button("下载可信篮球轨迹 CSV", result.ball_trajectory_path.read_bytes(), file_name="ball_trajectory.csv", mime="text/csv")
    if selected.clip_path and selected.clip_path.exists():
        st.download_button(f"下载第 {selected.shot_id} 球自动切分片段", selected.clip_path.read_bytes(), file_name=f"shot_{selected.shot_id:02d}.mp4", mime="video/mp4", width="stretch")


analyzer_parameters = set(inspect.signature(analyze_video).parameters)
missing_interface = REQUIRED_ANALYZER_PARAMETERS - analyzer_parameters
with st.sidebar:
    st.markdown("### SHOTLAB AI")
    st.caption("AI Basketball Shot Analyzer · V0.4 Web Beta")
    st.divider()
    st.header("分析设置")
    shooting_hand = st.radio("投篮手", ("右手", "左手"), horizontal=True)
    st.info("支持单次或连续投篮。建议单人全身入镜、人物高度占画面 45%–70%、优先 1080p/60fps。")
    st.caption("侧面更适合关节角/时序；侧前方约 30°–45°更容易观察肘部横向路径。")
    st.divider()
    st.caption("隐私提示：视频只用于本次分析。公开测试阶段请勿上传包含他人隐私或敏感信息的素材。")

if missing_interface:
    st.error("页面与分析引擎版本不一致。请完整重启 Streamlit 服务，不要继续使用旧进程。")
    st.code(f"缺少引擎参数: {', '.join(sorted(missing_interface))}")
if not MODEL_PATH.exists():
    st.error("缺少 pose_landmarker_lite.task，无法启动人体姿态分析。")
if not HAND_MODEL_PATH.exists() or not BALL_MODEL_PATH.exists():
    st.warning("手部或篮球模型缺失：人体分析仍可继续，手型/篮球轨迹将自动降级。")

st.markdown('<div class="shot-section-kicker">START ANALYSIS</div>', unsafe_allow_html=True)
st.subheader("上传你的投篮视频")
st.caption(f"支持 MP4 / MOV / AVI，单个文件不超过 {MAX_UPLOAD_MB} MB。手机端可直接从相册选择。")
uploaded = st.file_uploader(
    "上传投篮视频",
    type=("mp4", "mov", "avi"),
    help="优先使用 MP4/H.264；连续投篮时，每次动作间隔建议大于 0.8 秒。",
    label_visibility="collapsed",
)
st.markdown(
    '<div class="shot-privacy">🔒 当前版本不会把视频用于训练模型。分析会在服务端生成临时结果文件；请不要上传含敏感身份信息的视频。</div>',
    unsafe_allow_html=True,
)
if uploaded is not None:
    upload_error = validate_upload(uploaded.name, uploaded.size)
    upload_identity = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_identity") != upload_identity:
        st.session_state["upload_identity"] = upload_identity
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("report_path", None)
    if upload_error:
        st.error(upload_error)
    else:
        st.video(uploaded)
        _show_video_info(uploaded)
    analysis_disabled = (
        upload_error is not None
        or not MODEL_PATH.exists()
        or bool(missing_interface)
    )
    if st.button("开始 AI 动作分析", type="primary", disabled=analysis_disabled, width="stretch"):
        try:
            _run_analysis(uploaded, shooting_hand)
        except AnalysisError as error:
            st.error(f"分析失败：{error}")
        except Exception as error:
            st.error("分析失败：程序遇到未预期错误。请保留下方技术信息用于调试。")
            with st.expander("查看技术错误信息"):
                st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)))

if "analysis_result" in st.session_state:
    _render_result(st.session_state["analysis_result"], Path(st.session_state["report_path"]))
