"""生成可离线直接打开的 HTML 分析报告。"""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pose_analyzer import AnalysisResult, PHASE_LABELS


LIMITATION = (
    "本系统基于单摄像头二维姿态估计。摄像机角度、遮挡和景深会影响角度；"
    "手腕与篮球指标仅用于运动趋势，不能达到专业动作捕捉或医疗评估精度。"
    "出手候选与关键阶段由规则算法估计；系统不能可靠测量真实三维出射角、篮球旋转、手指施力或肌肉发力。"
)


def _format(value: float, unit: str = "°") -> str:
    return f"{value:.1f}{unit}" if math.isfinite(value) else "数据不足"


def _chart_data_uri(result: AnalysisResult) -> str:
    names = {
        "elbow_angle": "Elbow angle",
        "knee_angle": "Knee angle",
        "hip_angle": "Hip angle",
        "trunk_angle": "Trunk tilt",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, names.items()):
        axis.plot(result.data["time"], result.data[column], color="#2563eb", linewidth=1.6)
        for phase, frame in result.phases.as_dict().items():
            axis.axvline(frame / result.info.fps, color="#f59e0b", alpha=0.45, linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Degrees")
        axis.grid(alpha=0.22)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140, facecolor="white")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_data_uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def generate_report(result: AnalysisResult) -> Path:
    metrics = result.metrics
    cards = (
        ("出手附近肘角", _format(metrics["elbow_release"])),
        ("最低点膝角", _format(metrics["knee_lowest"])),
        ("最低点髋角", _format(metrics["hip_lowest"])),
        ("最大躯干倾角", _format(metrics["trunk_max_abs"])),
    )
    card_html = "".join(
        f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in cards
    )
    keyframes = "".join(
        f'<figure><img src="{_image_data_uri(path)}" alt="{escape(PHASE_LABELS[name])}">'
        f'<figcaption>{escape(PHASE_LABELS[name])}</figcaption></figure>'
        for name, path in result.keyframes.items()
    )
    summary = "".join(f"<li>{escape(item)}</li>" for item in result.summary)
    suggestions = "".join(f"<li>{escape(item)}</li>" for item in result.suggestions)
    quality_issues = "".join(f"<li>{escape(item)}</li>" for item in result.capture_quality.issues)
    quality_corrections = "".join(f"<li>{escape(item)}</li>" for item in result.capture_quality.corrections)
    model_notes = "".join(f"<li>{escape(item)}</li>" for item in result.model_notes)
    primary = result.shots[0]
    comparison_html = primary.angle_comparison.to_html(index=False, border=0, escape=True) if primary.angle_comparison is not None else "<p>数据不足</p>"
    timing_html = "".join(
        f"<tr><td>{escape(name)}</td><td>{_format(value, ' s')}</td></tr>"
        for name, value in (primary.timing or {}).items()
    )
    problems_html = "".join(
        f"<article class='finding'><h3>{index}. {escape(item.title)}</h3><p><strong>数据依据：</strong>{escape(item.evidence)}</p><p><strong>解释：</strong>{escape(item.explanation)}</p></article>"
        for index, item in enumerate(primary.problems or [], 1)
    ) or "<p>当前可靠数据中未发现足以输出的明显问题。</p>"
    actions_html = "".join(f"<li>{escape(item)}</li>" for item in (primary.action_suggestions or []))
    drills_html = "".join(
        f"<article class='finding'><h3>训练 {index}：{escape(item.name)}</h3><p><strong>重点：</strong>{escape(item.focus)}</p><p><strong>方法：</strong>{escape(item.method)}</p><p><strong>建议量：</strong>{escape(item.dosage)}</p></article>"
        for index, item in enumerate(primary.training_drills or [], 1)
    )
    science_html = "".join(
        f"<h3>{escape(category)}</h3><ul>{''.join(f'<li>{escape(item)}</li>' for item in items)}</ul>"
        for category, items in result.science_boundaries.items()
    )
    shot_sections = []
    for shot in result.shots:
        shot_summary = "".join(f"<li>{escape(item)}</li>" for item in shot.summary)
        shot_suggestions = "".join(f"<li>{escape(item)}</li>" for item in shot.suggestions)
        snap = _format(shot.metrics.get("wrist_snap_change", math.nan))
        release_x = shot.metrics.get("ball_release_x", math.nan)
        release_y = shot.metrics.get("ball_release_y", math.nan)
        point = f"({release_x:.2f}, {release_y:.2f})" if math.isfinite(release_x) and math.isfinite(release_y) else "数据不足"
        shot_sections.append(
            f'<article class="shot"><h3>第 {shot.shot_id} 球 · '
            f'{shot.window.release_frame / result.info.fps:.2f}s</h3>'
            f'<p>出手肘角：{_format(shot.metrics.get("elbow_release", math.nan))}　'
            f'最低点膝角：{_format(shot.metrics.get("knee_lowest", math.nan))}　'
            f'翻腕变化：{snap}　画面出手点：{point}</p>'
            f'<p><a href="{escape(shot.clip_path.name if shot.clip_path else "")}">打开该球自动切分视频</a></p>'
            f'<h4>动作观察</h4><ol>{shot_summary}</ol><h4>训练建议</h4><ul>{shot_suggestions}</ul></article>'
        )
    consistency = result.consistency
    consistency_html = (
        f'<p>出手肘角标准差：{_format(consistency.get("elbow_release_std", math.nan))}　'
        f'最低点膝角标准差：{_format(consistency.get("knee_lowest_std", math.nan))}　'
        f'翻腕变化标准差：{_format(consistency.get("wrist_snap_change_std", math.nan))}</p>'
        if len(result.shots) >= 2 else "<p>只识别到 1 次投篮，暂不输出跨次稳定性结论。</p>"
    )
    stats = result.stats.copy()
    for column in ("平均值", "最小值", "最大值", "标准差"):
        stats[column] = stats[column].map(lambda value: _format(float(value)))

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AI 投篮动作分析报告</title>
<style>
body{{margin:0;background:#f3f6fb;color:#172033;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1080px;margin:auto;padding:36px 22px 60px}}h1,h2{{line-height:1.25}}h1{{margin-bottom:4px}}
.muted{{color:#637083}}section{{background:white;border-radius:14px;padding:22px;margin-top:18px;box-shadow:0 6px 24px #1f3b6812}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:#eef4ff;padding:16px;border-radius:10px}}
.card span{{display:block;color:#637083}}.card strong{{font-size:25px;color:#1457c5}}.frames{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}
figure{{margin:0}}figure img{{width:100%;border-radius:9px}}figcaption{{font-weight:700;margin-top:6px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e5eaf2;text-align:left}}
.notice{{border-left:4px solid #f59e0b;background:#fff8e8}}@media(max-width:720px){{.cards,.frames{{grid-template-columns:1fr 1fr}}}}
.quality{{display:grid;grid-template-columns:180px 1fr;gap:18px}}.score{{font-size:42px;font-weight:800;color:#1457c5}}
.shot{{border-top:1px solid #e5eaf2;padding-top:10px;margin-top:14px}}a{{color:#1457c5}}
.finding{{border:1px solid #e5eaf2;border-radius:10px;padding:12px 16px;margin:10px 0}}.reference{{border-left:4px solid #2563eb;background:#eef4ff}}
</style></head><body><main>
<h1>AI Basketball Shot Analyzer V0.3</h1><div class="muted">二维动作数据、时序、火柴人重建与证据化训练建议</div>
<section><h2>视频与检测概况</h2><p>文件：{escape(result.info.filename)}　投篮手：{result.shooting_hand}<br>
时长：{result.info.duration:.2f} 秒　FPS：{result.info.fps:.2f}　分辨率：{result.info.width}×{result.info.height}<br>
有效分析帧：{result.valid_frames}/{len(result.data)}（{result.valid_frames / max(len(result.data), 1):.1%}）　自动切分：{len(result.shots)} 次</p>
<p><a href="{escape(result.original_video.name)}">打开原视频</a>　<a href="{escape(result.processed_video.name)}">打开 Pose Overlay</a>　<a href="{escape(result.stick_figure_video.name)}">打开火柴人视频</a>　<a href="{escape(result.trajectory_video.name)}">打开出手点/轨迹视频</a></p></section>
<section><h2>拍摄质量</h2><div class="quality"><div><div class="score">{result.capture_quality.score}/100</div><strong>{escape(result.capture_quality.level)}</strong></div><div><h3>检测问题</h3><ul>{quality_issues}</ul><h3>机位与拍摄修正</h3><ul>{quality_corrections}</ul></div></div></section>
<section><h2>模型可信度与降级</h2><ul>{model_notes or '<li>手部与篮球轨迹通过当前规则校验。</li>'}</ul></section>
<section><h2>核心指标</h2><div class="cards">{card_html}</div></section>
<section><h2>实际投篮数据 vs 参考范围</h2>{comparison_html}<div class="reference"><p>参考范围只用于动作观察，不代表所有球员必须达到完全相同的数值。研究显示投篮距离和熟练度会改变关节运动学。</p><p><a href="https://pubmed.ncbi.nlm.nih.gov/38314460/">距离变化研究</a> · <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8822900/">熟练度与投篮运动学研究</a></p></div></section>
<section><h2>关键帧</h2><div class="frames">{keyframes}</div></section>
<section><h2>角度—时间曲线</h2><img style="width:100%" src="{_chart_data_uri(result)}" alt="角度曲线"></section>
<section><h2>投篮时序</h2><table><thead><tr><th>时序指标</th><th>我的数据</th></tr></thead><tbody>{timing_html}</tbody></table><p class="muted">上下肢时序是关节运动规则估计，不是肌肉发力测量。</p></section>
<section><h2>关键阶段统计</h2>{stats.to_html(index=False, border=0, escape=True)}</section>
<section><h2>动作总结</h2><ol>{summary}</ol><h2>训练建议</h2><ul>{suggestions}</ul></section>
<section><h2>当前主要投篮问题</h2>{problems_html}</section>
<section><h2>动作调整建议</h2><ol>{actions_html}</ol></section>
<section><h2>专项训练建议</h2>{drills_html}</section>
<section><h2>手型观察</h2><p>{escape(primary.hand_analysis)}</p></section>
<section><h2>连续投篮一致性</h2>{consistency_html}</section>
<section><h2>逐球结果</h2>{''.join(shot_sections)}</section>
<section><h2>可测量 / 可估计 / 不可靠判断</h2>{science_html}</section>
<section class="notice"><h2>技术限制</h2><p>{escape(LIMITATION)}</p></section>
</main></body></html>"""
    report_path = result.output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
