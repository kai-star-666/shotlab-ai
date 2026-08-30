# ShotLab AI Architecture V3.2

## 三个确定性边界

```text
本地视频
  → Deterministic Pipeline（每球新建 CPU PoseLandmarker）
  → Stable Issue Engine（rules.v1.json）
  → Training Session Coaching Loop（IndexedDB）
```

`pipeline/video-analyzer.mjs` 固定 100ms 采样、最长 30 秒、VIDEO 模式、0.5 三项置信阈值、每球关闭模型，并锁定模型和运行文件 SHA-256。每球创建两套相互隔离的 CPU PoseLandmarker，对原图和水平镜像图使用相同视频时间戳推理；`orientation-normalizer.mjs` 将镜像结果还原到原坐标和左右身体语义，再按整段肩肘腕/髋膝踝完整度确定唯一方向，禁止逐帧来回切换。关键点随后按 visibility 0.55、越界/孤立跳变、最多 2 点插值、5 点二阶 Savitzky–Golay 的顺序处理。

动作后处理会分别选择可见的手臂侧和下肢侧。缺少脚踝时仍可用髋膝与手腕趋势定位动作阶段，但膝角和相关建议降为低可信度。正面/背面拍摄仍输出阶段与节奏，矢状面的肘膝髋、前倾/后仰不进入 Top 2。

阶段识别只输出 `loadingStart`、`lowestPoint`、`release`、`followThrough` 的采样序号和时间；近似并列候选固定取最早者，证据不足返回缺失，不由 AI 补选。

## Issue Engine

浏览器机器规则位于 `netlify-site/knowledge/rules.v1.json`，人类说明位于根目录 `knowledge/`。排序固定为：

```text
severityWeight × confidenceWeight × coachingPriority
```

同分按 `knowledgeOrder` 和 `issueCode`。低可信度动作问题不进 Top 2，只有 `CAPTURE_QUALITY_LOW` 可以在低可信度下成为首要问题。当前不接入 LLM。

## Training Session

`TrainingSessionV1` 锁定投篮手与距离类别；`ShotV1` 保存标准化 `record`、分析结果、问题代码、cue、历史比较和 320px JPEG 缩略图。IndexedDB 最多保存 30 个 Session，原视频只保留当前页面 Blob URL。

每个新 Shot 生成四个确定性比较层：`previous`、`first`、`sessionAverage` 和 `best`。历史趋势只使用中/高可信度样本，统一输出 `IMPROVING / STABLE / WORSENING / INCONSISTENT / UNCERTAIN`。Best Shot 按可信度、拍摄质量、问题惩罚、优点数量和多个训练区间综合评分，同分固定选择较早的 Shot。

个体目标区间采用渐进策略：区间外每球只推进一个固定步长，进入区间后收缩为当前值附近的保持带；五球 Baseline 建立后优先使用知识区间与个人稳定区间的有效交集。比较越过目标区间时标记 `overcorrected`，上一球没有 Priority 时保留当前确定性 cue，不允许产生空建议。

角度按 2°、节奏按 0.05 秒、手腕上身参考宽度按 5% 容差比较。低可信度、配置不同或人物尺度变化超过 25% 时返回 `not_comparable`。第 5 个中高可信度样本建立中位数/IQR Baseline。

## 隐私与部署

Netlify 只托管静态文件。CSP 将 `connect-src` 限制为同源，MediaPipe 的外部日志请求也会被浏览器阻止。视频、关键点、训练数据均不上传项目服务器。
