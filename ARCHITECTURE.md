# ShotLab AI Architecture V3

## 三个确定性边界

```text
本地视频
  → Deterministic Pipeline（每球新建 CPU PoseLandmarker）
  → Stable Issue Engine（rules.v1.json）
  → Training Session Coaching Loop（IndexedDB）
```

`pipeline/video-analyzer.mjs` 固定 100ms 采样、最长 30 秒、VIDEO 模式、0.5 三项置信阈值、每球关闭模型，并锁定模型和运行文件 SHA-256。关键点按 visibility 0.55、越界/孤立跳变、最多 2 点插值、5 点二阶 Savitzky–Golay 的顺序处理。原始点计算可信度，滤波点计算角度与阶段。

阶段识别只输出 `loadingStart`、`lowestPoint`、`release`、`followThrough` 的采样序号和时间；近似并列候选固定取最早者，证据不足返回缺失，不由 AI 补选。

## Issue Engine

浏览器机器规则位于 `netlify-site/knowledge/rules.v1.json`，人类说明位于根目录 `knowledge/`。排序固定为：

```text
severityWeight × confidenceWeight × coachingPriority
```

同分按 `knowledgeOrder` 和 `issueCode`。低可信度动作问题不进 Top 2，只有 `CAPTURE_QUALITY_LOW` 可以在低可信度下成为首要问题。当前不接入 LLM。

## Training Session

`TrainingSessionV1` 锁定投篮手与距离类别；`ShotV1` 保存数据、问题代码、比较、cue 和 320px JPEG 缩略图。IndexedDB 最多保存 30 个 Session，原视频只保留当前页面 Blob URL。

第二球按角度 2°、节奏 0.05 秒、手腕上身参考宽度 5% 容差比较。低可信度、配置不同或人物尺度变化超过 25% 时返回 `not_comparable`。进入参考区间后 cue 转为保持，越过区间则标记 `overcorrected`。第 5 个中高可信度样本建立中位数/IQR Baseline。

## 隐私与部署

Netlify 只托管静态文件。CSP 将 `connect-src` 限制为同源，MediaPipe 的外部日志请求也会被浏览器阻止。视频、关键点、训练数据均不上传项目服务器。
