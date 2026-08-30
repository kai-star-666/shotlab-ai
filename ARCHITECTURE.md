# Architecture

## 运行链路

```text
本地视频
  → 浏览器抽样
  → MediaPipe Pose Landmarker
  → 时间平滑与主投篮窗口
  → 角度/轨迹/节奏特征
  → 可信度与启发式教练规则
  → AnalysisResult 2.1
  → 所有页面区块与双回放
```

## AnalysisResult 2.1

核心字段：`summary`、`strengths`、`priorities`、`nextRep`、`metrics`、`jointAnalysis`、`rhythm`、`keyframes`、`processedVideo`、`skeletonVideo`、`charts`、`trainingPlan`、`confidence`、`personalBaseline`、`technicalLimitations`。

旧字段暂时保留供现有 UI 和测试兼容。页面必须只消费当前分析返回对象，不读取静态示例数据。

## 三层建议系统

1. 计算机视觉层：关键点、visibility、有效帧比例、人物画面占比、二维角度和时间序列。
2. 规则层：相对幅度、趋势、连续性、置信度降级、问题排序和训练映射。
3. 教练表达层：当前实测、训练参考、优点、问题、影响、下一球和长期处方。

研究显示投篮动作会随距离、技能和任务约束发生适应，所以参考区间不能被描述成统一标准。依据：[distance variation study](https://pubmed.ncbi.nlm.nih.gov/38314460/)、[distance and energy flow](https://pubmed.ncbi.nlm.nih.gov/30001184/)、[youth systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8005190/)。

## 部署与缓存

Netlify 只托管 `netlify-site/` 静态文件。媒体和推理不离开用户设备。脚本和样式使用版本参数，同时服务器要求每次重新验证静态资源，避免 HTML 与旧 JavaScript 缓存错配。

## 已知边界

- 浏览器不会可靠暴露视频源总帧数，因此该字段未知时显示“浏览器未提供”，不拿采样点冒充源帧；
- 处理后视频是页面内实时叠加回放，不上传或导出新 MP4；
- 单球不计算“多球稳定性”；
- 真正的微信小程序发布需要主体、备案域名和微信平台配置。
