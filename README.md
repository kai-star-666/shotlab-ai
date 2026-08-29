# ShotLab AI · Basketball Shot Analyzer V0.4 Web Beta

面向电脑与手机浏览器的 AI 投篮动作分析应用：上传单次或连续投篮视频，自动输出二维动作数据、动作时序、Pose Overlay、纯火柴人重建、实际值/参考范围、证据化问题、动作建议与专项训练。

## V0.4 Web 界面

- ShotLab AI 深色篮球科技视觉，桌面端和手机端自适应；
- 手机浏览器可直接从相册选择视频，适合通过微信内置浏览器访问；
- 上传前校验格式、空文件和 200 MB 大小上限；
- 清楚展示三步使用流程、拍摄要求、隐私提示和二维分析边界；
- 保留 V0.3 的完整分析、视频、图表、报告与下载能力。

## V0.3 核心功能

- 连续投篮自动切分：基于屈膝蓄力、手腕上移和球手分离事件，导出 `shot_01.mp4` 等逐球视频。
- 拍摄质量检测：检查亮度、模糊、全身入镜、人物大小、侧面机位和分辨率，给出具体修正。
- 出手与翻腕分析：使用 Hand Landmarker 估计手腕角与手指方向；低置信度或不合理变化自动隐藏。
- 篮球轨迹与出手点：候选轨迹必须经过“靠近投篮手→向上分离”校验；失败时宁可不显示，不画假轨迹。
- 多球一致性：比较出手肘角、最低点膝角、翻腕变化等指标的标准差。
- 完整人体指标：膝/髋/肘/躯干、双肩高度差、投篮肩抬高、肘部二维横向对齐和手腕相对高度。
- 8 阶段时间轴：Ready、Dip、Lowest、Upward Drive、Set Point、Release Candidate、Follow Through，视频足够长时增加 Landing。
- 火柴人视频：纯深色背景重建人体骨架，实时显示阶段、投篮侧关节标记和四类角度。
- 证据化诊断：每个问题包含数据依据和限定解释，再对应 2–4 个可执行训练；数据不足时不凑数。
- 可测量/可估计/不可靠判断分类，防止把二维手臂角度误写成真实篮球出射角。

## 安装与运行

```powershell
cd "C:\Users\39576\OneDrive\文档\ChatGPT\投篮机"
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## 云端部署

这个项目不是纯静态网页：OpenCV、MediaPipe 和视频编码需要持续运行的 Python 进程，因此不能把 `app.py` 直接作为 Netlify 静态站点运行。推荐先将完整应用部署到 Streamlit Community Cloud（或支持 Docker/Python 的服务器），再按需用 Netlify 承载品牌主页和自定义域名入口。

Streamlit Community Cloud 部署时填写：

```text
Repository: kai-star-666/shotlab-ai
Branch: main
Main file path: app.py
```

仓库内的 `requirements.txt`、`packages.txt` 与 `.streamlit/config.toml` 已包含云端依赖、系统包、主题和上传限制配置。

修改了 `pose_analyzer.py` 等已导入模块后，如果旧 Streamlit 进程仍在运行，请先在终端按 `Ctrl+C` 完整停止，再执行上述唯一启动命令。

## 每次分析的主要输出

```text
outputs/analysis_YYYYMMDD_HHMMSS_xxxxxx/
├─ original_video.mp4
├─ pose_overlay_video.mp4
├─ stick_figure_video.mp4
├─ trajectory_video.mp4
├─ pose_data.csv
├─ shot_summary.csv
├─ shot_01.mp4
├─ keyframe_ready.jpg
├─ keyframe_lowest.jpg
├─ keyframe_set_point.jpg
├─ keyframe_release_candidate.jpg
├─ keyframe_followthrough.jpg
└─ report.html
```

只有篮球轨迹通过物理连续性校验时才生成 `ball_trajectory.csv`。

项目根目录必须有以下模型（后两个缺失时会自动降级为纯姿态分析）：

```text
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite
```

## 推荐视频

- 一次完整投篮或连续投篮，每次动作间隔建议大于 0.8 秒；
- 只有一名主要人物，全身（尤其投篮侧肩、肘、腕、髋、膝、踝）入镜；
- 相机固定，放在投篮手一侧约 90°，人物高度约占画面 45%–70%；
- 优先 1080p/60fps；720p/30fps 可用于大关节趋势，但手部和篮球轨迹更容易降级；
- 第一次优先使用 MP4/H.264，MOV/AVI 是否能读取取决于本机 OpenCV 编解码支持。

## 科学性边界

结果来自单摄像头二维估计，出手点是归一化画面坐标，不是真实球场空间坐标。出手帧应解读为 `Release Candidate`；除非可信篮球轨迹存在，否则不会输出篮球二维初始飞行方向。系统不是医疗诊断或职业级动作捕捉；“数据不足”比伪精确结论更可取。

## 学习边界

Codex 可以承担工程骨架、容错、文件输出和测试。你至少要自己理解：三点夹角的向量意义、MediaPipe 关键点置信度、二维投影误差、阶段规则为何只是近似，以及阈值必须通过多条自己的视频校准。它们也是这个项目对 RoboMaster/机器人方向最可迁移的部分：传感数据质量、时序特征、规则系统与验证思维。
