# Shot Analyzer V0.2 Implementation Plan

> **For agentic workers:** Implement inline in the current session. Do not commit without user authorization.

**Goal:** Add continuous-shot segmentation, capture-quality guidance, ball-assisted release detection, trajectory video, experimental hand/wrist metrics, and multi-shot training advice.

**Architecture:** Keep pose processing as the coordinator. Add one pure time-series module and one optional model wrapper so multi-shot logic is unit-testable and detector failures degrade safely.

**Tech Stack:** Python 3.13, MediaPipe Tasks, OpenCV, NumPy, Pandas, Streamlit, Plotly, Matplotlib.

---

### Task 1: Pure video intelligence

- [ ] Write failing tests for quality scoring, release candidates, candidate deduplication and shot windows.
- [ ] Implement `video_intelligence.py` and make tests pass.

### Task 2: Hand and basketball model wrapper

- [ ] Write failing tests for hand-to-pose matching and wrist-angle calculations.
- [ ] Implement `vision_models.py`, model loading and frame result normalization.

### Task 3: Pipeline integration

- [ ] Extend frame schema with ball, hand, release-source and shot-id fields.
- [ ] Produce ball-trail overlay, release markers and `shot_NN.mp4` clips.
- [ ] Keep pose-only fallback operational.

### Task 4: Product surfaces

- [ ] Add quality/position guidance and multi-shot controls to Streamlit.
- [ ] Add multi-shot comparison, hand metrics and trajectory assets to HTML report.

### Task 5: Verification

- [ ] Run all unit tests and model load probes.
- [ ] Re-run the user's original single shot.
- [ ] Create a three-shot concatenated fixture and verify segmentation, clips, trajectory video and report.
