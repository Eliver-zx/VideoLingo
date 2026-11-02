# VideoLingo

本地优先的多阶段 AI 视频转录、术语对齐翻译、字幕压制与配音工作台。

## Quick Start & Verify

- **Quick Launch**: Double-click `Start.command` (macOS) or `Start.bat` (Windows), or run `python3 api.py` (`http://127.0.0.1:8501`)
- **Run Tests** (conda base `python3`, no local `.venv`): `pytest tests/`

## Language

### 工作区与执行 (Workspace & Execution)

**Task Workspace (任务工作区)**:
位于 `output/<task_id>/` 下的隔离目录，持有单个媒体任务的源媒体引用、独立配置快照、阶段产物与运行状态清单（`manifest.json`）。
_Avoid_: Project folder, job directory, output folder

**Waterfall Stage (瀑布流阶段)**:
任务详情视图中按 01–06 编号组织的六个高阶用户可见阶段（媒体挂载、词级识别与断句、术语提取与审校、翻译与配音词审校、字幕切分与烧录、TTS 配音合成），聚合了底层十二步管线脚本。
_Avoid_: Pipeline step, tab, phase

**Review Checkpoint (审校检查点)**:
流水线在术语提取完成后（译前）或翻译完成后（译后）主动进入暂停态（`paused`），等待人工在工作台修订数据后再继续下游执行的停靠点。
_Avoid_: Pause hook, manual breakpoint, stop state

### 术语与字幕资产 (Glossary & Subtitles)

**Glossary Domain (术语领域)**:
位于 `glossary/<domain>/` 下的垂直领域词库命名空间（如 `destiny2`），包含可编辑领域词表（`terms.csv`）与可选的权威大词典（`dict.json`）。
_Avoid_: Category, preset, dictionary folder

**Term Promotion (术语晋升)**:
将单任务提取或审校后的术语（`terms.json`）勾选归档并合并写入当前绑定的术语领域词表（`glossary/<domain>/terms.csv`）的动作。
_Avoid_: Export terms, save to global, sync glossary

**Utterance (语义句段)**:
经词级时间戳与语义断句生成的单条时间轴对照单元，同时承载原文字段（`src`）、画面字幕译文（`tgt`）与配音台词（`tts_text`）。
_Avoid_: Subtitle line, cue, sentence row
