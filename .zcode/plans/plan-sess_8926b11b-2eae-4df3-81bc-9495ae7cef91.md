# 目录重构：按数据格式版本拆分 v2.1（旧）与 v3.0（新）

## 拆分依据（已核实的导入关系）
- **v2.1 旧路径**完全自包含：`config.py`、`types.py`、`pipeline.py`、`parallel.py`、`report.py`、`wizard.py`、`video_utils.py` + `dataset/{reader,inspector,validate,stats,writer}.py` + `rules/`（内部互相引用，不引用新代码）。
- **v3.0 新路径**只互相引用：`dataset/{v3,v3_streaming,v3_stream_stats}.py`、`dataset/{episode_review,review_profile,review_report,video_review,libero}.py`、`lingbot/`、`training_readiness.py`。
- `cli.py` 是唯一同时引用两组的调度层（v3 相关 import 已是惰性的）。

## 目标结构
```
lerobot_cleaner/
├── cli.py                      # 保留，命令分组：v2.1 组 / v3.0 组
├── v21/                        # 旧 GR00T v2.1 清洗路径
│   ├── config.py  types.py  pipeline.py  parallel.py
│   ├── report.py  wizard.py  video_utils.py
│   ├── reader.py  inspector.py  validate.py  stats.py  writer.py
│   └── rules/                  # r1~r7 七条规则
└── v30/                        # 新 LeRobot v3.0 + LingBot-VLA 路径
    ├── v3.py  v3_streaming.py  v3_stream_stats.py
    ├── episode_review.py  review_profile.py  review_report.py
    ├── video_review.py  libero.py
    ├── training_readiness.py
    └── lingbot/                # v2.1→v3.0 导出 + LingBot 规范化
        ├── config.py  generate.py  vla.py  __init__.py
```
文件名保持不变（如 `v3.py` 不改名），只移动位置，`git mv` 保留历史。原 `dataset/`、`rules/` 及顶层旧模块移空后删除；`dataset/__init__.py` 目前是空文件，无对外导出，安全。`lingbot/` 归入 v30 侧，因为它是面向 LingBot-VLA（v3.0）的导出/规范化层。不加兼容 shim，全部引用一次性改净。

## 实施步骤
1. **git mv 移动文件**：按上表把 20 个模块移入 `v21/`、`v30/`、`v30/lingbot/`。
2. **批量更新 import**：
   - v21 组内部：`lerobot_cleaner.config` → `lerobot_cleaner.v21.config`，`lerobot_cleaner.dataset.reader` → `lerobot_cleaner.v21.reader` 等（涉及 pipeline、parallel、types、wizard、rules/*、writer、inspector、validate、lingbot/vla 无需改）。
   - v30 组内部：`lerobot_cleaner.dataset.v3*` / `review_*` → `lerobot_cleaner.v30.*`（涉及 v3_streaming、episode_review、review_profile、review_report）。
   - `cli.py`：12 处顶层 import 改为 `v21.*`，惰性 import 的 `dataset.v3` / `dataset.v3_streaming` 改为 `v30.*`。
   - `lerobot_cleaner/__init__.py`：`from lerobot_cleaner.config import ...` → `v21.config`。
3. **更新测试**：`tests/` 拆成 `tests/v21/`（test_config、test_rules、test_stats、test_validate、test_e2e）和 `tests/v30/`（test_v3、test_v3_streaming、test_libero、test_review_workflow、test_lingbot_*、test_lingbot_vla），更新所有 import；`conftest.py` 若两组夹具都有则留在顶层并改 import。
4. **更新 scripts/**：`run_droid_clean.py`、`run_libero_clean.py`、`run_lingbot_norm.py`、`run_lingbot_probe.py`、`package_streaming_update.py` 中的 import 路径。
5. **更新文档**：README.md、PROJECT_STATUS.md、docs/*.md 中的目录树和文件路径描述（如 `lerobot_cleaner/dataset/v3.py` → `lerobot_cleaner/v30/v3.py`），并加一段说明新目录划分。
6. **验证**：`python -m pytest` 全量跑通；`lerobot_cleaner --help` 与各命令入口能正常加载；确认无残留 `lerobot_cleaner.dataset` / 顶层 `lerobot_cleaner.config` 引用（grep 检查）。

## 不做的事
- 不改任何逻辑代码，只动位置和 import。
- 不重命名模块文件，不保留旧路径兼容别名。
- 不动 `configs/`、`docs/` 的文件组织（只更新其中提到的路径字符串）。