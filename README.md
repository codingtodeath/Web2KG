# web2kg

web2kg 用网站图谱帮助 Web Agent 找页面、走已有路径，再操作现场控件。图谱保存真实页面状态和操作证据；结构相近的页面归入模板，由功能卡说明用途，再由网站框架概括整体功能。

目前主要在 WebArena-Verified Reddit 上开发。执行沿用 WebNavigator 的 Agent 和动作接口，并增加检索、导航、观察适配、后台描述及任务结束后的图谱维护。

## 从哪里开始

工作目录是 `D:/Agent/MyCode/web2kg`，保留 `main`，当前在 `dev` 分支开发。项目 Python 是 `.venv-research/Scripts/python.exe`，不要直接使用系统 Anaconda base 运行实验。

双击 `start_console.cmd` 可打开总控。也可以在项目目录执行：

```powershell
$py = 'D:/Agent/MyCode/web2kg/.venv-research/Scripts/python.exe'
& $py scripts/start_control.py
```

总控默认端口为 8787，人工核验端口为 8788。打开总控不会自动启动付费实验；人工核验需要另行启动浏览器和匹配的本地 Jina 服务。

首次运行请准备自己的图目录，或在本机配置 `runs/active-graph.json`。[资产说明](docs/assets.md)

## 当前流程

1. **建图。** 浏览器有界探索，Jev 可负责候选排名和首次探索判断；页面描述在后台生成。最近建图使用 Jev 与 Qwen3.7-flash。
2. **建立知识层。** 完成探索后，用平均链接聚类生成模板，再生成所有功能卡、网站框架和本地 Jina 截图索引。
3. **执行任务。** 检索当前节点或入口可达的页面，由选择器挑选目标，再回放允许的路径。到达现场后，Actor 继续操作。
4. **维护资产。** 描述与 Actor 可并行，但未解释的新执行边不能回放。任务结束后才裁剪图谱、更新模板、补索引和更新卡片。

模板小更新和全局重整是两种不同操作，触发条件见[模板规则](templates.md)。模型名称、调用预算和重试策略由具体运行配置决定；通用脚本的旧默认值尚未全部统一为 Flash。

## 实验结果

Web2KG 使用 225 节点图谱和 Qwen3.7-flash，在启用根因定位的两批实验中共完成 106 道单站点 Reddit 任务，75 道获得官方成功评分。 [查看 Hard、非 Hard 分项及逐题成绩](docs/results.md)。

## 文档入口

| 文档 | 适合解决的问题 |
|---|---|
| [使用说明](docs/usage.md) | 怎样查看、运行和检查；哪些入口会调用模型 |
| [项目架构](docs/architecture.md) | 各阶段如何衔接，哪些步骤可以并行 |
| [模板规则](templates.md) | 怎样聚类、制卡、更新和清退节点 |
| [运行环境](docs/environment.md) | Python、Jina、站点端口、代理与私密配置 |
| [资产说明](docs/assets.md) | 应该使用哪份图，旧结果在哪里 |
| [实验结果](docs/results.md) | 已验证的结果、失败版本和结论边界 |
| [协作约定](AGENTS.md) | 修改代码和运行实验时需要遵守的约定 |

## 代码目录

`web2kg/common/` 管理浏览器、状态、存储和模型通信；`stage1/` 负责建图与知识层；`stage2/` 负责检索、导航和执行；`evaluation/` 接入评测；`visualization/` 提供查看界面。`scripts/` 包含运行及维护入口，`runs/` 保存独立实验，`auth/` 保存不提交的私密配置。

运行相关测试前设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。优先选择覆盖改动的测试；不把整套测试、GPU 编码或付费任务作为每次修改的默认检查。
