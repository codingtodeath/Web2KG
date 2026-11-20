# 运行环境

[使用说明](usage.md) · [资产说明](assets.md)

## Python 与工作目录

在 `D:/Agent/MyCode/web2kg` 的 `dev` 分支开发，保留 `main` 分支。系统默认 Python 是 Anaconda base，但项目运行和测试应使用下表中的专用环境。

| 用途 | 路径 |
|---|---|
| 项目 Python | `D:/Agent/MyCode/web2kg/.venv-research/Scripts/python.exe` |
| Jina GPU Python | `D:/Anaconda/envs/web2kg-jina-gpu/python.exe` |
| 已有 Jina 权重 | `D:/Agent/MyCode/browser-use/webNavigator/models/jina-embeddings-v4` |
| 执行依赖记录 | [webnavigator-execution-lock.txt](../environments/webnavigator-execution-lock.txt) |
| Jina 依赖记录 | [jina-gpu-lock.txt](../environments/jina-gpu-lock.txt) |

环境已经安装，不需要为普通开发重新下载模型或升级系统 base。新机器安装可参考 [setup_jina_gpu.ps1](../scripts/setup_jina_gpu.ps1)；安装流程不等于可以直接替换旧索引所用的编码环境。

## 模型与代理

近期建图由 Jev 排名、判断首次探索及可选局部状态等价，由 Qwen3.7-flash 生成描述、卡片和框架。近期 Hard 执行实验的各 LLM 角色均使用 Qwen3.7-flash。

这套配置由实验计划显式指定。通用 `run_template_agent.py` 仍有 Terra/Luna/Sol 默认值，不能根据文件名或只修改 Actor 的 `--model` 就认为所有角色已经换成 Flash。

- Qwen 近期实验使用百炼直连，配置来源包括私密的 `auth/qwen-hard.env`。
- Jev 读取项目 `.env` 中既有 `api_key`，也支持显式 `JEV_API_KEY` 或 `TYPESAFE_API_KEY`。Jev 不是 Jina。
- Jev 可以通过 `ranking_proxy` 使用现有本地 HTTP 代理；最近实验使用 `127.0.0.1:7890`。不要擅自关闭代理或改变其他进程的网络设置。
- `auth/llm-crawl.env`、登录态及上述配置都属于私密文件，不提交、打印或放进报告。

## 本地 Jina

总控常用 `http://127.0.0.1:8793/v1/embeddings`，实验也会使用自己的端口。端口号不代表服务一直在运行，启动前应检查已有进程。

Jina 的模型、精度、预处理和编码脚本身份必须与输入索引一致。近期索引使用视觉 FP32 流程；不能把另一精度、模型或处理流程的向量混进去。完整要求以输入索引的 encoder identity 为准。

部分历史索引绑定了特定编码脚本的字节。实验登记 `encoder_runtime` 时，应从该目录启动服务；即使只是换行符转换，也可能改变源码指纹。不要修改指纹或向量来绕过核对。

复用身份匹配的服务，或用原脚本和本地权重启动本次独占服务。显存不足时保留等待和错误证据，不关闭用户的其他 GPU 应用，也不并行加载多份模型碰运气。

## 网站与进程所有权

| 服务 | 端口 | 操作范围 |
|---|---|---|
| 既有 Reddit 服务 | 9999 | 保留 |
| 开发 Reddit | 10499/10498 | 保留，不作为正式评测重置目标 |
| 自有评测副本 | 10599/10598 | 仅在名称和所有者标签都匹配时重置 |
| 独立检查副本 | 10699/10698 | 必须匹配 `web2kg.execution_check` 标签 |
| 总控 / 人工核验 | 默认 8787/8788 | 冲突时改端口，不关闭已有窗口 |

评测容器名称是 `web2kg-template-eval`，要求标签 `web2kg.template_eval=template-agent-hard10-v1`。其他服务，包括既有 GitLab 和用户浏览器，不因本次实验而重置。

登录态通常保存在 `auth/`。不同副本的 URL、端口和认证要匹配，不能把保存登录态当作站点数据已经重置。

## 本地检查

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& ./.venv-research/Scripts/python.exe -X utf8 -B -m pytest tests/test_numeric_observation.py -q
```

上例是观察适配的相关测试，不是所有修改都必须跑它。选择覆盖本次改动的测试；GPU 编码、容器重置和付费调用不属于文档修改的验证步骤。

## 基准站点登录

通过进程环境提供所用站点的密码，例如 Reddit 使用 `VERIFIED_REDDIT_PASSWORD`，GitLab 使用 `VERIFIED_GITLAB_PASSWORD`。两个第三方适配目录通过 `provenance.json` 的 `distribution_overrides` 登记发布版配置的哈希。

购物站点的登录名通过 `VERIFIED_SHOPPING_USER` 设置。WebNavigator 屏蔽规则默认匹配本机地址，其他部署请将 `WEBNAVIGATOR_BLOCK_HOST` 设置为站点主机名或 IP（不含端口）。
