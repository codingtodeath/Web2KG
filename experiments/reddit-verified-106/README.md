# Reddit 全量对照实验 · Qwen3.7-flash

这里收录 WebArena-Verified 单站 Reddit 全部 106 道题。每道题可以对照阅读 Web2KG 和 WebNavigator 的动作轨迹。Hard 部分沿用归档选定的 42 题（含 724 的原始失败），另 64 题来自后续完成的 non-Hard 实验。两部分跨代码版本，结果适合描述这批任务的表现。

| 方法 | Hard | non-Hard | 全部 | Actor 轨迹记录数 |
| --- | ---: | ---: | ---: | ---: |
| Web2KG（225 节点起始图） | 32/42 | 43/64 | 75/106（70.8%） | 774 |
| WebNavigator | 31/42 | 46/64 | 77/106（72.6%） | 908 |

### 怎么读轨迹

下表按任务编号排列。打开 JSON，在 `intent` 看任务要求，在 `trajectory` 按顺序看模型记录的观察摘要、理由和动作。动作引用的控件编号能在该轮观察中匹配时，`matching_observation_lines` 会给出原文片段。记录中的最终回答属于模型输出，任务结果以 `official_score` 为准。Web2KG 的 615、618 和 720 三题保留了原始空轨迹，错误状态见各自的 `run_status`。

这是逐步动作的公开导出。网址中的本机及部署地址已替换为占位地址，敏感参数和身份信息已脱敏。导出范围不含原始截图、完整 AX 树、模型请求全文、登录态、网络抓包或评分器参考答案。

### 实验用图

- [Web2KG 建图结果](graphs/web2kg/graph.json)保存 225 个节点、263 条有向边、页面描述、控件、动作及已有的回放判定。[建图时的模板、卡片与框架](graphs/web2kg/construction.json)和[后续五页取样的卡片与框架](graphs/web2kg/five-page.json)分别保留。两份源图的节点与边文件哈希相同。这里展示建好的基础图；执行中每题交接的增量图未逐题发布。
- [WebNavigator 基线图](graphs/webnavigator/graph.json)来自对照实际加载的发布图，含 225 个节点、224 条有向边和 124 组动作记录。边上的动作编号可在 `actions` 查找，每个节点还保留 URL、直达标志和控件信息。
- 两图均为经过脱敏的结构化 JSON，适合查看与分析。原始截图、截图向量和运行时缓存未包含在导出中，不能直接用这份归档复现实验的视觉检索。

[机器可读任务索引](tasks.json) · [来源校验与分组统计](provenance.json)

### 逐题查看

分数为官方得分，括号内为轨迹文件的记录数。

| 题号 | 难度 | Web2KG | WebNavigator |
| ---: | --- | --- | --- |
| 27 | non-hard | [1（5）](traces/web2kg/27.json) | [1（6）](traces/webnavigator/27.json) |
| 28 | hard | [1（7）](traces/web2kg/28.json) | [1（6）](traces/webnavigator/28.json) |
| 29 | hard | [1（6）](traces/web2kg/29.json) | [1（5）](traces/webnavigator/29.json) |
| 30 | non-hard | [1（5）](traces/web2kg/30.json) | [1（5）](traces/webnavigator/30.json) |
| 31 | hard | [1（5）](traces/web2kg/31.json) | [1（6）](traces/webnavigator/31.json) |
| 66 | hard | [0（11）](traces/web2kg/66.json) | [0（23）](traces/webnavigator/66.json) |
| 67 | hard | [0（15）](traces/web2kg/67.json) | [0（27）](traces/webnavigator/67.json) |
| 68 | hard | [0（8）](traces/web2kg/68.json) | [0（26）](traces/webnavigator/68.json) |
| 69 | non-hard | [0（11）](traces/web2kg/69.json) | [0（22）](traces/webnavigator/69.json) |
| 399 | hard | [1（6）](traces/web2kg/399.json) | [1（5）](traces/webnavigator/399.json) |
| 400 | hard | [1（6）](traces/web2kg/400.json) | [1（5）](traces/webnavigator/400.json) |
| 401 | hard | [1（6）](traces/web2kg/401.json) | [1（7）](traces/webnavigator/401.json) |
| 402 | non-hard | [1（6）](traces/web2kg/402.json) | [1（5）](traces/webnavigator/402.json) |
| 403 | non-hard | [1（6）](traces/web2kg/403.json) | [1（7）](traces/webnavigator/403.json) |
| 404 | non-hard | [1（6）](traces/web2kg/404.json) | [1（8）](traces/webnavigator/404.json) |
| 405 | non-hard | [1（7）](traces/web2kg/405.json) | [1（6）](traces/webnavigator/405.json) |
| 406 | non-hard | [0（5）](traces/web2kg/406.json) | [1（10）](traces/webnavigator/406.json) |
| 407 | hard | [1（7）](traces/web2kg/407.json) | [1（6）](traces/webnavigator/407.json) |
| 408 | hard | [1（6）](traces/web2kg/408.json) | [1（5）](traces/webnavigator/408.json) |
| 409 | hard | [1（4）](traces/web2kg/409.json) | [1（4）](traces/webnavigator/409.json) |
| 410 | hard | [1（4）](traces/web2kg/410.json) | [1（5）](traces/webnavigator/410.json) |
| 580 | hard | [1（8）](traces/web2kg/580.json) | [1（7）](traces/webnavigator/580.json) |
| 581 | hard | [1（8）](traces/web2kg/581.json) | [1（8）](traces/webnavigator/581.json) |
| 582 | non-hard | [1（8）](traces/web2kg/582.json) | [1（14）](traces/webnavigator/582.json) |
| 583 | non-hard | [1（8）](traces/web2kg/583.json) | [1（11）](traces/webnavigator/583.json) |
| 584 | hard | [1（8）](traces/web2kg/584.json) | [1（11）](traces/webnavigator/584.json) |
| 595 | non-hard | [1（5）](traces/web2kg/595.json) | [1（4）](traces/webnavigator/595.json) |
| 596 | hard | [1（13）](traces/web2kg/596.json) | [0（13）](traces/webnavigator/596.json) |
| 597 | hard | [1（10）](traces/web2kg/597.json) | [1（8）](traces/webnavigator/597.json) |
| 598 | hard | [1（9）](traces/web2kg/598.json) | [0（17）](traces/webnavigator/598.json) |
| 599 | non-hard | [0（6）](traces/web2kg/599.json) | [0（6）](traces/webnavigator/599.json) |
| 600 | non-hard | [0（1）](traces/web2kg/600.json) | [1（6）](traces/webnavigator/600.json) |
| 601 | non-hard | [0（7）](traces/web2kg/601.json) | [1（6）](traces/webnavigator/601.json) |
| 602 | non-hard | [1（7）](traces/web2kg/602.json) | [1（5）](traces/webnavigator/602.json) |
| 603 | hard | [1（7）](traces/web2kg/603.json) | [1（5）](traces/webnavigator/603.json) |
| 604 | non-hard | [0（6）](traces/web2kg/604.json) | [0（6）](traces/webnavigator/604.json) |
| 605 | non-hard | [0（6）](traces/web2kg/605.json) | [0（8）](traces/webnavigator/605.json) |
| 606 | non-hard | [1（7）](traces/web2kg/606.json) | [1（5）](traces/webnavigator/606.json) |
| 607 | non-hard | [1（7）](traces/web2kg/607.json) | [1（8）](traces/webnavigator/607.json) |
| 608 | non-hard | [0（7）](traces/web2kg/608.json) | [0（6）](traces/webnavigator/608.json) |
| 609 | non-hard | [0（7）](traces/web2kg/609.json) | [0（8）](traces/webnavigator/609.json) |
| 610 | hard | [1（8）](traces/web2kg/610.json) | [1（8）](traces/webnavigator/610.json) |
| 611 | non-hard | [0（7）](traces/web2kg/611.json) | [0（8）](traces/webnavigator/611.json) |
| 612 | non-hard | [0（9）](traces/web2kg/612.json) | [0（8）](traces/webnavigator/612.json) |
| 613 | non-hard | [0（6）](traces/web2kg/613.json) | [0（10）](traces/webnavigator/613.json) |
| 614 | non-hard | [0（9）](traces/web2kg/614.json) | [0（8）](traces/webnavigator/614.json) |
| 615 | hard | [0（0）](traces/web2kg/615.json) | [0（8）](traces/webnavigator/615.json) |
| 616 | non-hard | [1（8）](traces/web2kg/616.json) | [1（7）](traces/webnavigator/616.json) |
| 617 | hard | [0（1）](traces/web2kg/617.json) | [0（10）](traces/webnavigator/617.json) |
| 618 | hard | [0（0）](traces/web2kg/618.json) | [1（8）](traces/webnavigator/618.json) |
| 619 | non-hard | [1（7）](traces/web2kg/619.json) | [1（8）](traces/webnavigator/619.json) |
| 620 | non-hard | [1（7）](traces/web2kg/620.json) | [1（6）](traces/webnavigator/620.json) |
| 621 | non-hard | [1（7）](traces/web2kg/621.json) | [1（6）](traces/webnavigator/621.json) |
| 622 | non-hard | [1（7）](traces/web2kg/622.json) | [1（6）](traces/webnavigator/622.json) |
| 623 | non-hard | [1（7）](traces/web2kg/623.json) | [1（6）](traces/webnavigator/623.json) |
| 624 | hard | [1（7）](traces/web2kg/624.json) | [1（9）](traces/webnavigator/624.json) |
| 625 | hard | [1（9）](traces/web2kg/625.json) | [0（8）](traces/webnavigator/625.json) |
| 626 | non-hard | [1（7）](traces/web2kg/626.json) | [1（7）](traces/webnavigator/626.json) |
| 627 | non-hard | [1（7）](traces/web2kg/627.json) | [1（7）](traces/webnavigator/627.json) |
| 628 | non-hard | [1（7）](traces/web2kg/628.json) | [1（7）](traces/webnavigator/628.json) |
| 629 | non-hard | [1（7）](traces/web2kg/629.json) | [1（6）](traces/webnavigator/629.json) |
| 630 | non-hard | [1（7）](traces/web2kg/630.json) | [1（6）](traces/webnavigator/630.json) |
| 631 | non-hard | [0（7）](traces/web2kg/631.json) | [1（7）](traces/webnavigator/631.json) |
| 632 | non-hard | [1（7）](traces/web2kg/632.json) | [1（8）](traces/webnavigator/632.json) |
| 633 | non-hard | [1（7）](traces/web2kg/633.json) | [1（9）](traces/webnavigator/633.json) |
| 634 | hard | [1（7）](traces/web2kg/634.json) | [1（6）](traces/webnavigator/634.json) |
| 635 | non-hard | [1（7）](traces/web2kg/635.json) | [1（6）](traces/webnavigator/635.json) |
| 636 | hard | [1（8）](traces/web2kg/636.json) | [1（10）](traces/webnavigator/636.json) |
| 637 | non-hard | [1（7）](traces/web2kg/637.json) | [1（6）](traces/webnavigator/637.json) |
| 638 | hard | [0（7）](traces/web2kg/638.json) | [1（7）](traces/webnavigator/638.json) |
| 639 | non-hard | [1（7）](traces/web2kg/639.json) | [1（6）](traces/webnavigator/639.json) |
| 640 | hard | [1（7）](traces/web2kg/640.json) | [1（7）](traces/webnavigator/640.json) |
| 641 | non-hard | [1（7）](traces/web2kg/641.json) | [1（7）](traces/webnavigator/641.json) |
| 642 | non-hard | [1（7）](traces/web2kg/642.json) | [1（7）](traces/webnavigator/642.json) |
| 643 | non-hard | [1（7）](traces/web2kg/643.json) | [1（10）](traces/webnavigator/643.json) |
| 644 | hard | [1（10）](traces/web2kg/644.json) | [1（9）](traces/webnavigator/644.json) |
| 645 | hard | [1（6）](traces/web2kg/645.json) | [1（6）](traces/webnavigator/645.json) |
| 646 | hard | [1（8）](traces/web2kg/646.json) | [1（5）](traces/webnavigator/646.json) |
| 647 | hard | [1（6）](traces/web2kg/647.json) | [1（7）](traces/webnavigator/647.json) |
| 648 | non-hard | [1（6）](traces/web2kg/648.json) | [1（8）](traces/webnavigator/648.json) |
| 649 | non-hard | [1（6）](traces/web2kg/649.json) | [1（5）](traces/webnavigator/649.json) |
| 650 | non-hard | [0（3）](traces/web2kg/650.json) | [1（3）](traces/webnavigator/650.json) |
| 651 | hard | [1（4）](traces/web2kg/651.json) | [1（3）](traces/webnavigator/651.json) |
| 652 | non-hard | [1（3）](traces/web2kg/652.json) | [1（3）](traces/webnavigator/652.json) |
| 714 | non-hard | [1（5）](traces/web2kg/714.json) | [1（9）](traces/webnavigator/714.json) |
| 715 | hard | [0（5）](traces/web2kg/715.json) | [0（10）](traces/webnavigator/715.json) |
| 716 | non-hard | [1（10）](traces/web2kg/716.json) | [1（11）](traces/webnavigator/716.json) |
| 717 | non-hard | [1（11）](traces/web2kg/717.json) | [1（13）](traces/webnavigator/717.json) |
| 718 | non-hard | [1（12）](traces/web2kg/718.json) | [1（11）](traces/webnavigator/718.json) |
| 719 | non-hard | [0（3）](traces/web2kg/719.json) | [0（2）](traces/webnavigator/719.json) |
| 720 | non-hard | [0（0）](traces/web2kg/720.json) | [1（10）](traces/webnavigator/720.json) |
| 721 | hard | [1（26）](traces/web2kg/721.json) | [0（5）](traces/webnavigator/721.json) |
| 722 | non-hard | [0（7）](traces/web2kg/722.json) | [0（21）](traces/webnavigator/722.json) |
| 723 | non-hard | [0（6）](traces/web2kg/723.json) | [0（10）](traces/webnavigator/723.json) |
| 724 | hard | [0（4）](traces/web2kg/724.json) | [1（16）](traces/webnavigator/724.json) |
| 725 | non-hard | [0（15）](traces/web2kg/725.json) | [0（21）](traces/webnavigator/725.json) |
| 726 | non-hard | [0（5）](traces/web2kg/726.json) | [0（15）](traces/webnavigator/726.json) |
| 727 | non-hard | [1（9）](traces/web2kg/727.json) | [0（11）](traces/webnavigator/727.json) |
| 728 | non-hard | [1（13）](traces/web2kg/728.json) | [1（7）](traces/webnavigator/728.json) |
| 729 | non-hard | [1（12）](traces/web2kg/729.json) | [0（28）](traces/webnavigator/729.json) |
| 730 | hard | [0（29）](traces/web2kg/730.json) | [1（14）](traces/webnavigator/730.json) |
| 731 | non-hard | [1（15）](traces/web2kg/731.json) | [0（7）](traces/webnavigator/731.json) |
| 732 | non-hard | [1（5）](traces/web2kg/732.json) | [1（5）](traces/webnavigator/732.json) |
| 733 | hard | [1（6）](traces/web2kg/733.json) | [1（6）](traces/webnavigator/733.json) |
| 734 | hard | [1（5）](traces/web2kg/734.json) | [1（8）](traces/webnavigator/734.json) |
| 735 | hard | [1（6）](traces/web2kg/735.json) | [0（5）](traces/webnavigator/735.json) |
