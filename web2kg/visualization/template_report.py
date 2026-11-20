"""Self-contained inspection report for offline template assignments."""
import json
import hashlib
from importlib.resources import files


def export_template_report(graph, result, directory, cards_path=None):
    nodes = {k: {f: n.get(f) for f in ("id", "title", "url", "depth")} for k, n in graph.nodes.items()}
    cards = result.get('inherited_cards', {})
    if cards_path:
        data = json.loads(cards_path.read_text(encoding="utf-8"))
        source = data["source"]
        layer_path = (graph.directory / source["layer"]).resolve()
        if (layer_path != (directory / "templates.json").resolve()
                or hashlib.sha256(layer_path.read_bytes()).hexdigest() != source["layer_sha256"]
                or hashlib.sha256((graph.directory / "graph.json").read_bytes()).hexdigest() != source["graph_sha256"]):
            raise ValueError("Function cards do not match the displayed template layer")
        cards = {key: row["card"] for key, row in data["cards"].items() if row["status"] == "completed"}
    evidence = {e["id"]: {"source": e["source"], "target": e["target"],
                           "action": {k: e["action"].get(k) for k in ("kind", "role", "name")}}
                for e in graph.data["edges"]}
    payload = json.dumps({**result, "nodes": nodes, "cards": cards, "edge_evidence": evidence},
                         ensure_ascii=False).replace("<", "\\u003c")
    template = files("web2kg.visualization").joinpath("templates.html").read_text(encoding="utf-8")
    (directory / "report.html").write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")
    lines = ["# 页面模板离线结果", "", "算法：" + result["version"], "",
             "匹配规则：" + result["parameters"].get("match_rule", "weighted_score_only")
             + "；阈值：" + str(result["parameters"]["threshold"]), "",
             "父模板 / 入边 / 页面归一化权重：" + " / ".join(f"{w:.4f}" for w in result["parameters"]["weights"]), "",
             "页面内部计分项：" + json.dumps(result['parameters']['page_features'], ensure_ascii=False), "",
             "探索性结构归类，不代表语义正确率或可执行模板。原节点、状态、边与截图不变。", "",
             "| 模板 | 状态数 | URL数 | 代表页面 |", "| --- | ---: | ---: | --- |"]
    for group in result["templates"]:
        urls = {nodes[k]["url"] for k in group["members"]}
        title = nodes[group["representative"]]["title"] or "(untitled)"
        lines.append(f"| {group['id']} | {len(group['members'])} | {len(urls)} | {title.replace('|', '/')} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
