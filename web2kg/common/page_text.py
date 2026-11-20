from .storage import digest


def page_text(node):
    names = list(dict.fromkeys(a["name"] for a in node["actions"] if a["name"]))
    return (f"Title: {node['title']}\nURL: {node['url']}\n"
            f"Headings: {' | '.join(node['headings'])}\n"
            f"Controls: {' | '.join(names)[:1800]}\nContent: {node['text'][:3000]}")[:6500]


def fingerprint(graph):
    return digest({key: page_text(value) for key, value in graph.nodes.items()})
