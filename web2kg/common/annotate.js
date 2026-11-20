(actions) => {
  const host = document.createElement("div");
  host.setAttribute("data-web2kg-overlay", "");
  host.setAttribute("aria-hidden", "true");
  host.style.cssText =
    "position:absolute;left:0;top:0;width:0;height:0;z-index:2147483647;pointer-events:none;";
  const shadow = host.attachShadow({ mode: "open" });
  const width = document.documentElement.scrollWidth;
  const height = document.documentElement.scrollHeight;
  const items = actions.map((action, index) => {
    const state = action.annotation_state || "candidate";
    const color = {
      candidate: "#1266c5",
      selected: "#ea580c",
      attempted: "#c026d3",
    }[state];
    const item = {
      id: action.id,
      number: action.number || index + 1,
      activity: state,
      color,
    };
    const element = document.evaluate(
      action.xpath,
      document,
      null,
      XPathResult.FIRST_ORDERED_NODE_TYPE,
      null,
    ).singleNodeValue;
    if (!element) return { ...item, status: "missing" };
    const box = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    if (
      box.width <= 0 ||
      box.height <= 0 ||
      style.visibility === "hidden" ||
      style.display === "none" ||
      Number(style.opacity) === 0
    )
      return { ...item, status: "hidden" };
    const x = box.left + scrollX,
      y = box.top + scrollY;
    // Screen-reader-only links can exist in AX but lie outside the document.
    if (
      x + box.width <= 0 ||
      y + box.height <= 0 ||
      x >= width ||
      y >= height ||
      style.clip !== "auto" ||
      style.clipPath !== "none"
    )
      return { ...item, status: "clipped" };
    const rect = {
      x: Math.max(0, x),
      y: Math.max(0, y),
      width: Math.min(width, x + box.width) - Math.max(0, x),
      height: Math.min(height, y + box.height) - Math.max(0, y),
    };
    const outline = document.createElement("div");
    outline.style.cssText = `position:absolute;box-sizing:border-box;left:${rect.x}px;top:${rect.y}px;width:${rect.width}px;height:${rect.height}px;border:${state === "candidate" ? 2 : 3}px solid ${color};border-radius:3px;background:${color}12;z-index:${state === "candidate" ? 0 : 1};`;
    const label = document.createElement("span");
    label.textContent = `${item.number}${state === "attempted" ? " · 已试" : state === "selected" ? " · 入选" : ""}`;
    label.style.cssText =
      "position:absolute;left:0;top:0;transform:translateY(-55%);font:bold 11px/14px monospace;color:white;background:#1266c5;padding:0 3px;border-radius:2px;white-space:nowrap;";
    label.style.background = color;
    if (rect.y < 8) label.style.transform = "none";
    outline.append(label);
    shadow.append(outline);
    return { ...item, status: "marked", rect };
  });
  document.documentElement.append(host);
  return { host, items, width, height };
};
