"""Save login state for a local WebArena-Verified Reddit research copy."""
import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:10499")
    parser.add_argument("--out", type=Path, default=Path("auth/dev-reddit.json"))
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("Only local Verified environments are supported")
    # Public benchmark account; environment overrides are never printed or saved in the graph.
    user = os.getenv("VERIFIED_REDDIT_USER", "MarvelsGrantMan136")
    password = os.environ['VERIFIED_REDDIT_PASSWORD']
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(extra_http_headers={"X-Postmill-Auto-Login": f"{user}:{password}"})
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle")
        if not page.get_by_role("button", name=user, exact=True).count():
            raise RuntimeError("Verified auto-login did not expose the account menu")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(args.out))
        browser.close()
    print("Reddit login state saved (credentials omitted).")


if __name__ == "__main__":
    main()
