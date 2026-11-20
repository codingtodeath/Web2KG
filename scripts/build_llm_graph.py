"""Build the independent 120-state Reddit graph with Responses ranking/descriptions."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/webnavigator-reddit-llm-120-v4")
    parser.add_argument("--identity-mode", choices=["interaction", "webnavigator", "semantic"], default="interaction")
    parser.add_argument("--description-similarity", type=float, default=0.8)
    parser.add_argument("--env", type=Path, default=ROOT / "auth/llm-crawl.env")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--serial-exploration', action='store_true', help='Use the legacy synchronous explorer; required for its resume')
    parser.add_argument("--index", action="store_true", help="Build a whole-screenshot Jina index after crawling")
    args = parser.parse_args()
    command = [sys.executable, "-m", "web2kg", "crawl", "--url", "http://127.0.0.1:10499/forums/all",
               "--test-site", "--out", str(args.out.resolve()), "--exploration", "llm_ranked",
               "--max-states", "120", "--max-depth", "5", "--actions-per-page", "12",
               "--max-attempts", "240", "--candidate-limit", "160", "--llm-max-calls", "360",
               "--identity-mode", args.identity_mode, "--description-similarity", str(args.description_similarity),
               "--llm-output-tokens", "2048", "--llm-env", str(args.env.resolve()),
               "--storage-state", str(ROOT / "auth/dev-reddit.json")]
    if args.resume:
        command.append("--resume")
    if args.serial_exploration:
        command.append('--serial-exploration')
    subprocess.run(command, cwd=ROOT, check=True)
    if args.index:
        subprocess.run([sys.executable, "-m", "web2kg", "index", str(args.out.resolve()),
                        "--mode", "jina_screenshot"], cwd=ROOT, check=True)
    print(f"Graph/log viewer: {args.out.resolve() / 'viewer.html'}", flush=True)


if __name__ == "__main__":
    main()
