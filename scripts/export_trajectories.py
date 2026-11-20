"""Build an offline HTML viewer from completed, ordered execution batches."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web2kg.visualization.trajectory_report import export_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path, help="Oldest to newest")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = export_report(args.batches, args.output)
    print(f"Exported {len(result['tasks'])} tasks / {sum(t['steps'] for t in result['tasks'])} steps: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
