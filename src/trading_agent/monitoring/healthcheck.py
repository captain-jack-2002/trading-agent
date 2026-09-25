"""Container readiness probe, runnable with python -m."""

import sys
from urllib.request import urlopen


def main() -> int:
    """Use the readiness endpoint's HTTP status as the health contract."""
    try:
        with urlopen("http://127.0.0.1:8000/ready", timeout=3):
            return 0
    except OSError as exc:
        print(f"Readiness check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
