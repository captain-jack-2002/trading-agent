"""Verify built distributions contain application code and no local runtime state."""

from pathlib import Path
from tarfile import open as open_tar
from zipfile import ZipFile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        str(path.relative_to(root / "src"))
        for path in (root / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    forbidden = {".uv-cache", ".venv", ".tmp", ".podman", "__pycache__", ".env"}
    wheel = root / "dist/trading_agent-0.1.0-py3-none-any.whl"
    with ZipFile(wheel) as archive:
        names = archive.namelist()
    missing = expected - set(names)
    assert not missing, f"Wheel missing source modules: {sorted(missing)}"
    assert any("LICENSE" in name for name in names)
    assert not any(forbidden.intersection(Path(name).parts) for name in names)
    with open_tar(root / "dist/trading_agent-0.1.0.tar.gz") as archive:
        names = archive.getnames()
    assert not any(forbidden.intersection(Path(name).parts) for name in names)
    assert any(name.endswith("/src/trading_agent/execution/paper.py") for name in names)
    print(f"Artifacts verified: {len(expected)} Python modules in wheel; no local runtime state")


if __name__ == "__main__":
    main()
