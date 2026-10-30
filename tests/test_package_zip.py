import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_zip", ROOT / "scripts" / "package_zip.py")
package_zip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_zip)


def test_zip_leaves_out_what_teammates_recreate(tmp_path):
    out = tmp_path / "semantic-cache.zip"
    n, _ = package_zip.package(out)
    names = zipfile.ZipFile(out).namelist()
    assert n == len(names) and all(name.startswith("semantic-cache/") for name in names)
    for must in ("README.md", "config.toml", "semcache/linalg.py", "data/faq.csv", "data/demo_script.json",
                 "frontend/dist/index.html", "frontend/package.json", "docs/screenshots/six-stages.gif"):
        assert f"semantic-cache/{must}" in names, must
    for part in ("/.venv/", "/node_modules/", "/artifacts/", "/data/quora/", "/__pycache__/",
                 "/.pytest_cache/", "/__MACOSX/", ".DS_Store", ".pyc", "docs/screenshots/1920x1080"):
        assert not any(part in name for name in names), part


def test_exclusion_rules():
    P = package_zip.PurePosixPath
    assert package_zip.excluded(P("data/quora/train.csv"))
    assert package_zip.excluded(P("frontend/node_modules/x/index.js"))
    assert package_zip.excluded(P(".env"))
    assert not package_zip.excluded(P("data/faq.csv"))
    assert not package_zip.excluded(P("docs/screenshots/six-stages.gif"))
