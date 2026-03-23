from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_main_entrypoint_has_no_direct_domain_or_repo_wiring_imports():
    content = _read("main.py")

    forbidden = [
        "from app.decision",
        "from app.core",
        "from app.infrastructure.repositories",
        "from app.data_access.data_manager import DataManager",
    ]

    hits = [needle for needle in forbidden if needle in content]
    assert not hits, f"main.py contains forbidden direct wiring imports: {hits}"


def test_ui_entrypoint_uses_adapterized_interfaces_builder():
    content = _read("app/ui/app.py")

    assert "from app.interfaces.compat import ui_contract" in content
    assert "create_ui_app(" in content


def test_run_dev_is_interfaces_first():
    content = _read("run_dev.py")

    assert "from app.interfaces.web.ui_app import build_default_ui_app" in content
    assert "from app.ui.app import app" not in content
