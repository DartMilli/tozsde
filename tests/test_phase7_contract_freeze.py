from pathlib import Path


def test_ui_app_frozen_symbols_exist():
    import app.ui.app as ui_app

    required = [
        "DataManager",
        "get_supported_tickers",
        "render_template",
        "load_data",
        "sanitize_dataframe",
        "compute_signals",
        "get_params",
        "Backtester",
        "get_equity_curve_buffer",
        "get_drawdown_curve_buffer",
    ]

    missing = [name for name in required if not hasattr(ui_app, name)]
    assert not missing, f"Missing frozen UI compatibility symbols: {missing}"


def test_main_frozen_wrapper_symbols_exist():
    import main

    required = [
        "parse_arguments",
        "run_daily",
        "run_weekly",
        "run_monthly",
        "run_walk_forward_manual",
        "run_train_rl_manual",
        "run_validation",
        "run_paper_history",
        "_SETTINGS",
    ]

    missing = [name for name in required if not hasattr(main, name)]
    assert not missing, f"Missing frozen main compatibility symbols: {missing}"


def test_legacy_data_manager_import_boundary_is_singleton():
    app_root = Path(__file__).resolve().parents[1] / "app"
    needle = "from app.data_access.data_manager import DataManager"

    matches = []
    for file_path in app_root.rglob("*.py"):
        if file_path.name == "data_manager.py":
            continue
        content = file_path.read_text(encoding="utf-8")
        if needle in content:
            matches.append(file_path)

    expected = [
        app_root / "infrastructure" / "repositories" / "data_manager_repository.py"
    ]
    assert matches == expected, (
        "Direct legacy DataManager import boundary changed. "
        f"Expected only {expected}, got {matches}"
    )
