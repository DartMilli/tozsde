import argparse
from datetime import datetime

from app.core.data_loader import get_supported_ticker_list
from app.core.data_loader import run_full_download
from app.core.genetic_optimizer import run_optimization
from app.core.model_trainer import run_training, run_backtest
from app.ui.app import app as flask_app  # A Flask app importálása

# ---- Központi Konfiguráció ----
# Minden folyamat innen veszi a paramétereket
CONFIG = {
    "START_DATE": "2020-01-01",
    "END_DATE": datetime.today().strftime("%Y-%m-%d"),
    "TICKERS": [
        t
        for t in get_supported_ticker_list()
        if t not in ["OTP.BD", "MOL.BD", "RICHTER.BD"]
    ],
    "OPTIMIZER_GENERATIONS": 30,
    "OPTIMIZER_POPULATION": 50,
    "RL_TIMESTEPS": 100_000,
}


def handle_update(args):
    """Adatok letöltését végző parancs."""
    print("Adatok letöltése és frissítése...")
    run_full_download(CONFIG)
    print("Adatok frissítve.")


def handle_train(args):
    """Optimalizálást és tanítást végző parancs."""
    print("Paraméter optimalizáció indítása...")
    run_optimization(CONFIG)
    print("Optimalizáció befejezve.")

    print("RL Modellek tanítása...")
    run_training(CONFIG, force_retrain=args.force)
    print("Modellek tanítása befejezve.")


def handle_train_and_test(args):
    """Optimalizálást és tanítást és visszatesztelést végző parancs."""
    print("Paraméter optimalizáció indítása...")
    run_optimization(CONFIG)
    print("Optimalizáció befejezve.")

    print("RL Modellek tanítása...")
    run_training(CONFIG, force_retrain=args.force)
    print("Modellek tanítása befejezve.")

    print("RL Modellek tesztelése...")
    run_backtest(CONFIG)
    print("Modellek tesztelve.")


def handle_backtest(args):
    """Modellek ellentőrzése és visszatesztelése"""
    print("RL Modellek tesztelése...")
    run_backtest(CONFIG)
    print("Modellek tesztelve.")


def handle_runserver(args):
    """A Flask webszerver indítása."""
    print(f"Webszerver indítása a(z) {args.host}:{args.port} címen...")
    flask_app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kereskedő Applikáció Vezérlőpult.")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Elérhető parancsok"
    )

    # 'update' parancs
    parser_update = subparsers.add_parser(
        "update", help="Historikus adatok letöltése és frissítése."
    )
    parser_update.set_defaults(func=handle_update)

    # 'train' parancs
    parser_train = subparsers.add_parser(
        "train", help="Paraméter optimalizáció és RL modellek tanítása."
    )
    parser_train.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Minden modell újratanításának kényszerítése.",
    )
    parser_train.set_defaults(func=handle_train)

    # 'train-and-test' parancs
    parser_train = subparsers.add_parser(
        "train-and-test",
        help="Paraméter optimalizáció és RL modellek tanítása és ellenőrzése.",
    )
    parser_train.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Minden modell újratanításának kényszerítése.",
    )
    parser_train.set_defaults(func=handle_train_and_test)

    # 'backtest' parancs
    parser_backtest = subparsers.add_parser("backtest", help="RL modellek tesztelése.")
    parser_update.set_defaults(func=handle_backtest)

    # 'runserver' parancs
    parser_server = subparsers.add_parser(
        "runserver", help="A Flask webszerver elindítása."
    )
    parser_server.add_argument(
        "--host", type=str, default="127.0.0.1", help="A szerver host címe."
    )
    parser_server.add_argument(
        "--port", type=int, default=5000, help="A szerver portja."
    )
    parser_server.set_defaults(func=handle_runserver)

    args = parser.parse_args()
    args.func(args)
