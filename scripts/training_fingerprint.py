import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from app.config.config import Config, EXCLUDED_TICKERS
from app.models.model_trainer import TradingEnv


DEFAULT_WATCHED = [
    "app/models",
    "app/decision",
    "app/analysis",
    "app/data_access",
    "app/optimization",
    "app/backtesting/walk_forward.py",
    "app/config/config.py",
    "requirements.txt",
]

SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", "logs", "backups"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".db"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
                for name in files:
                    file_path = Path(root) / name
                    if any(part in SKIP_DIR_NAMES for part in file_path.parts):
                        continue
                    if file_path.suffix in SKIP_SUFFIXES:
                        continue
                    yield file_path
        elif path.is_file():
            if path.suffix in SKIP_SUFFIXES:
                continue
            yield path


def compute_fingerprint(root: Path, watched: List[str]) -> Dict:
    watched_paths = [root / Path(p) for p in watched]
    file_hashes = {}
    for path in sorted(iter_files(watched_paths)):
        rel = path.relative_to(root).as_posix()
        file_hashes[rel] = sha256_file(path)

    tickers = sorted(Config.get_supported_tickers())

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "watched_paths": watched,
        "file_hashes": file_hashes,
        "tickers": tickers,
        "excluded_tickers": EXCLUDED_TICKERS,
        "rl_timesteps": Config.RL_TIMESTEPS,
        "optimizer_population": Config.OPTIMIZER_POPULATION,
        "optimizer_generations": Config.OPTIMIZER_GENERATIONS,
        "ga_before_rl": True,
        "training_start_date": Config.START_DATE,
        "training_end_date": Config.END_DATE,
        "model_types": ["DQN", "PPO"],
        "reward_strategies": TradingEnv.get_reward_strategies(),
        "python_version": sys.version.split()[0],
    }


def write_summary(path: Path, fingerprint: Dict) -> None:
    summary = {
        "generated_at": fingerprint.get("generated_at"),
        "tickers": fingerprint.get("tickers", []),
        "excluded_tickers": fingerprint.get("excluded_tickers", []),
        "model_types": fingerprint.get("model_types", []),
        "reward_strategies": fingerprint.get("reward_strategies", []),
        "rl_timesteps": fingerprint.get("rl_timesteps"),
        "optimizer_population": fingerprint.get("optimizer_population"),
        "optimizer_generations": fingerprint.get("optimizer_generations"),
        "ga_before_rl": fingerprint.get("ga_before_rl"),
        "training_start_date": fingerprint.get("training_start_date"),
        "training_end_date": fingerprint.get("training_end_date"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def normalize_for_compare(payload: Dict) -> Dict:
    copy = dict(payload)
    copy.pop("generated_at", None)
    return copy


def models_exist(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    return any(p.suffix == ".zip" for p in model_dir.glob("*.zip"))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute training fingerprint")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write fingerprint to models/training_fingerprint.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare fingerprint with stored one",
    )
    parser.add_argument(
        "--output",
        default=str(Config.MODEL_DIR / "training_fingerprint.json"),
        help="Fingerprint JSON path",
    )
    parser.add_argument(
        "--summary-out",
        default=str(Config.MODEL_DIR / "training_fingerprint_summary.json"),
        help="Summary JSON path",
    )
    parser.add_argument(
        "--no-models-ok",
        action="store_true",
        help="Do not fail when models are missing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output)
    summary_path = Path(args.summary_out)

    fingerprint = compute_fingerprint(root, DEFAULT_WATCHED)

    has_models = models_exist(Config.MODEL_DIR)
    if not has_models and not args.no_models_ok and args.check:
        print("NO_MODELS: training required")
        sys.exit(3)

    if args.check:
        if not output_path.exists():
            print("NO_FINGERPRINT: training required")
            sys.exit(2)

        existing = json.loads(output_path.read_text())
        if normalize_for_compare(existing) != normalize_for_compare(fingerprint):
            print("FINGERPRINT_CHANGED: training required")
            sys.exit(1)

        print("FINGERPRINT_OK: training not required")
        sys.exit(0)

    if args.write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(fingerprint, indent=2))
        write_summary(summary_path, fingerprint)
        print(f"WROTE {output_path}")
        print(f"WROTE {summary_path}")
        return

    print(json.dumps(fingerprint, indent=2))


if __name__ == "__main__":
    main()
