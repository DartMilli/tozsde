import argparse
from datetime import datetime

from app.config.config import Config
from app.data_access.data_loader import ensure_data_cached
from app.models.model_trainer import train_rl_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single RL model")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument(
        "--model-type",
        default="DQN",
        choices=["DQN", "PPO"],
        help="RL model type",
    )
    parser.add_argument(
        "--reward-strategy",
        default="portfolio_value",
        help="Reward strategy name",
    )
    parser.add_argument(
        "--start",
        default=Config.START_DATE,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=Config.END_DATE,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=Config.RL_TIMESTEPS,
        help="Training timesteps",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional model output path",
    )
    parser.add_argument(
        "--wf-score",
        type=float,
        default=None,
        help="Optional walk-forward score",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not ensure_data_cached(args.ticker, start=args.start, end=args.end):
        print("cache_status: incomplete")
        raise SystemExit(1)
    train_rl_agent(
        ticker=args.ticker,
        model_type=args.model_type,
        start=args.start,
        end=args.end,
        timesteps=args.timesteps,
        model_path=args.model_path,
        reward_strategy=args.reward_strategy,
        wf_score=args.wf_score,
    )
    print(
        "trained",
        args.ticker,
        args.model_type,
        args.reward_strategy,
        datetime.utcnow().isoformat(),
    )


if __name__ == "__main__":
    main()
