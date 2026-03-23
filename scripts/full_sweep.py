import argparse

from app.backtesting.walk_forward import run_walk_forward
from app.config import get_conf
from app.data_access.data_loader import ensure_data_cached
from app.analysis.analyzer import get_params
from app.infrastructure.logger import setup_logger
from app.optimization.fitness import normalize_wf_score
from app.models.model_trainer import (
    run_backtest,
    run_backtest_minimal,
    run_training,
    train_rl_agent,
)

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full training sweep")
    parser.add_argument(
        "--mode",
        choices=["minimal", "full"],
        default="full",
        help="Run minimal (WF -> RL) or full sweep",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Optional single ticker override",
    )
    parser.add_argument(
        "--model-types",
        default="PPO",
        help="Comma-separated model types for minimal mode (default: PPO)",
    )
    parser.add_argument(
        "--reward-strategy",
        default="portfolio_value",
        help="Reward strategy for minimal mode",
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip backtest/promote step",
    )
    parser.add_argument(
        "--promote-top-n",
        type=int,
        default=3,
        help="Number of top models to promote",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="Generate plots during backtest",
    )
    return parser.parse_args()


def _parse_model_types(value: str):
    raw_items = [item.strip().upper() for item in value.split(",") if item.strip()]
    allowed = {"DQN", "PPO"}
    invalid = [item for item in raw_items if item not in allowed]
    if invalid:
        raise SystemExit(f"Invalid model types: {', '.join(invalid)}")
    return raw_items or ["PPO"]


def main():
    args = parse_args()
    cfg = get_conf(None)
    tickers = [args.ticker] if args.ticker else cfg.get_supported_tickers()

    if args.mode == "minimal":
        model_types = _parse_model_types(args.model_types)
        for ticker in tickers:
            if not ensure_data_cached(
                ticker,
                start=getattr(cfg, "START_DATE", None),
                end=getattr(cfg, "END_DATE", None),
            ):
                raise SystemExit(f"cache_status: incomplete ({ticker})")
            wf_summary = run_walk_forward(ticker)
            if not wf_summary:
                logger.warning(
                    f"{ticker}: walk-forward returned no summary; skipping RL"
                )
                continue
            wf_score = wf_summary.get("normalized_score")
            if wf_score is None:
                raw_fitness = wf_summary.get("raw_fitness")
                if raw_fitness is None:
                    raw_fitness = wf_summary.get("wf_fitness", 0.0)
                wf_score = normalize_wf_score(raw_fitness)
            params = wf_summary.get("best_params") or get_params(ticker)
            for model_type in model_types:
                train_rl_agent(
                    ticker=ticker,
                    model_type=model_type,
                    start=getattr(cfg, "START_DATE", None),
                    end=getattr(cfg, "END_DATE", None),
                    params=params,
                    timesteps=getattr(cfg, "RL_TIMESTEPS", 100000),
                    reward_strategy=args.reward_strategy,
                    wf_score=wf_score,
                    wf_summary=wf_summary,
                )
        if not args.no_backtest:
            run_backtest_minimal(
                tickers=tickers,
                model_types=model_types,
                reward_strategy=args.reward_strategy,
                save_to_file=True,
            )
        return

    for ticker in tickers:
        start_date = getattr(cfg, "START_DATE", None)
        end_date = getattr(cfg, "END_DATE", None)
        if not ensure_data_cached(ticker, start=start_date, end=end_date):
            raise SystemExit(f"cache_status: incomplete ({ticker})")
        run_walk_forward(ticker)

    run_training(tickers=tickers)

    if not args.no_backtest:
        run_backtest(
            save_to_file=True,
            create_plots=args.with_plots,
            promote_top_n=args.promote_top_n,
            tickers=tickers,
        )


if __name__ == "__main__":
    main()
