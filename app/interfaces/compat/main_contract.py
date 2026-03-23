def run_daily(container, dry_run: bool = False, ticker: str = None):
    return container.daily_pipeline.run(dry_run=dry_run, ticker=ticker)


def run_weekly(container, dry_run: bool = False):
    return container.weekly_reliability.run(dry_run=dry_run)


def run_monthly(container, dry_run: bool = False):
    return container.monthly_retraining.run(dry_run=dry_run)


def run_walk_forward_manual(container, ticker: str, dry_run: bool = False):
    return container.walk_forward.run(ticker=ticker)


def run_train_rl_manual(container, ticker: str, dry_run: bool = False):
    return container.train_rl.run(ticker=ticker)


def run_validation(
    container,
    ticker: str = None,
    start_date: str = None,
    end_date: str = None,
    scenario: str = "elevated_volatility",
    include_calibration: bool = True,
    repeat: int = 1,
    compare_repeat: bool = False,
):
    return container.phase5_validation.run(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        scenario=scenario,
        include_calibration=include_calibration,
        repeat=repeat,
        compare_repeat=compare_repeat,
    )


def run_paper_history(container, ticker: str, start_date: str, end_date: str):
    return container.historical_paper.run(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
