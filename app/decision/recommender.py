from app.core.decision.recommender import (
    generate_daily_recommendation_payload as core_generate_daily_recommendation_payload,
)
from app.data_access.data_cleaner import prepare_df
from app.services.dependencies import MarketDataFetcher
from app.decision.decision_engine import DecisionEngine
from app.decision.recommendation_builder import (
    build_explanation,
    build_recommendation,
)
from app.decision.safety_rules import SafetyRuleEngine


try:
    from app.models.rl_inference import RLModelEnsembleRunner
except Exception:
    RLModelEnsembleRunner = None


def load_data(ticker: str, start: str, end: str):
    return MarketDataFetcher().load_data(ticker, start=start, end=end)


def generate_daily_recommendation_payload(
    ticker: str,
    history_store,
    top_n: int = 3,
    debug=True,
    data_fetcher=None,
    model_runner=None,
    as_of_date=None,
    settings=None,
) -> dict:
    return core_generate_daily_recommendation_payload(
        ticker=ticker,
        history_store=history_store,
        top_n=top_n,
        debug=debug,
        data_fetcher=data_fetcher,
        model_runner=model_runner,
        as_of_date=as_of_date,
        settings=settings,
        load_data_fn=load_data,
        prepare_df_fn=prepare_df,
        rl_runner_cls=RLModelEnsembleRunner,
        safety_rule_engine_cls=SafetyRuleEngine,
        decision_engine_cls=DecisionEngine,
        build_recommendation_fn=build_recommendation,
        build_explanation_fn=build_explanation,
    )
