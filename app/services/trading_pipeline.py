import hashlib
import logging
import os
from datetime import date, timedelta
from typing import Dict, List, Optional

from app.backtesting.history_store import HistoryStore
from app.config.build_settings import build_settings
from app.data_access.data_cleaner import prepare_df
from app.core.decision.allocation import allocate_capital
from app.core.decision.decision_policy import apply_decision_policy
from app.core.decision.recommendation_builder import (
    build_explanation,
    build_recommendation,
)
from app.core.decision.volatility import compute_normalized_volatility
from app.core.decision.recommender import generate_daily_recommendation_payload
from app.reporting.audit_builder import build_audit_metadata, build_audit_summary
from app.services.dependencies import (
    EmailNotifier,
    MarketDataFetcher,
    ModelEnsembleRunner,
)
from app.analysis.go_live_metrics import compute_drawdown_summary, compute_loss_streak
from app.infrastructure.repositories import DataManagerRepository
from app.data_access.data_loader import get_supported_ticker_list


class TradingPipelineService:
    """
    Central orchestration layer for daily trading pipeline steps.
    """

    def __init__(
        self,
        history_store: HistoryStore,
        settings=None,
        logger=None,
        data_fetcher: MarketDataFetcher = None,
        model_runner: ModelEnsembleRunner = None,
        email_notifier: EmailNotifier = None,
        execution_engine=None,
        state_repo=None,
    ):
        if history_store is None:
            raise ValueError("history_store is required")
        if data_fetcher is None:
            raise ValueError("data_fetcher is required")
        if model_runner is None:
            raise ValueError("model_runner is required")
        if email_notifier is None:
            raise ValueError("email_notifier is required")
        if execution_engine is None:
            raise ValueError("execution_engine is required")

        self.history_store = history_store
        self.logger = logger or logging.getLogger(__name__)
        self.settings = settings
        self.data_fetcher = data_fetcher
        self.model_runner = model_runner
        self.email_notifier = email_notifier
        self.execution_engine = execution_engine
        self.state_repo = state_repo or DataManagerRepository(settings=self.settings)

    def _get_settings(self):
        return self.settings or build_settings()

    # --- Step primitives (kept thin for now) ---

    def load_market_data(self, ticker: str, start: str, end: Optional[str] = None):
        return self.data_fetcher.load_data(ticker, start=start, end=end)

    def compute_signals(self, df, ticker: str):
        return prepare_df(df.copy(), ticker)

    def run_decision_models(
        self,
        ticker: str,
        top_n: int = 3,
        debug: bool = True,
        as_of_date=None,
    ):
        return generate_daily_recommendation_payload(
            ticker=ticker,
            history_store=self.history_store,
            top_n=top_n,
            debug=debug,
            data_fetcher=self.data_fetcher,
            model_runner=self.model_runner,
            as_of_date=as_of_date,
        )

    def apply_safety_rules(self, decision: Dict, audit: Dict) -> Dict:
        if os.getenv("VALIDATION_DISABLE_POLICY", "false").lower() == "true":
            return decision
        return apply_decision_policy(decision, audit, settings=self.settings)

    def allocate_capital(self, candidates: List[Dict]) -> List[Dict]:
        return allocate_capital(candidates)

    def persist_decision(
        self,
        payload: Dict,
        decision: Dict,
        explanation: Dict,
        audit: Dict,
        position_sizing: Dict = None,
        decision_source: str = None,
    ) -> int:
        return self.history_store.save_decision(
            payload=payload,
            decision=decision,
            explanation=explanation,
            audit=audit,
            model_votes=payload.get("model_votes", []),
            safety_overrides={
                "safety_override": decision.get("safety_override"),
                "no_trade_reason": decision.get("no_trade_reason"),
                "reasons": decision.get("reasons", []),
                "warnings": decision.get("warnings", []),
            },
            position_sizing=position_sizing,
            decision_source=decision_source,
            model_id=payload.get("model_id"),
            timestamp=payload.get("timestamp"),
        )

    def send_notifications(self, subject: str, body: str, recipient: str) -> None:
        self.email_notifier.send(subject, body, recipient)

    def execute_trades(self, decisions: List[Dict], as_of: date) -> None:
        self.execution_engine.execute(decisions=decisions, as_of=as_of)

    def _log_go_live_metrics(self, ticker: str) -> None:
        drawdown = compute_drawdown_summary(ticker)
        streak = compute_loss_streak(ticker)

        if drawdown["status"] == "ok":
            self.logger.info(
                "GO_LIVE_METRICS %s outcomes=%s max_drawdown=%.4f current_drawdown=%.4f loss_streak=%s",
                ticker,
                drawdown.get("rows"),
                drawdown.get("max_drawdown", 0.0),
                drawdown.get("current_drawdown", 0.0),
                streak.get("loss_streak", 0),
            )
        else:
            self.logger.info(
                "GO_LIVE_METRICS %s outcomes=0 loss_streak=%s status=no_data",
                ticker,
                streak.get("loss_streak", 0),
            )

    # --- Daily pipeline helper ---

    def build_daily_candidate(
        self,
        ticker: str,
        debug: bool = True,
        as_of_date=None,
    ) -> Dict:
        payload = self.run_decision_models(
            ticker=ticker,
            debug=debug,
            as_of_date=as_of_date,
        )

        if payload.get("error") == "NO_MODELS":
            cfg = self._get_settings()
            allow_fallback = getattr(cfg, "ALLOW_NO_MODEL_FALLBACK")
            if allow_fallback:
                self.logger.warning(
                    "NO_MODELS fallback enabled; generating HOLD decision for %s",
                    ticker,
                )
                return self._build_fallback_candidate_no_models(ticker, as_of_date)

        if payload.get("error"):
            raise ValueError(payload["error"])

        if "decision" in payload:
            decision = payload["decision"]
            explanation = payload.get("explanation")
        else:
            decision = build_recommendation(payload, settings=self.settings)
            explanation = build_explanation(payload, decision, settings=self.settings)

        audit = build_audit_metadata(payload, decision)

        decision = self.apply_safety_rules(decision, audit)
        if explanation is None:
            explanation = build_explanation(payload, decision, settings=self.settings)

        return {
            "ticker": ticker,
            "payload": payload,
            "decision": decision,
            "explanation": explanation,
            "audit": audit,
        }

    def _build_fallback_candidate_no_models(
        self,
        ticker: str,
        as_of_date=None,
    ) -> Dict:
        today = as_of_date or date.today()
        start = today - timedelta(days=180)
        df_full = self.data_fetcher.load_data(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=today.strftime("%Y-%m-%d"),
        )
        if df_full.empty:
            raise ValueError("NO_DATA")

        latest_price = float(df_full["Close"].iloc[-1]) if not df_full.empty else None
        volatility = (
            compute_normalized_volatility(df_full) if not df_full.empty else 0.0
        )

        features_hash = None
        if not df_full.empty:
            row = df_full.iloc[-1]
            pieces = [f"{col}={row[col]}" for col in df_full.columns]
            base = "|".join(pieces)
            features_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()

        payload = {
            "ticker": ticker,
            "timestamp": today.isoformat(),
            "as_of_date": today.isoformat(),
            "latest_price": latest_price,
            "volatility": volatility,
            "model_votes": [],
            "avg_confidence": 0.0,
            "avg_wf_score": 0.0,
            "ensemble_quality": 0.0,
            "action_code": 0,
            "decision_source": "fallback",
            "features_hash": features_hash,
        }

        decision = build_recommendation(payload, settings=self.settings)
        cfg = self._get_settings()
        decision.update(
            {
                "action_code": 0,
                "action": getattr(cfg, "ACTION_LABELS")[getattr(cfg, "LANG")][0],
                "strength": "NO_TRADE",
                "no_trade": True,
                "no_trade_reason": "FALLBACK_NO_MODELS",
            }
        )

        audit = build_audit_metadata(payload, decision)
        explanation = build_explanation(payload, decision)

        return {
            "ticker": ticker,
            "payload": payload,
            "decision": decision,
            "explanation": explanation,
            "audit": audit,
            "decision_source": "fallback",
        }

    def get_tickers_to_process(self, ticker: Optional[str] = None) -> List[str]:
        if ticker:
            return [ticker]
        # prefer explicit TICKERS on settings when present
        cfg = self._get_settings()
        if getattr(cfg, "TICKERS", None):
            return getattr(cfg, "TICKERS")
        excluded = set(getattr(cfg, "EXCLUDED_TICKERS", []))
        tickers = list(get_supported_ticker_list())
        return [t for t in tickers if t not in excluded]

    def run_daily(self, dry_run: bool = False, ticker: Optional[str] = None) -> None:
        """Back-compatible entrypoint delegating orchestration to use-case layer."""
        from app.application.use_cases.daily_pipeline_use_case import (
            DailyPipelineUseCase,
        )

        DailyPipelineUseCase(self).run(dry_run=dry_run, ticker=ticker)
