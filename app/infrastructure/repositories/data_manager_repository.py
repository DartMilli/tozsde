"""Compatibility adapter for legacy data access during migration.

The adapter keeps the historical DataManager-shaped surface while gradually
delegating domain-specific operations to dedicated repository classes.
"""

from typing import Any, Optional, TYPE_CHECKING

from app.infrastructure.repositories.sqlite_decision_repository import (
    SqliteDecisionRepository,
)
from app.infrastructure.repositories.sqlite_model_repository import (
    SqliteModelRepository,
)
from app.infrastructure.repositories.sqlite_ohlcv_repository import (
    SqliteOhlcvRepository,
)

if TYPE_CHECKING:
    from app.data_access.data_manager import DataManager


class DataManagerRepository:
    def __init__(self, data_manager: Optional[Any] = None, settings=None):
        self._settings = settings

        # Avoid wrapping adapters into adapters; keep a single legacy backend.
        if isinstance(data_manager, DataManagerRepository):
            self._dm = data_manager._dm
        elif data_manager is not None:
            self._dm = data_manager
        else:
            from app.data_access.data_manager import DataManager as LegacyDataManager

            try:
                self._dm = LegacyDataManager(settings=settings)
            except TypeError:
                self._dm = LegacyDataManager()

        self._ohlcv_repo = SqliteOhlcvRepository(
            data_manager=self._dm,
            settings=settings,
        )
        self._decision_repo = None
        self._model_repo = None

    def _get_decision_repo(self):
        if self._decision_repo is None:
            try:
                self._decision_repo = SqliteDecisionRepository(settings=self._settings)
            except Exception:
                self._decision_repo = False
        return self._decision_repo

    def _get_model_repo(self):
        if self._model_repo is None:
            try:
                self._model_repo = SqliteModelRepository(settings=self._settings)
            except Exception:
                self._model_repo = False
        return self._model_repo

    def load_ohlcv(self, ticker: str, start_date=None) -> Any:
        return self._ohlcv_repo.load_ohlcv(ticker=ticker, start_date=start_date)

    def fetch_ohlcv(self, ticker: str, start_date=None, end_date=None) -> Any:
        # Keep legacy end-date behavior when backend exposes get_ohlcv.
        if end_date is not None and hasattr(self._dm, "get_ohlcv"):
            return self._dm.get_ohlcv(ticker, start_date, end_date)
        return self.load_ohlcv(ticker=ticker, start_date=start_date)

    def save_ohlcv(self, ticker: str, df: Any) -> None:
        return self._ohlcv_repo.save_ohlcv(ticker=ticker, df=df)

    def save_decision(self, decision: dict) -> int:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_decision(decision)
        return self._dm.save_decision(decision)

    def fetch_decision(self, decision_id: int) -> dict:
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_decision(decision_id)
        return self._dm.fetch_decision(decision_id)

    def fetch_decisions_for_ticker(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[dict]:
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_decisions_for_ticker(ticker, start_date, end_date)
        return self._dm.fetch_decisions_for_ticker(ticker, start_date, end_date)

    def save_history_record(self, **kwargs):
        repo = self._get_decision_repo()
        if repo:
            return repo.save_history_record(**kwargs)
        return self._dm.save_history_record(**kwargs)

    def fetch_history_records_by_ticker(self, ticker: str):
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_history_records_by_ticker(ticker)
        return self._dm.fetch_history_records_by_ticker(ticker)

    def has_decision_for_date(self, ticker: str, as_of_date) -> bool:
        repo = self._get_decision_repo()
        if repo:
            return repo.has_decision_for_date(ticker=ticker, as_of_date=as_of_date)
        return self._dm.has_decision_for_date(ticker=ticker, as_of_date=as_of_date)

    def fetch_history_range(self, ticker: str, start_iso: str, end_iso: str):
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_history_range(
                ticker=ticker,
                start_iso=start_iso,
                end_iso=end_iso,
            )
        return self._dm.fetch_history_range(
            ticker=ticker,
            start_iso=start_iso,
            end_iso=end_iso,
        )

    def fetch_recent_outcomes(self, ticker: str, n: int = 3):
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_recent_outcomes(ticker=ticker, n=n)
        return self._dm.fetch_recent_outcomes(ticker=ticker, n=n)

    def save_outcome(
        self,
        decision_id: int,
        ticker: str,
        decision_timestamp: str,
        pnl_pct: float,
        success: bool,
        future_return: float,
        exit_reason: str,
        horizon_days: int,
        outcome_json: str,
    ) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_outcome(
                decision_id=decision_id,
                ticker=ticker,
                decision_timestamp=decision_timestamp,
                pnl_pct=pnl_pct,
                success=success,
                future_return=future_return,
                exit_reason=exit_reason,
                horizon_days=horizon_days,
                outcome_json=outcome_json,
            )
        return self._dm.save_outcome(
            decision_id=decision_id,
            ticker=ticker,
            decision_timestamp=decision_timestamp,
            pnl_pct=pnl_pct,
            success=success,
            future_return=future_return,
            exit_reason=exit_reason,
            horizon_days=horizon_days,
            outcome_json=outcome_json,
        )

    def get_unevaluated_buy_decisions(self, limit=100):
        repo = self._get_decision_repo()
        if repo:
            return repo.get_unevaluated_buy_decisions(limit=limit)
        return self._dm.get_unevaluated_buy_decisions(limit=limit)

    def save_portfolio_state(
        self,
        timestamp: str,
        cash: float,
        equity: float,
        pnl_pct: float,
        positions_json: str,
        source: str,
    ) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_portfolio_state(
                timestamp=timestamp,
                cash=cash,
                equity=equity,
                pnl_pct=pnl_pct,
                positions_json=positions_json,
                source=source,
            )
        return self._dm.save_portfolio_state(
            timestamp=timestamp,
            cash=cash,
            equity=equity,
            pnl_pct=pnl_pct,
            positions_json=positions_json,
            source=source,
        )

    def fetch_portfolio_state_range(self, start_iso: str, end_iso: str):
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_portfolio_state_range(
                start_iso=start_iso, end_iso=end_iso
            )
        return self._dm.fetch_portfolio_state_range(
            start_iso=start_iso, end_iso=end_iso
        )

    def fetch_latest_portfolio_state(self, source: str = None):
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_latest_portfolio_state(source=source)
        return self._dm.fetch_latest_portfolio_state(source=source)

    def save_decision_effectiveness(
        self,
        decision_id: int,
        ticker: str,
        effectiveness_score: float,
        components_json: str,
    ) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_decision_effectiveness(
                decision_id=decision_id,
                ticker=ticker,
                effectiveness_score=effectiveness_score,
                components_json=components_json,
            )
        return self._dm.save_decision_effectiveness(
            decision_id=decision_id,
            ticker=ticker,
            effectiveness_score=effectiveness_score,
            components_json=components_json,
        )

    def save_decision_effectiveness_rolling(
        self,
        ticker: str,
        window_days: int,
        as_of_date: str,
        metrics_json: str,
    ) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_decision_effectiveness_rolling(
                ticker=ticker,
                window_days=window_days,
                as_of_date=as_of_date,
                metrics_json=metrics_json,
            )
        return self._dm.save_decision_effectiveness_rolling(
            ticker=ticker,
            window_days=window_days,
            as_of_date=as_of_date,
            metrics_json=metrics_json,
        )

    def save_decision_quality_metrics(
        self, ticker: str, start_date: str, end_date: str, metrics_json: str
    ) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_decision_quality_metrics(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                metrics_json=metrics_json,
            )
        return self._dm.save_decision_quality_metrics(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            metrics_json=metrics_json,
        )

    def save_confidence_calibration(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        method: str,
        params_json: str,
        metrics_json: str,
    ) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_confidence_calibration(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                method=method,
                params_json=params_json,
                metrics_json=metrics_json,
            )
        return self._dm.save_confidence_calibration(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            method=method,
            params_json=params_json,
            metrics_json=metrics_json,
        )

    def fetch_latest_confidence_calibration(
        self, ticker: str, as_of_date: str | None = None
    ):
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_latest_confidence_calibration(
                ticker=ticker,
                as_of_date=as_of_date,
            )
        return self._dm.fetch_latest_confidence_calibration(
            ticker=ticker,
            as_of_date=as_of_date,
        )

    def save_wf_stability_metrics(self, ticker: str, metrics_json: str) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_wf_stability_metrics(
                ticker=ticker, metrics_json=metrics_json
            )
        return self._dm.save_wf_stability_metrics(
            ticker=ticker, metrics_json=metrics_json
        )

    def save_safety_stress_results(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        scenario: str,
        results_json: str,
    ) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_safety_stress_results(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                scenario=scenario,
                results_json=results_json,
            )
        return self._dm.save_safety_stress_results(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            scenario=scenario,
            results_json=results_json,
        )

    def save_validation_report(self, report_json: str) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_validation_report(report_json=report_json)
        return self._dm.save_validation_report(report_json=report_json)

    def fetch_latest_validation_report(self):
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_latest_validation_report()
        return self._dm.fetch_latest_validation_report()

    def update_history_audit(self, row_id, new_audit_blob) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.update_history_audit(
                row_id=row_id, new_audit_blob=new_audit_blob
            )
        return self._dm.update_history_audit(
            row_id=row_id, new_audit_blob=new_audit_blob
        )

    def save_model_reliability(self, ticker, date, score_details) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_model_reliability(
                ticker=ticker, date=date, score_details=score_details
            )
        return self._dm.save_model_reliability(
            ticker=ticker,
            date=date,
            score_details=score_details,
        )

    def save_market_data(self, symbol: str, df) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_market_data(symbol=symbol, df=df)
        return self._dm.save_market_data(symbol=symbol, df=df)

    def get_market_data(self, symbol: str, days: int = 1):
        repo = self._get_decision_repo()
        if repo:
            return repo.get_market_data(symbol=symbol, days=days)
        return self._dm.get_market_data(symbol=symbol, days=days)

    def get_ticker_historical_recommendations(self, ticker, start, end):
        repo = self._get_decision_repo()
        if repo:
            return repo.get_ticker_historical_recommendations(
                ticker=ticker,
                start=start,
                end=end,
            )
        return self._dm.get_ticker_historical_recommendations(
            ticker=ticker,
            start=start,
            end=end,
        )

    def get_strategy_accuracy(self, ticker, lookback_decisions=20):
        repo = self._get_decision_repo()
        if repo:
            return repo.get_strategy_accuracy(
                ticker=ticker,
                lookback_decisions=lookback_decisions,
            )
        return self._dm.get_strategy_accuracy(
            ticker=ticker,
            lookback_decisions=lookback_decisions,
        )

    def get_today_recommendations(self):
        repo = self._get_decision_repo()
        if repo:
            return repo.get_today_recommendations()
        return self._dm.get_today_recommendations()

    def save_walk_forward_result(self, ticker: str, result_json: str) -> None:
        repo = self._get_decision_repo()
        if repo:
            return repo.save_walk_forward_result(ticker=ticker, result_json=result_json)
        return self._dm.save_walk_forward_result(ticker=ticker, result_json=result_json)

    def fetch_latest_decision_quality_metrics(self, ticker: str):
        repo = self._get_decision_repo()
        if repo:
            return repo.fetch_latest_decision_quality_metrics(ticker=ticker)
        return self._dm.fetch_latest_decision_quality_metrics(ticker=ticker)

    def update_market_data(self):
        return self._dm.update_market_data()

    def initialize_tables(self) -> None:
        repo = self._get_decision_repo()
        if repo:
            repo.initialize_tables()
        if hasattr(self._dm, "initialize_tables"):
            self._dm.initialize_tables()

    def connection(self):
        repo = self._get_decision_repo()
        if repo:
            return repo.connection()
        return self._dm.connection()

    def save_model(self, **model_info) -> None:
        repo = self._get_model_repo()
        if repo:
            return repo.save_model(model_info)
        if hasattr(self._dm, "save_model"):
            return self._dm.save_model(**model_info)
        if hasattr(self._dm, "register_model"):
            return self._dm.register_model(**model_info)
        raise AttributeError("save_model/register_model is not available")

    def fetch_model(self, model_id: str) -> dict:
        repo = self._get_model_repo()
        if repo:
            return repo.fetch_model(model_id)
        return self._dm.fetch_model(model_id)

    def fetch_models_for_ticker(self, ticker: str) -> list[dict]:
        repo = self._get_model_repo()
        if repo:
            return repo.fetch_models_for_ticker(ticker)
        return self._dm.fetch_models_for_ticker(ticker)

    def register_model(
        self,
        model_id: str,
        ticker: str,
        model_type: str,
        wf_score: float,
        model_path: str,
        status: str = "candidate",
        metadata_json: str = None,
    ) -> None:
        repo = self._get_model_repo()
        if repo:
            return repo.register_model(
                model_id=model_id,
                ticker=ticker,
                model_type=model_type,
                wf_score=wf_score,
                model_path=model_path,
                status=status,
                metadata_json=metadata_json,
            )
        return self._dm.register_model(
            model_id=model_id,
            ticker=ticker,
            model_type=model_type,
            wf_score=wf_score,
            model_path=model_path,
            status=status,
            metadata_json=metadata_json,
        )

    def update_model_status(self, model_id: str, status: str) -> None:
        repo = self._get_model_repo()
        if repo:
            return repo.update_model_status(model_id=model_id, status=status)
        return self._dm.update_model_status(model_id=model_id, status=status)

    def fetch_active_models(self, ticker: str, limit: int = 3):
        repo = self._get_model_repo()
        if repo:
            return repo.fetch_active_models(ticker=ticker, limit=limit)
        return self._dm.fetch_active_models(ticker=ticker, limit=limit)

    def save_model_trust_metrics(
        self,
        model_path: str,
        ticker: str,
        trust_weight: float,
        metrics_json: str,
    ) -> None:
        repo = self._get_model_repo()
        if repo:
            return repo.save_model_trust_metrics(
                model_path=model_path,
                ticker=ticker,
                trust_weight=trust_weight,
                metrics_json=metrics_json,
            )
        return self._dm.save_model_trust_metrics(
            model_path=model_path,
            ticker=ticker,
            trust_weight=trust_weight,
            metrics_json=metrics_json,
        )

    def fetch_latest_model_trust_weights(self, ticker: str):
        repo = self._get_model_repo()
        if repo:
            return repo.fetch_latest_model_trust_weights(ticker=ticker)
        return self._dm.fetch_latest_model_trust_weights(ticker=ticker)

    def get_top_models(self, ticker: str, limit: int = 3):
        repo = self._get_model_repo()
        if repo:
            return repo.get_top_models(ticker=ticker, limit=limit)
        return self._dm.get_top_models(ticker=ticker, limit=limit)
