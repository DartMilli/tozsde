import logging
from datetime import date, timedelta

from app.config.build_settings import build_settings
from app.data_access.data_loader import get_market_volatility_index
from app.infrastructure.repositories import DataManagerRepository

logger = logging.getLogger(__name__)


STRICTNESS_MULTIPLIERS = {
    "RELAXED": {"cooldown": 0.75, "vix": 1.1},
    "NORMAL": {"cooldown": 1.0, "vix": 1.0},
    "STRICT": {"cooldown": 1.5, "vix": 0.8},
}


class SafetyRuleEngine:
    """Applies global safety rules on top of a decision.

    Args:
        history_store: Required – provides decision history for cooldown/drawdown checks.
        settings: Strongly recommended – pass an explicit Settings instance (injected
            via bootstrap).  If omitted, ``build_settings()`` is called as a fallback,
            which performs I/O (.env read, directory creation) and prevents isolated
            unit testing.  This fallback is **deprecated** and will be removed in a
            future sprint (A1 – Sprint 14 tech-debt).
    """

    def __init__(self, history_store, settings=None):
        self.history_store = history_store
        self.settings = settings
        self.last_check_date = None
        self.is_bear_cache = False
        self._drawdown_disabled_logged = False
        if settings is None:
            logger.debug(
                "SafetyRuleEngine instantiated without explicit settings; "
                "build_settings() fallback will be used. Pass settings= for "
                "clean-architecture compliance."
            )

    def apply(
        self,
        ticker: str,
        decision: dict,
        today: date,
    ) -> dict:
        reasons = []

        cfg = self._get_settings()
        regime_policy = decision.get("regime_policy") or {}
        strictness = regime_policy.get("safety_strictness", "NORMAL")
        hold_label = getattr(cfg, "ACTION_LABELS")[getattr(cfg, "LANG")][0]

        def _force_hold(reason_code: str, reason_text: str):
            decision["action_code"] = 0
            decision["action"] = hold_label
            decision["no_trade"] = True
            decision["no_trade_reason"] = reason_code
            decision["strength"] = "NO_TRADE"
            reasons.append(reason_text)

        if self._in_cooldown(ticker, today, strictness=strictness):
            _force_hold("COOLDOWN", "cooldown active")

        if self._recent_drawdown(ticker):
            _force_hold("DRAWDOWN_GUARD", "recent drawdown")

        if decision.get("ensemble_quality") == "CHAOTIC":
            _force_hold("CHAOTIC_ENSEMBLE", "chaotic ensemble")

        if decision["action_code"] == 1:
            if self._is_bear_market():
                if decision["strength"] != "STRONG":
                    _force_hold(
                        "BEAR_MARKET_FILTER",
                        "Global market is bearish (SPY < SMA200). Only STRONG signals allowed.",
                    )
                else:
                    decision.setdefault("warnings", []).append(
                        "Trading against global bear trend."
                    )

        current_vix = self._get_market_volatility_index()

        cfg = self._get_settings()
        vix_multiplier = STRICTNESS_MULTIPLIERS.get(
            strictness,
            STRICTNESS_MULTIPLIERS["NORMAL"],
        )["vix"]
        max_vix = getattr(cfg, "MAX_VIX_THRESHOLD") * vix_multiplier
        if current_vix and current_vix > max_vix:
            if decision["action_code"] == 1:
                _force_hold(
                    "HIGH_VIX_PANIC", f"Market fear is extreme (VIX={current_vix:.1f})"
                )

        decision["safety_override"] = bool(reasons)
        decision.setdefault("reasons", []).extend(reasons)
        return decision

    def _in_cooldown(
        self,
        ticker: str,
        today: date,
        strictness: str = "NORMAL",
    ) -> bool:
        cfg = self._get_settings()
        cooldown_multiplier = STRICTNESS_MULTIPLIERS.get(
            strictness,
            STRICTNESS_MULTIPLIERS["NORMAL"],
        )["cooldown"]
        lookback_days = max(
            1,
            round(getattr(cfg, "COOLDOWN_DAYS") * cooldown_multiplier),
        )
        decisions = self.history_store.load_range(
            ticker,
            start=today - timedelta(days=lookback_days),
            end=today,
        )

        trade_actions = [d["action_code"] for d in decisions if d["action_code"] != 0]
        return len(trade_actions) >= getattr(cfg, "COOLDOWN_MAX_TRADES")

    def _recent_drawdown(self, ticker: str) -> bool:
        cfg = self._get_settings()
        enable_drawdown = getattr(cfg, "ENABLE_DRAWDOWN_GUARD")
        if not enable_drawdown:
            if not self._drawdown_disabled_logged:
                logger.info(
                    "Drawdown guard disabled (ENABLE_DRAWDOWN_GUARD=false); skipping check."
                )
                self._drawdown_disabled_logged = True
            return False

        outcomes = self.history_store.load_recent_outcomes(
            ticker,
            n=getattr(cfg, "DRAWDOWN_LOOKBACK"),
        )
        if not outcomes:
            return False

        failures = [not o["success"] for o in outcomes]
        return all(failures)

    def _is_bear_market(self, current_date=None) -> bool:
        if current_date is None:
            current_date = date.today()
        if self.last_check_date == current_date:
            return self.is_bear_cache

        try:
            if getattr(self, "history_store", None) is not None and hasattr(
                self.history_store, "dm"
            ):
                dm = self.history_store.dm
            elif getattr(self, "dm", None) is not None:
                dm = self.dm
            else:
                dm = self._create_data_repository()
        except Exception:
            dm = self._create_data_repository()

        is_bear = False
        try:
            is_bear = dm.get_market_regime_is_bear(ref_date=current_date)
        except Exception:
            is_bear = False

        self.last_check_date = current_date
        self.is_bear_cache = is_bear
        return is_bear

    def _create_data_repository(self):
        try:
            return DataManagerRepository(settings=self._get_settings())
        except TypeError:
            return DataManagerRepository()

    def _get_market_volatility_index(self):
        return get_market_volatility_index()

    def _get_settings(self):
        return self.settings or build_settings()
