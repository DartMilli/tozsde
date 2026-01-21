from datetime import date, timedelta
from app.config.config import Config
from app.data_access.data_loader import get_market_volatility_index
from app.data_access.data_manager import DataManager


class SafetyRuleEngine:
    """
    Applies global safety rules on top of a decision.
    """

    def __init__(self, history_store):
        self.history_store = history_store
        self.last_check_date = None
        self.is_bear_cache = False

    def apply(
        self,
        ticker: str,
        decision: dict,
        today: date,
    ) -> dict:
        """
        Returns possibly modified decision with safety overrides.
        """
        reasons = []

        hold_label = Config.ACTION_LABELS[Config.LANG][0]

        def _force_hold(reason_code: str, reason_text: str):
            decision["action_code"] = 0
            decision["action"] = hold_label
            decision["no_trade"] = True
            decision["no_trade_reason"] = reason_code
            decision["strength"] = "NO_TRADE"
            reasons.append(reason_text)

        # 1) Cooldown
        if self._in_cooldown(ticker, today):
            _force_hold("COOLDOWN", "cooldown active")

        # 2) Drawdown guard
        if self._recent_drawdown(ticker):
            _force_hold("DRAWDOWN_GUARD", "recent drawdown")

        # 3) Ensemble quality
        if decision.get("ensemble_quality") == "CHAOTIC":
            _force_hold("CHAOTIC_ENSEMBLE", "chaotic ensemble")

        # --- ÚJ: MARKET REGIME CHECK (P8) ---
        # Ha vételi jelzés van, ellenőrizzük a globális trendet
        if decision["action_code"] == 1:
            if self._is_bear_market():
                # Medvepiacon vagyunk. Szigorítás!
                # Csak akkor engedjük a vételt, ha a szignál 'STRONG'
                if decision["strength"] != "STRONG":
                    _force_hold(
                        "BEAR_MARKET_FILTER",
                        "Global market is bearish (SPY < SMA200). Only STRONG signals allowed.",
                    )
                else:
                    # Ha erős a jel, akkor is feljegyezzük figyelmeztetésnek
                    decision.setdefault("warnings", []).append(
                        "Trading against global bear trend."
                    )

        # 4) MACRO FEAR GUARD (ÚJ)
        # Lekérjük a VIX-et. Élesben ezt cache-elni kéne, de napi 1 futásnál belefér.
        current_vix = get_market_volatility_index()

        if current_vix and current_vix > Config.MAX_VIX_THRESHOLD:
            # VIX > 30 extrém félelmet jelent (pl. Covid, háború)
            # Ilyenkor csak ELADNI vagy TARTANI szabad, venni tilos.
            if decision["action_code"] == 1:  # BUY
                _force_hold(
                    "HIGH_VIX_PANIC", f"Market fear is extreme (VIX={current_vix:.1f})"
                )

        decision["safety_override"] = bool(reasons)
        decision.setdefault("reasons", []).extend(reasons)
        return decision

    # ---------------- internal ----------------

    def _in_cooldown(self, ticker: str, today: date) -> bool:
        lookback_days = Config.COOLDOWN_DAYS
        decisions = self.history_store.load_range(
            ticker,
            start=today - timedelta(days=lookback_days),
            end=today,
        )

        trade_actions = [d["action_code"] for d in decisions if d["action_code"] != 0]
        return len(trade_actions) >= Config.COOLDOWN_MAX_TRADES

    def _recent_drawdown(self, ticker: str) -> bool:
        # simple version: last N outcomes all failed
        outcomes = self.history_store.load_recent_outcomes(
            ticker,
            n=Config.DRAWDOWN_LOOKBACK,
        )
        if not outcomes:
            return False

        failures = [not o["success"] for o in outcomes]
        return all(failures)

    def _is_bear_market(self, current_date) -> bool:  # Új paraméter!
        # Simple caching: Ha ugyanaz a nap, ne kérdezzük le újra az adatbázist
        if self.last_check_date == current_date:
            return self.is_bear_cache

        dm = DataManager()
        is_bear = dm.get_market_regime_is_bear(ref_date=current_date)

        self.last_check_date = current_date
        self.is_bear_cache = is_bear
        return is_bear
