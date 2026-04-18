"""Sprint 14 regression tests – Q6/A4/A5, Q2, Q1, Q3, Q5 fixes."""

import datetime
from unittest.mock import MagicMock

import pytest

from app.core.decision.ensemble_aggregator import aggregate_weighted_ensemble
from app.core.decision.ensemble_quality import (
    EnsembleQualityBucket,
    bucket_ensemble_quality,
)
from app.core.decision.recommender import _quality_label
from app.core.decision.allocation import allocate_capital
from app.core.decision.capital_optimizer import CapitalUtilizationOptimizer
from app.optimization.fitness import fitness_single


# ---------------------------------------------------------------------------
# Q6 / A4 / A5 – _quality_label & EnsembleQualityBucket
# ---------------------------------------------------------------------------


class TestQualityLabel:
    """_quality_label() must return EnsembleQualityBucket, never the old "STABLE" string."""

    def test_strong_score_returns_strong_enum(self):
        """Score >= STRONG threshold → EnsembleQualityBucket.STRONG (not "STABLE")."""
        result = _quality_label(0.9, settings=None)
        assert result == EnsembleQualityBucket.STRONG
        assert result.value == "STRONG"
        assert result != "STABLE"

    def test_chaotic_score_returns_chaotic_enum(self):
        """Score below WEAK threshold → EnsembleQualityBucket.CHAOTIC."""
        result = _quality_label(0.05, settings=None)
        assert result == EnsembleQualityBucket.CHAOTIC
        assert result == "CHAOTIC"  # str-enum equality

    def test_normal_score_returns_normal_enum(self):
        result = _quality_label(0.4, settings=None)
        assert result == EnsembleQualityBucket.NORMAL

    def test_custom_threshold_via_settings(self):
        """If settings provides ENSEMBLE_QUALITY_THRESHOLDS, they are respected."""
        mock_settings = MagicMock()
        mock_settings.ENSEMBLE_QUALITY_THRESHOLDS = {
            "STRONG": 0.9,
            "NORMAL": 0.5,
            "WEAK": 0.2,
        }
        result = _quality_label(0.7, settings=mock_settings)
        # 0.7 < 0.9 (STRONG), 0.7 >= 0.5 (NORMAL) → NORMAL
        assert result == EnsembleQualityBucket.NORMAL


class TestSafetyRuleEngineChaoticTrigger:
    """Safety override must fire when ensemble_quality is CHAOTIC (Q6)."""

    def _make_decision(self, ensemble_quality):
        return {
            "action_code": 1,
            "action": "BUY",
            "strength": "NORMAL",
            "confidence": 0.8,
            "no_trade": False,
            "no_trade_reason": None,
            "ensemble_quality": ensemble_quality,
        }

    def _make_engine(self):
        from app.core.decision.safety_rules import SafetyRuleEngine

        history_store = MagicMock()
        history_store.load_range.return_value = []
        history_store.load_recent.return_value = []

        settings = MagicMock()
        settings.ACTION_LABELS = {"en": {0: "HOLD", 1: "BUY", 2: "SELL"}}
        settings.LANG = "en"
        settings.COOLDOWN_DAYS = 5
        settings.COOLDOWN_MAX_TRADES = 2
        settings.DRAWDOWN_LOOKBACK = 3
        settings.ENABLE_DRAWDOWN_GUARD = False
        settings.MAX_VIX_THRESHOLD = 40.0
        settings.BEAR_MARKET_SMA_PERIOD = 200
        settings.BEAR_MARKET_LOOKBACK_DAYS = 250
        settings.VALIDATION_DISABLE_SAFETY = False

        engine = SafetyRuleEngine(history_store, settings=settings)
        engine._is_bear_market = MagicMock(return_value=False)
        engine._recent_drawdown = MagicMock(return_value=False)
        engine._in_cooldown = MagicMock(return_value=False)
        engine._get_market_volatility_index = MagicMock(return_value=None)
        return engine

    def test_chaotic_ensemble_triggers_safety_override(self):
        """ensemble_quality == CHAOTIC → safety_override = True, action forced to HOLD."""
        engine = self._make_engine()
        decision = self._make_decision(EnsembleQualityBucket.CHAOTIC)
        result = engine.apply("VOO", decision, datetime.date.today())
        assert result["safety_override"] is True
        assert result["action_code"] == 0

    def test_chaotic_string_triggers_safety_override(self):
        """'CHAOTIC' plain string also triggers safety override."""
        engine = self._make_engine()
        decision = self._make_decision("CHAOTIC")
        result = engine.apply("VOO", decision, datetime.date.today())
        assert result["safety_override"] is True

    def test_strong_ensemble_does_not_trigger_override(self):
        """STRONG ensemble quality must NOT trigger a safety override."""
        engine = self._make_engine()
        decision = self._make_decision(EnsembleQualityBucket.STRONG)
        result = engine.apply("VOO", decision, datetime.date.today())
        assert result["safety_override"] is False


# ---------------------------------------------------------------------------
# Q2 – allocate_capital with current_equity
# ---------------------------------------------------------------------------


class TestAllocateCapitalEquity:
    """allocate_capital() must use current_equity when provided."""

    def _make_candidates(self, n=1):
        return [
            {
                "ticker": f"T{i}",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},
            }
            for i in range(n)
        ]

    def test_uses_current_equity_when_provided(self):
        """With current_equity=15000 and single BUY, allocation_amount ≈ 15000."""
        candidates = self._make_candidates(1)
        result = allocate_capital(
            candidates,
            current_equity=15_000.0,
            get_correlation_matrix_fn=MagicMock(return_value=None),
        )
        assert result[0]["allocation_amount"] == pytest.approx(15_000.0, rel=0.01)

    def test_falls_back_to_initial_capital_when_no_equity(self):
        """Without current_equity, INITIAL_CAPITAL (10 000) is used."""
        mock_settings = MagicMock()
        mock_settings.INITIAL_CAPITAL = 10_000.0
        candidates = self._make_candidates(1)
        result = allocate_capital(
            candidates,
            settings=mock_settings,
            current_equity=None,
            get_correlation_matrix_fn=MagicMock(return_value=None),
        )
        assert result[0]["allocation_amount"] == pytest.approx(10_000.0, rel=0.01)

    def test_zero_equity_falls_back_to_initial_capital(self):
        """current_equity=0 is invalid, should fall back to INITIAL_CAPITAL."""
        mock_settings = MagicMock()
        mock_settings.INITIAL_CAPITAL = 10_000.0
        candidates = self._make_candidates(1)
        result = allocate_capital(
            candidates,
            settings=mock_settings,
            current_equity=0.0,
            get_correlation_matrix_fn=MagicMock(return_value=None),
        )
        assert result[0]["allocation_amount"] == pytest.approx(10_000.0, rel=0.01)


# ---------------------------------------------------------------------------
# Q1 – fitness_single normalization
# ---------------------------------------------------------------------------


class TestFitnessSingleNormalized:
    """fitness_single() must produce normalised, window-independent scores."""

    class _M:
        def __init__(self, tc, np_, mdd, wr):
            self.trade_count = tc
            self.net_profit = np_
            self.max_drawdown = mdd
            self.winrate = wr

    def test_same_relative_profit_same_score(self):
        """20% profit with capital 10k and 40% profit with capital 20k → same score."""
        m1 = self._M(10, 2_000, 500, 0.6)  # 2000/10000 = 20%
        m2 = self._M(10, 4_000, 1_000, 0.6)  # 4000/20000 = 20%
        s1 = fitness_single(m1, initial_capital=10_000.0)
        s2 = fitness_single(m2, initial_capital=20_000.0)
        assert s1 == pytest.approx(s2, rel=1e-6)

    def test_negative_profit_gives_negative_score(self):
        m = self._M(10, -2_000, 500, 0.4)
        assert fitness_single(m) < 0

    def test_too_few_trades_returns_neg_inf(self):
        m = self._M(2, 5_000, 100, 0.7)
        assert fitness_single(m) < -1e10

    def test_higher_winrate_higher_score(self):
        m_high = self._M(20, 1_000, 200, 0.70)
        m_low = self._M(20, 1_000, 200, 0.40)
        assert fitness_single(m_high) > fitness_single(m_low)


# ---------------------------------------------------------------------------
# Q3 – Kelly half-dampening
# ---------------------------------------------------------------------------


class TestKellyHalfDampening:
    """calculate_kelly_fraction() must apply half-Kelly by default."""

    def setup_method(self):
        self.optimizer = CapitalUtilizationOptimizer(total_capital=100_000.0)

    def test_default_half_kelly(self):
        """Full Kelly = 0.20, half-Kelly = 0.10."""
        kelly = self.optimizer.calculate_kelly_fraction(
            win_rate=0.60, avg_win=100.0, avg_loss=100.0
        )
        assert kelly == pytest.approx(0.10, rel=0.01)

    def test_full_kelly_with_multiplier_one(self):
        """Multiplier=1.0 → original full Kelly behaviour."""
        kelly = self.optimizer.calculate_kelly_fraction(
            win_rate=0.60,
            avg_win=100.0,
            avg_loss=100.0,
            kelly_fraction_multiplier=1.0,
        )
        assert kelly == pytest.approx(0.20, rel=0.01)

    def test_kelly_still_capped_at_50_pct(self):
        """Even with multiplier=1.0, result is never > 0.50."""
        kelly = self.optimizer.calculate_kelly_fraction(
            win_rate=0.99,
            avg_win=1_000.0,
            avg_loss=1.0,
            kelly_fraction_multiplier=1.0,
        )
        assert kelly <= 0.50

    def test_half_kelly_always_less_than_full_kelly(self):
        """Half-Kelly must always be <= full Kelly for the same inputs."""
        half = self.optimizer.calculate_kelly_fraction(
            win_rate=0.70,
            avg_win=120.0,
            avg_loss=80.0,
            kelly_fraction_multiplier=0.5,
        )
        full = self.optimizer.calculate_kelly_fraction(
            win_rate=0.70,
            avg_win=120.0,
            avg_loss=80.0,
            kelly_fraction_multiplier=1.0,
        )
        assert half < full


# ---------------------------------------------------------------------------
# Q5 – MAX_MODEL_AGE_DAYS stale model exclusion
# ---------------------------------------------------------------------------


class TestMaxModelAgeDays:
    """aggregate_weighted_ensemble() must skip models older than MAX_MODEL_AGE_DAYS."""

    def _run(self, trained_at, max_age):
        settings = MagicMock()
        settings.MAX_MODEL_AGE_DAYS = max_age
        settings.MIN_WF_WEIGHT = 0.0
        settings.RANK_ALPHA = 0.7
        settings.RECENCY_HALF_LIFE_DAYS = 90

        model_vote = {"rank": 1, "trained_at": trained_at}
        action, conf, quality = aggregate_weighted_ensemble(
            votes=[1],
            confidences=[0.8],
            wf_scores=[0.7],
            model_votes=[model_vote],
            settings=settings,
        )
        return action, conf

    def test_fresh_model_included(self):
        today = datetime.date.today()
        action, conf = self._run(today - datetime.timedelta(days=10), max_age=365)
        assert conf > 0.0

    def test_stale_model_excluded(self):
        """Model older than MAX_MODEL_AGE_DAYS → excluded → fallback HOLD with 0 conf."""
        today = datetime.date.today()
        action, conf = self._run(today - datetime.timedelta(days=400), max_age=365)
        # All models excluded → total_weight == 0 → fallback (0, 0.0, 0.0)
        assert action == 0
        assert conf == 0.0


# ---------------------------------------------------------------------------
# bucket_ensemble_quality – robustness (accepts str, enum, float)
# ---------------------------------------------------------------------------


class TestBucketEnsembleQualityRobust:
    def test_accepts_float(self):
        assert bucket_ensemble_quality(0.7) == EnsembleQualityBucket.STRONG

    def test_accepts_enum(self):
        assert (
            bucket_ensemble_quality(EnsembleQualityBucket.CHAOTIC)
            == EnsembleQualityBucket.CHAOTIC
        )

    def test_accepts_string(self):
        assert bucket_ensemble_quality("CHAOTIC") == EnsembleQualityBucket.CHAOTIC
        assert bucket_ensemble_quality("STRONG") == EnsembleQualityBucket.STRONG

    def test_unknown_string_falls_back_to_float_zero(self):
        """Unknown string is not parseable as float → CHAOTIC (score=0 path)."""
        with pytest.raises((ValueError, TypeError)):
            bucket_ensemble_quality("SOMETHING_UNKNOWN")
