"""
Tests for NoTradeDecisionLogger module.
"""

import pytest
from datetime import datetime, timedelta
from app.infrastructure.decision_logger import (
    NoTradeDecisionLogger,
    NoTradeDecision,
    NoTradeReason
)


@pytest.fixture
def logger():
    """Create a test logger (without database persistence)."""
    return NoTradeDecisionLogger(db_path=None)


class TestNoTradeReasonEnum:
    """Test NoTradeReason enum."""
    
    def test_all_reasons_exist(self):
        """Test all expected no-trade reasons are defined."""
        expected_reasons = [
            NoTradeReason.LOW_CONFIDENCE,
            NoTradeReason.HIGH_CORRELATION,
            NoTradeReason.INSUFFICIENT_CAPITAL,
            NoTradeReason.POSITION_LIMIT_REACHED,
            NoTradeReason.MARKET_REGIMEN,
            NoTradeReason.DIVERSIFICATION_CONSTRAINT,
            NoTradeReason.RISK_THRESHOLD_EXCEEDED,
            NoTradeReason.DECISION_DISABLED,
            NoTradeReason.OTHER
        ]
        
        for reason in expected_reasons:
            assert reason in NoTradeReason
    
    def test_reason_values_are_strings(self):
        """Test reason values are string enums."""
        for reason in NoTradeReason:
            assert isinstance(reason.value, str)


class TestNoTradeDecisionCreation:
    """Test creating no-trade decisions."""
    
    def test_create_simple_decision(self):
        """Test creating a simple no-trade decision."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="AAPL",
            strategy="momentum",
            reason=NoTradeReason.LOW_CONFIDENCE,
            confidence_score=0.35
        )
        
        assert decision.ticker == "AAPL"
        assert decision.strategy == "momentum"
        assert decision.reason == NoTradeReason.LOW_CONFIDENCE
        assert decision.confidence_score == 0.35
    
    def test_create_detailed_decision(self):
        """Test creating a detailed no-trade decision with all fields."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="AAPL",
            strategy="momentum",
            reason=NoTradeReason.HIGH_CORRELATION,
            confidence_score=0.65,
            market_regime="VOLATILE",
            correlation_value=0.85,
            available_capital=25000.0,
            portfolio_risk=0.15,
            details="Correlation with MSFT too high"
        )
        
        assert decision.correlation_value == 0.85
        assert decision.available_capital == 25000.0
        assert decision.portfolio_risk == 0.15
        assert decision.details == "Correlation with MSFT too high"


class TestLoggingWithoutDatabase:
    """Test logging without database persistence."""
    
    def test_log_without_database(self, logger):
        """Test logging when no database configured."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="AAPL",
            strategy="momentum",
            reason=NoTradeReason.LOW_CONFIDENCE
        )
        
        result = logger.log_no_trade_decision(decision)
        
        assert result is False
    
    def test_get_decisions_without_database(self, logger):
        """Test retrieving decisions when no database configured."""
        decisions = logger.get_no_trade_decisions()
        
        assert decisions == []
    
    def test_get_analysis_without_database(self, logger):
        """Test getting analysis when no database configured."""
        analysis = logger.get_no_trade_analysis()
        
        assert analysis == {}


class TestSimplifiedLoggingInterface:
    """Test simplified logging interface."""
    
    def test_log_no_trade_simple(self, logger):
        """Test simplified logging interface."""
        result = logger.log_no_trade_simple(
            ticker="AAPL",
            strategy="momentum",
            reason=NoTradeReason.LOW_CONFIDENCE,
            confidence_score=0.35
        )
        
        assert result is False  # No database, so False


class TestNoTradeReasonCategories:
    """Test different no-trade reason categories."""
    
    def test_low_confidence_reason(self):
        """Test LOW_CONFIDENCE reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE,
            confidence_score=0.3
        )
        
        assert decision.reason == NoTradeReason.LOW_CONFIDENCE
    
    def test_high_correlation_reason(self):
        """Test HIGH_CORRELATION reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.HIGH_CORRELATION,
            correlation_value=0.92
        )
        
        assert decision.reason == NoTradeReason.HIGH_CORRELATION
        assert decision.correlation_value == 0.92
    
    def test_insufficient_capital_reason(self):
        """Test INSUFFICIENT_CAPITAL reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.INSUFFICIENT_CAPITAL,
            available_capital=500.0
        )
        
        assert decision.reason == NoTradeReason.INSUFFICIENT_CAPITAL
    
    def test_position_limit_reason(self):
        """Test POSITION_LIMIT_REACHED reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.POSITION_LIMIT_REACHED
        )
        
        assert decision.reason == NoTradeReason.POSITION_LIMIT_REACHED
    
    def test_market_regime_reason(self):
        """Test MARKET_REGIME reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.MARKET_REGIMEN,
            market_regime="VOLATILE"
        )
        
        assert decision.reason == NoTradeReason.MARKET_REGIMEN
        assert decision.market_regime == "VOLATILE"
    
    def test_diversification_reason(self):
        """Test DIVERSIFICATION_CONSTRAINT reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.DIVERSIFICATION_CONSTRAINT,
            details="Position would exceed 10% portfolio weight"
        )
        
        assert decision.reason == NoTradeReason.DIVERSIFICATION_CONSTRAINT
    
    def test_risk_threshold_reason(self):
        """Test RISK_THRESHOLD_EXCEEDED reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.RISK_THRESHOLD_EXCEEDED,
            portfolio_risk=0.28
        )
        
        assert decision.reason == NoTradeReason.RISK_THRESHOLD_EXCEEDED
        assert decision.portfolio_risk == 0.28
    
    def test_decision_disabled_reason(self):
        """Test DECISION_DISABLED reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.DECISION_DISABLED
        )
        
        assert decision.reason == NoTradeReason.DECISION_DISABLED
    
    def test_other_reason(self):
        """Test OTHER reason."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.OTHER,
            details="Unspecified reason"
        )
        
        assert decision.reason == NoTradeReason.OTHER


class TestTimestamps:
    """Test timestamp handling."""
    
    def test_decision_timestamp_current(self):
        """Test decision gets current timestamp."""
        now = datetime.now()
        decision = NoTradeDecision(
            timestamp=now,
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE
        )
        
        assert decision.timestamp == now
    
    def test_decision_timestamp_past(self):
        """Test decision can have past timestamp."""
        past = datetime.now() - timedelta(days=1)
        decision = NoTradeDecision(
            timestamp=past,
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE
        )
        
        assert decision.timestamp == past
    
    def test_decision_timestamp_future(self):
        """Test decision can have future timestamp."""
        future = datetime.now() + timedelta(days=1)
        decision = NoTradeDecision(
            timestamp=future,
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE
        )
        
        assert decision.timestamp == future


class TestDecisionDetails:
    """Test decision detail fields."""
    
    def test_optional_fields_can_be_none(self):
        """Test all optional fields can be None."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE
        )
        
        assert decision.confidence_score is None
        assert decision.market_regime is None
        assert decision.correlation_value is None
        assert decision.available_capital is None
        assert decision.portfolio_risk is None
        assert decision.details is None
    
    def test_all_fields_populated(self):
        """Test decision with all fields populated."""
        now = datetime.now()
        decision = NoTradeDecision(
            timestamp=now,
            ticker="AAPL",
            strategy="momentum",
            reason=NoTradeReason.HIGH_CORRELATION,
            confidence_score=0.72,
            market_regime="BULL",
            correlation_value=0.88,
            available_capital=15000.0,
            portfolio_risk=0.18,
            details="Correlated with existing MSFT position"
        )
        
        assert decision.timestamp == now
        assert decision.ticker == "AAPL"
        assert decision.strategy == "momentum"
        assert decision.reason == NoTradeReason.HIGH_CORRELATION
        assert decision.confidence_score == 0.72
        assert decision.market_regime == "BULL"
        assert decision.correlation_value == 0.88
        assert decision.available_capital == 15000.0
        assert decision.portfolio_risk == 0.18
        assert decision.details == "Correlated with existing MSFT position"


class TestExportFunctionality:
    """Test export functionality."""
    
    def test_export_without_database(self, logger):
        """Test export when no database configured."""
        result = logger.export_no_trade_decisions("/tmp/test.json")
        
        assert result is False


class TestCleanupFunctionality:
    """Test cleanup functionality."""
    
    def test_clear_old_decisions_without_database(self, logger):
        """Test clear old decisions when no database configured."""
        deleted = logger.clear_old_decisions(days_to_keep=30)
        
        assert deleted == 0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_ticker(self):
        """Test decision with empty ticker."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE
        )
        
        assert decision.ticker == ""
    
    def test_empty_strategy(self):
        """Test decision with empty strategy."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="",
            reason=NoTradeReason.LOW_CONFIDENCE
        )
        
        assert decision.strategy == ""
    
    def test_very_long_details(self):
        """Test decision with very long details string."""
        long_details = "X" * 10000
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE,
            details=long_details
        )
        
        assert decision.details == long_details
    
    def test_zero_values(self):
        """Test decision with zero numeric values."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE,
            confidence_score=0.0,
            correlation_value=0.0,
            available_capital=0.0,
            portfolio_risk=0.0
        )
        
        assert decision.confidence_score == 0.0
        assert decision.correlation_value == 0.0
        assert decision.available_capital == 0.0
        assert decision.portfolio_risk == 0.0
    
    def test_max_numeric_values(self):
        """Test decision with maximum numeric values."""
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="TEST",
            strategy="test",
            reason=NoTradeReason.LOW_CONFIDENCE,
            confidence_score=1.0,
            correlation_value=1.0,
            available_capital=1000000.0,
            portfolio_risk=1.0
        )
        
        assert decision.confidence_score == 1.0
        assert decision.correlation_value == 1.0
        assert decision.available_capital == 1000000.0
        assert decision.portfolio_risk == 1.0
