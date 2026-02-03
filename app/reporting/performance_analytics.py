"""
Performance Analytics - Comprehensive trading performance analysis module.

This module provides detailed performance metrics, risk analysis, and portfolio
statistics for trading strategies. Includes return metrics, drawdown analysis,
risk-adjusted returns, and rolling window performance tracking.

Classes:
    PerformanceMetrics: Dataclass for performance results
    DrawdownAnalysis: Dataclass for drawdown statistics
    PerformanceAnalytics: Main analytics engine
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
import sqlite3
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    period_start: datetime
    period_end: datetime


@dataclass
class DrawdownAnalysis:
    """Drawdown analysis results."""
    max_drawdown: float
    max_drawdown_duration_days: int
    current_drawdown: float
    drawdown_start: Optional[datetime]
    drawdown_end: Optional[datetime]
    recovery_date: Optional[datetime]
    time_to_recovery_days: Optional[int]
    drawdowns: List[Tuple[datetime, datetime, float]]  # [(start, end, magnitude)]


@dataclass
class RollingMetrics:
    """Rolling window performance metrics."""
    window_size_days: int
    returns: List[float]
    volatilities: List[float]
    sharpe_ratios: List[float]
    dates: List[datetime]


class PerformanceAnalytics:
    """
    Comprehensive performance analytics engine.
    
    Calculates:
    - Return metrics (total, annualized, risk-adjusted)
    - Drawdown analysis (max, duration, recovery)
    - Win/loss statistics
    - Risk ratios (Sharpe, Sortino, Calmar)
    - Rolling window performance
    """
    
    def __init__(self, db_path: str = None, risk_free_rate: float = 0.02):
        """
        Initialize performance analytics.
        
        Args:
            db_path: Path to SQLite database with trade history
            risk_free_rate: Annual risk-free rate for ratio calculations (default 2%)
        """
        self.db_path = db_path
        self.risk_free_rate = risk_free_rate
    
    def calculate_performance_metrics(
        self,
        returns: List[float],
        dates: List[datetime],
        trades: Optional[List[Dict]] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: List of daily returns (as decimals, e.g., 0.01 = 1%)
            dates: List of dates corresponding to returns
            trades: Optional list of trade records
        
        Returns:
            PerformanceMetrics: Complete performance statistics
        """
        if not returns or not dates:
            raise ValueError("Returns and dates cannot be empty")
        
        if len(returns) != len(dates):
            raise ValueError("Returns and dates must have same length")
        
        # Basic return metrics
        total_return = self._calculate_total_return(returns)
        annualized_return = self._calculate_annualized_return(returns, dates)
        volatility = self._calculate_volatility(returns, dates)
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns, dates)
        sortino_ratio = self._calculate_sortino_ratio(returns, dates)
        
        # Drawdown metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Trade statistics
        if trades:
            win_rate, profit_factor, total_trades, winning_trades, losing_trades = \
                self._calculate_trade_statistics(trades)
            avg_win, avg_loss, best_trade, worst_trade = self._calculate_trade_extremes(trades)
        else:
            win_rate = profit_factor = 0.0
            total_trades = winning_trades = losing_trades = 0
            avg_win = avg_loss = best_trade = worst_trade = 0.0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            period_start=dates[0] if dates else datetime.now(),
            period_end=dates[-1] if dates else datetime.now()
        )
    
    def analyze_drawdowns(
        self,
        returns: List[float],
        dates: List[datetime]
    ) -> DrawdownAnalysis:
        """
        Analyze drawdown characteristics.
        
        Args:
            returns: List of returns
            dates: List of dates
        
        Returns:
            DrawdownAnalysis: Detailed drawdown statistics
        """
        if not returns or not dates:
            raise ValueError("Returns and dates cannot be empty")
        
        # Calculate cumulative returns for equity curve
        equity_curve = [1.0]
        for r in returns:
            equity_curve.append(equity_curve[-1] * (1 + r))
        
        # Find all drawdowns
        drawdowns = []
        peak = equity_curve[0]
        peak_date = dates[0]
        in_drawdown = False
        drawdown_start = None
        
        for i, (equity, date) in enumerate(zip(equity_curve[1:], dates), start=1):
            if equity > peak:
                # New high
                if in_drawdown:
                    # End of drawdown
                    dd_magnitude = (peak - equity_curve[i-1]) / peak
                    drawdowns.append((drawdown_start, dates[i-1], dd_magnitude))
                    in_drawdown = False
                peak = equity
                peak_date = date
            else:
                # In drawdown
                if not in_drawdown:
                    drawdown_start = peak_date
                    in_drawdown = True
        
        # Close final drawdown if still in one
        if in_drawdown:
            dd_magnitude = (peak - equity_curve[-1]) / peak
            drawdowns.append((drawdown_start, dates[-1], dd_magnitude))
        
        # Find max drawdown
        if drawdowns:
            max_dd_tuple = max(drawdowns, key=lambda x: x[2])
            max_drawdown = max_dd_tuple[2]
            max_dd_start = max_dd_tuple[0]
            max_dd_end = max_dd_tuple[1]
            
            # Calculate duration
            duration = (max_dd_end - max_dd_start).days
            
            # Find recovery date
            recovery_date = None
            time_to_recovery = None
            max_dd_idx = dates.index(max_dd_end)
            peak_value = equity_curve[dates.index(max_dd_start)]
            
            for i in range(max_dd_idx + 1, len(equity_curve)):
                if equity_curve[i] >= peak_value:
                    recovery_date = dates[i-1] if i-1 < len(dates) else dates[-1]
                    time_to_recovery = (recovery_date - max_dd_end).days
                    break
        else:
            max_drawdown = 0.0
            max_dd_start = None
            max_dd_end = None
            duration = 0
            recovery_date = None
            time_to_recovery = None
        
        # Current drawdown
        current_peak = max(equity_curve)
        current_equity = equity_curve[-1]
        current_drawdown = (current_peak - current_equity) / current_peak if current_peak > 0 else 0.0
        
        return DrawdownAnalysis(
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=duration,
            current_drawdown=current_drawdown,
            drawdown_start=max_dd_start,
            drawdown_end=max_dd_end,
            recovery_date=recovery_date,
            time_to_recovery_days=time_to_recovery,
            drawdowns=drawdowns
        )
    
    def calculate_rolling_metrics(
        self,
        returns: List[float],
        dates: List[datetime],
        window_days: int = 30
    ) -> RollingMetrics:
        """
        Calculate rolling window performance metrics.
        
        Args:
            returns: List of returns
            dates: List of dates
            window_days: Rolling window size in days
        
        Returns:
            RollingMetrics: Rolling performance statistics
        """
        if len(returns) < window_days:
            logger.warning(f"Not enough data for {window_days}-day rolling window")
            return RollingMetrics(
                window_size_days=window_days,
                returns=[],
                volatilities=[],
                sharpe_ratios=[],
                dates=[]
            )
        
        rolling_returns = []
        rolling_vols = []
        rolling_sharpes = []
        rolling_dates = []
        
        for i in range(window_days, len(returns) + 1):
            window_returns = returns[i-window_days:i]
            window_dates = dates[i-window_days:i]
            
            # Calculate metrics for window
            period_return = self._calculate_total_return(window_returns)
            volatility = self._calculate_volatility(window_returns, window_dates)
            sharpe = self._calculate_sharpe_ratio(window_returns, window_dates)
            
            rolling_returns.append(period_return)
            rolling_vols.append(volatility)
            rolling_sharpes.append(sharpe)
            rolling_dates.append(dates[i-1])
        
        return RollingMetrics(
            window_size_days=window_days,
            returns=rolling_returns,
            volatilities=rolling_vols,
            sharpe_ratios=rolling_sharpes,
            dates=rolling_dates
        )
    
    def _calculate_total_return(self, returns: List[float]) -> float:
        """Calculate cumulative return."""
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r)
        return cumulative - 1.0
    
    def _calculate_annualized_return(self, returns: List[float], dates: List[datetime]) -> float:
        """Calculate annualized return."""
        if len(dates) < 2:
            return 0.0
        
        total_return = self._calculate_total_return(returns)
        days = (dates[-1] - dates[0]).days
        
        if days <= 0:
            return 0.0
        
        years = days / 365.25
        annualized = (1 + total_return) ** (1 / years) - 1
        
        return annualized
    
    def _calculate_volatility(self, returns: List[float], dates: List[datetime]) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)
        
        # Annualize (assume daily returns)
        annualized_vol = std_dev * math.sqrt(252)
        
        return annualized_vol
    
    def _calculate_sharpe_ratio(self, returns: List[float], dates: List[datetime]) -> float:
        """Calculate Sharpe ratio."""
        annualized_return = self._calculate_annualized_return(returns, dates)
        volatility = self._calculate_volatility(returns, dates)
        
        if volatility == 0:
            return 0.0
        
        sharpe = (annualized_return - self.risk_free_rate) / volatility
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: List[float], dates: List[datetime]) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        annualized_return = self._calculate_annualized_return(returns, dates)
        
        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf') if annualized_return > self.risk_free_rate else 0.0
        
        mean_negative = sum(negative_returns) / len(negative_returns)
        downside_variance = sum((r - mean_negative) ** 2 for r in negative_returns) / len(negative_returns)
        downside_std = math.sqrt(downside_variance)
        
        # Annualize
        annualized_downside_std = downside_std * math.sqrt(252)
        
        if annualized_downside_std == 0:
            return 0.0
        
        sortino = (annualized_return - self.risk_free_rate) / annualized_downside_std
        return sortino
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        equity_curve = [1.0]
        for r in returns:
            equity_curve.append(equity_curve[-1] * (1 + r))
        
        max_dd = 0.0
        peak = equity_curve[0]
        
        for equity in equity_curve[1:]:
            if equity > peak:
                peak = equity
            else:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Tuple[float, float, int, int, int]:
        """Calculate win rate and profit factor from trades."""
        if not trades:
            return 0.0, 0.0, 0, 0, 0
        
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        total = len(trades)
        winning = len(wins)
        losing = len(losses)
        
        win_rate = winning / total if total > 0 else 0.0
        
        # Profit factor = gross profit / gross loss
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return win_rate, profit_factor, total, winning, losing
    
    def _calculate_trade_extremes(self, trades: List[Dict]) -> Tuple[float, float, float, float]:
        """Calculate average win/loss and best/worst trade."""
        if not trades:
            return 0.0, 0.0, 0.0, 0.0
        
        wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
        all_pnls = [t.get('pnl', 0) for t in trades]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        best_trade = max(all_pnls) if all_pnls else 0.0
        worst_trade = min(all_pnls) if all_pnls else 0.0
        
        return avg_win, avg_loss, best_trade, worst_trade
    
    def load_returns_from_db(
        self,
        days_back: int = 90
    ) -> Tuple[List[float], List[datetime]]:
        """
        Load returns data from database.
        
        Args:
            days_back: Number of days to load
        
        Returns:
            Tuple of (returns list, dates list)
        """
        if not self.db_path:
            return [], []
        
        returns = []
        dates = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Assuming a portfolio_equity table exists
            cursor.execute("""
                SELECT date, equity_value
                FROM portfolio_equity
                WHERE date >= ?
                ORDER BY date ASC
            """, (cutoff_date,))
            
            rows = cursor.fetchall()
            
            if len(rows) >= 2:
                for i in range(1, len(rows)):
                    prev_equity = rows[i-1][1]
                    curr_equity = rows[i][1]
                    
                    if prev_equity > 0:
                        daily_return = (curr_equity - prev_equity) / prev_equity
                        returns.append(daily_return)
                        date_value = rows[i][0]
                        try:
                            parsed_date = datetime.fromisoformat(date_value)
                        except AttributeError:
                            parsed_date = datetime.strptime(date_value, "%Y-%m-%d")
                        dates.append(parsed_date)
            
            conn.close()
        except (sqlite3.Error, Exception) as e:
            logger.error(f"Error loading returns from database: {e}")
        
        return returns, dates
