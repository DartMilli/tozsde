from typing import Dict

from app.infrastructure.repositories import DataManagerRepository


def compute_drawdown_summary(ticker: str, data_manager=None, settings=None) -> Dict:
    dm = data_manager
    if dm is None:
        try:
            dm = DataManagerRepository(settings=settings)
        except TypeError:
            dm = DataManagerRepository()

    query = """
        SELECT decision_timestamp, pnl_pct
        FROM outcomes
        WHERE ticker = ?
        ORDER BY decision_timestamp ASC
    """
    with dm.connection() as conn:
        rows = conn.execute(query, (ticker,)).fetchall()

    if not rows:
        return {"status": "no_data", "rows": 0}

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    for _, pnl_pct in rows:
        pnl_pct = float(pnl_pct or 0.0)
        equity *= 1.0 + pnl_pct
        if equity > peak:
            peak = equity
        drawdown = (equity - peak) / peak if peak else 0.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    current_drawdown = (equity - peak) / peak if peak else 0.0

    return {
        "status": "ok",
        "rows": len(rows),
        "max_drawdown": round(abs(max_drawdown), 6),
        "current_drawdown": round(abs(current_drawdown), 6),
    }


def compute_loss_streak(ticker: str, data_manager=None, settings=None) -> Dict:
    dm = data_manager
    if dm is None:
        try:
            dm = DataManagerRepository(settings=settings)
        except TypeError:
            dm = DataManagerRepository()

    query = """
        SELECT decision_timestamp, success
        FROM outcomes
        WHERE ticker = ?
        ORDER BY decision_timestamp DESC
    """
    with dm.connection() as conn:
        rows = conn.execute(query, (ticker,)).fetchall()

    if not rows:
        return {"status": "no_data", "rows": 0, "loss_streak": 0}

    streak = 0
    for _, success in rows:
        if success is None:
            break
        if int(success) == 0:
            streak += 1
        else:
            break

    return {"status": "ok", "rows": len(rows), "loss_streak": streak}
