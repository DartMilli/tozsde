from datetime import datetime
from app.backtesting.history_store import HistoryStore
from app.data_access.data_loader import load_data
from app.backtesting.decision_replay import replay_decision
from app.backtesting.reward_engine import realize_reward


def run_replay_for_ticker(
    ticker: str,
    horizon_days: int = 5,
):
    history = HistoryStore()

    # 1️⃣ betöltjük az ÖSSZES rekordot (append-only log)
    records = list(history.iter_records(ticker=ticker))
    if not records:
        return []

    # 2️⃣ árfolyam adat – legkorábbi decision-től
    start_ts = min(datetime.fromisoformat(r["timestamp"]) for r in records)
    df = load_data(
        ticker,
        start=start_ts.strftime("%Y-%m-%d"),
    )

    results = []

    for record in records:
        decision = record.get("decision", {})
        action_code = decision.get("action_code", 0)

        replay_input = {
            "timestamp": record["timestamp"],
            "action": action_code,
        }

        outcome = replay_decision(
            decision_event=replay_input,
            df=df,
            horizon_days=horizon_days,
        )

        reward = None

        if outcome.get("status") == "OK":
            reward = realize_reward(
                raw_return=outcome["raw_return"],
                decision=record.get("decision", {}),
                audit=record.get("audit", {}),
            )

        results.append(
            {
                "ticker": ticker,
                "timestamp": record["timestamp"],
                "action_code": action_code,
                "raw_return": outcome.get("raw_return"),
                "reward": reward,
                "confidence": record["decision"].get("confidence"),
                "decision_level": record["audit"].get("decision_level"),
            }
        )

    return results
