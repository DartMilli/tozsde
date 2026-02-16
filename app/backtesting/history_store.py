import json
from datetime import datetime
from typing import Dict

from app.data_access.data_manager import DataManager

"""
❗ HistoryStore NEM helyettesíti a DB-t

nincs index
nincs UPSERT
nincs napi lekérdezés
nem UI-ra való

❗ recommendations DB NEM audit log

csak utolsó döntés
overwrite-olható
kevés metaadat
"""


class HistoryStore:
    """
    P4.5 – Decision History Store

    Responsibility:
        - Persist finalized daily decisions
        - Append-only storage
        - NO business logic
        - NO calculations
        - NO interpretation

    Used by:
        - recommender / main pipeline
        - later: reliability scoring, audits, UI
    """

    def __init__(self):
        self.dm = DataManager()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def save_decision(
        self,
        payload: Dict,
        decision: Dict,
        explanation: Dict,
        audit: Dict,
        model_votes: list = None,
        safety_overrides: Dict = None,
        position_sizing: Dict = None,
        decision_source: str = None,
        model_id=None,
        timestamp=None,
        as_of_date=None,
        execution_price=None,
        model_version=None,
        features_hash=None,
        reliability_score=None,
    ) -> None:
        """
        Persist one finalized decision.

        Parameters
        ----------
        payload : Dict
            Raw ensemble payload (votes, wf scores, metadata)
        decision : Dict
            Final decision output from decision_builder
        explanation : Dict
            Human-readable explanation (HU / EN)
        """
        resolved_as_of = as_of_date or payload.get("as_of_date")
        resolved_execution_price = (
            execution_price
            if execution_price is not None
            else payload.get("execution_price")
        )
        if resolved_execution_price is None:
            resolved_execution_price = payload.get("latest_price")

        resolved_reliability = reliability_score
        if resolved_reliability is None:
            resolved_reliability = decision.get("reliability_score")
        if resolved_reliability is None:
            resolved_reliability = payload.get("reliability_score")

        return self.dm.save_history_record(
            ticker=payload.get("ticker"),
            model_id=model_id,
            model_version=model_version or payload.get("model_version"),
            action_code=decision.get("action_code"),
            label=decision.get("action"),
            confidence=decision.get("confidence"),
            wf_score=decision.get("wf_score"),
            reliability_score=resolved_reliability,
            execution_price=resolved_execution_price,
            features_hash=features_hash or payload.get("features_hash"),
            d_blob=json.dumps(decision, default=str),
            a_blob=json.dumps(audit, default=str),
            explanation_json=json.dumps(explanation, default=str),
            model_votes_json=json.dumps(model_votes or [], default=str),
            safety_overrides_json=json.dumps(safety_overrides or {}, default=str),
            position_sizing_json=json.dumps(position_sizing or {}, default=str),
            decision_source=decision_source,
            timestamp=timestamp or payload.get("timestamp"),
            as_of_date=resolved_as_of,
        )
        # record = {
        #    "timestamp": datetime.utcnow().isoformat(),
        #    "ticker": payload["ticker"],
        #    "decision": {
        #        "action": decision["action"],
        #        "action_code": decision["action_code"],
        #        "strength": decision["strength"],
        #        "confidence": decision["confidence"],
        #        "wf_score": decision["wf_score"],
        #        "ensemble_quality": decision["ensemble_quality"],
        #        "quality_score": decision["quality_score"],
        #        "no_trade": decision.get("no_trade", False),
        #        "no_trade_reason": decision.get("no_trade_reason"),
        #        "original_action": decision.get("original_action"),
        #    },
        #    "ensemble": {
        #        "quality": payload.get("ensemble_quality"),
        #        "model_votes": payload.get("model_votes", []),
        #        "wf_scores": payload.get("wf_scores", []),
        #    },
        #    "explanation": explanation,
        #    "audit": audit,
        # }
        #
        # self._append_to_monthly_file(record)

    def iter_records(self, ticker: str):
        """Yield decision_history rows (DAO-backed; no raw SQL here)."""
        for rec in self.dm.fetch_history_records_by_ticker(ticker):
            yield rec

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def load_range(self, ticker: str, start, end) -> list:
        """Returns list of decisions between [start, end] inclusive."""
        start_dt = start.date() if hasattr(start, "date") else start
        end_dt = end.date() if hasattr(end, "date") else end

        rows = self.dm.fetch_history_range(
            ticker=ticker,
            start_iso=(
                start_dt.isoformat()
                if hasattr(start_dt, "isoformat")
                else str(start_dt)
            ),
            end_iso=end_dt.isoformat() if hasattr(end_dt, "isoformat") else str(end_dt),
        )
        out = []
        for ts, action_label, decision_blob, audit_blob in rows:
            d = datetime.fromisoformat(ts).date()
            out.append(
                {
                    "date": d.isoformat(),
                    "action_code": (
                        json.loads(decision_blob) if decision_blob else {}
                    ).get("action_code", 0),
                    "decision": json.loads(decision_blob) if decision_blob else {},
                }
            )
        return out

    def load_recent_outcomes(self, ticker: str, n: int = 3) -> list[dict]:
        """DAO-backed stub; returns [] until an outcomes table is introduced."""
        return self.dm.fetch_recent_outcomes(ticker, n=n)
