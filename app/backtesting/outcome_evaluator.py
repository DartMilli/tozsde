import json
import pandas as pd
from datetime import datetime, timedelta
from app.infrastructure.repositories import DataManagerRepository


class OutcomeEvaluator:
    """
    P8 modul: Vegignezi a multbeli donteseket, es kiszamolja,
    hogy nyeresegesek voltak-e (Realized PnL).
    Ezzel valosul meg a visszacsatolas (Feedback Loop).
    """

    def __init__(self, outcomes_repo=None, settings=None):
        self.dm = outcomes_repo
        if self.dm is None:
            self.dm = DataManagerRepository(settings=settings)

    def evaluate_past_decisions(self, lookback_days=30, hold_period=10):
        """
        Frissiti a history-t az eredmenyekkel.
        Csak a VETELI (action_code=1) donteseket vizsgalja.
        """
        print("Multbeli dontesek kiertekelese...")

        # 1. Lekerjuk az elmult idoszak donteseit
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Mivel a DataManager.get_history_range tickerenkent ker,
        # itt egy nyers SQL hatekonyabb lenne az osszes ticker-re.
        # De tartsuk be a retegeket: iteraljunk a configban levo tickereken (vagy kerjunk egy ujat a DM-tol).
        # Egyszerusites: Feltetelezzuk, hogy van egy listank, vagy bovitjuk a DM-et.

        # Javasolt bovites a DataManager-ben (lasd Lepes C), itt most feltetelezem:
        pending_decisions = self.dm.get_unevaluated_buy_decisions(limit=100)

        for row in pending_decisions:
            row_id, timestamp_str, ticker, audit_blob_str = row

            pnl = self._calculate_trade_result(
                ticker,
                timestamp_str,
                hold_period=hold_period,
            )

            if pnl is not None:
                outcome = {
                    "pnl_pct": round(pnl, 4),
                    "evaluated_at": datetime.now().isoformat(),
                    "exit_reason": "FIXED_HOLD",
                    "horizon_days": hold_period,
                }

                self.dm.save_outcome(
                    decision_id=row_id,
                    ticker=ticker,
                    decision_timestamp=timestamp_str,
                    pnl_pct=outcome["pnl_pct"],
                    success=outcome["pnl_pct"] > 0,
                    future_return=outcome["pnl_pct"],
                    exit_reason=outcome["exit_reason"],
                    horizon_days=outcome["horizon_days"],
                    outcome_json=json.dumps(outcome),
                )
                print(f" -> {ticker} ({timestamp_str}) eredmeny: {pnl*100:.2f}%")

    def _calculate_trade_result(self, ticker, entry_date_str, hold_period=10):
        """
        Megnezi, mi tortent az arfolyammal a belepes utan X nappal.
        Egyszerusitett logika: Fix 10 napos tartas eredmenye.
        """
        entry_date = pd.Timestamp(entry_date_str)
        # Par nappal kesobbi adatok kellenek
        check_end_date = entry_date + timedelta(days=hold_period + 5)

        df = self.dm.load_ohlcv(ticker)  # Ez cache-elt adatokat hasznal

        # Szures a relevans idoszakra
        df_future = df[df.index > entry_date]

        if df_future.empty:
            return None  # Meg nincs jovobeli adat (mai dontes)

        # Belepo ar (a dontes masnapjanak nyitoja vagy aznapi close - egyszerusitsunk close-ra)
        try:
            # Megkeressuk a pontos entry datumhoz legkozelebbi arat
            entry_idx = df.index.get_indexer([entry_date], method="nearest")[0]
            entry_price = df.iloc[entry_idx]["Close"]

            # Kilepo ar (hold_period nappal kesobb)
            exit_idx = entry_idx + hold_period
            if exit_idx >= len(df):
                exit_idx = len(df) - 1  # Ami van adat

            # Ha tul kozel van meg, nem ertekeljuk
            if (df.index[exit_idx] - df.index[entry_idx]).days < 3:
                return None

            exit_price = df.iloc[exit_idx]["Close"]

            return (exit_price - entry_price) / entry_price

        except Exception as e:
            # Hiba eseten (pl. hianyzo adat) skip
            return None
