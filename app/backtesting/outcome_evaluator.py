import json
import pandas as pd
from datetime import datetime, timedelta
from app.data_access.data_manager import DataManager


class OutcomeEvaluator:
    """
    P8 modul: Végignézi a múltbeli döntéseket, és kiszámolja,
    hogy nyereségesek voltak-e (Realized PnL).
    Ezzel valósul meg a visszacsatolás (Feedback Loop).
    """

    def __init__(self):
        self.dm = DataManager()

    def evaluate_past_decisions(self, lookback_days=30, hold_period=10):
        """
        Frissíti a history-t az eredményekkel.
        Csak a VÉTELI (action_code=1) döntéseket vizsgálja.
        """
        print("Múltbeli döntések kiértékelése...")

        # 1. Lekérjük az elmúlt időszak döntéseit
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Mivel a DataManager.get_history_range tickerenként kér,
        # itt egy nyers SQL hatékonyabb lenne az összes ticker-re.
        # De tartsuk be a rétegeket: iteráljunk a configban lévő tickereken (vagy kérjünk egy újat a DM-től).
        # Egyszerűsítés: Feltételezzük, hogy van egy listánk, vagy bővítjük a DM-et.

        # Javasolt bővítés a DataManager-ben (lásd Lépés C), itt most feltételezem:
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
                print(f" -> {ticker} ({timestamp_str}) eredmény: {pnl*100:.2f}%")

    def _calculate_trade_result(self, ticker, entry_date_str, hold_period=10):
        """
        Megnézi, mi történt az árfolyammal a belépés után X nappal.
        Egyszerűsített logika: Fix 10 napos tartás eredménye.
        """
        entry_date = pd.Timestamp(entry_date_str)
        # Pár nappal későbbi adatok kellenek
        check_end_date = entry_date + timedelta(days=hold_period + 5)

        df = self.dm.load_ohlcv(ticker)  # Ez cache-elt adatokat használ

        # Szűrés a releváns időszakra
        df_future = df[df.index > entry_date]

        if df_future.empty:
            return None  # Még nincs jövőbeli adat (mai döntés)

        # Belépő ár (a döntés másnapjának nyitója vagy aznapi close - egyszerűsítsünk close-ra)
        try:
            # Megkeressük a pontos entry dátumhoz legközelebbi árat
            entry_idx = df.index.get_indexer([entry_date], method="nearest")[0]
            entry_price = df.iloc[entry_idx]["Close"]

            # Kilépő ár (hold_period nappal később)
            exit_idx = entry_idx + hold_period
            if exit_idx >= len(df):
                exit_idx = len(df) - 1  # Ami van adat

            # Ha túl közel van még, nem értékeljük
            if (df.index[exit_idx] - df.index[entry_idx]).days < 3:
                return None

            exit_price = df.iloc[exit_idx]["Close"]

            return (exit_price - entry_price) / entry_price

        except Exception as e:
            # Hiba esetén (pl. hiányzó adat) skip
            return None
