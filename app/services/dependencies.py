from typing import Any, Optional

from app.data_access.data_loader import load_data
from app.models.rl_inference import RLModelEnsembleRunner
from app.notifications.mailer import send_email


class MarketDataFetcher:
    """Thin wrapper for market data retrieval."""

    def __init__(self):
        self._cache = {}

    def load_data(self, ticker: str, start: str, end: Optional[str] = None):
        cache_key = (ticker, start, end)
        if cache_key not in self._cache:
            self._cache[cache_key] = load_data(ticker, start=start, end=end)
        return self._cache[cache_key]


class ModelEnsembleRunner:
    """Thin wrapper for model loading/inference."""

    def __init__(self, model_dir, env_class: Any):
        self._runner = RLModelEnsembleRunner(model_dir=model_dir, env_class=env_class)

    def run_ensemble(self, df, ticker: str, top_n: int = 3, debug: bool = True):
        return self._runner.run_ensemble(
            df=df,
            ticker=ticker,
            top_n=top_n,
            debug=debug,
        )


class EmailNotifier:
    """Thin wrapper for email notifications."""

    def send(self, subject: str, body: str, recipient: str):
        send_email(subject, body, recipient)
