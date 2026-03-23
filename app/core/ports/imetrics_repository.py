from typing import Protocol, Any


class IMetricsRepository(Protocol):
    def save_metrics(self, metrics: dict) -> None: ...
    def fetch_metrics(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[dict]: ...
