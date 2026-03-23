from typing import Protocol, Any


class IOhlcvRepository(Protocol):
    def load_ohlcv(self, ticker: str) -> Any: ...

    def save_ohlcv(self, ticker: str, df: Any) -> None: ...
