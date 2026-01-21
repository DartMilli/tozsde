from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionCostModel:
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    def apply(self, raw_return: float) -> float:
        """
        Adjust raw return with realistic trading costs.
        """
        cost = self.commission + self.slippage
        return raw_return - cost
