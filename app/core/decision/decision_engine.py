from datetime import date


class DecisionEngine:
    def __init__(self, safety_engine=None, enable_safety=False, today=None):
        self.safety_engine = safety_engine
        self.enable_safety = enable_safety
        self.today = today or date.today()

    def run(self, ticker: str, decision: dict) -> dict:
        if self.enable_safety and self.safety_engine:
            decision = self.safety_engine.apply(
                ticker=ticker,
                decision=decision,
                today=self.today,
            )

        return decision
