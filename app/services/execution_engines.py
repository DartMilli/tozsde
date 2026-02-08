class NoopExecutionEngine:
    """Execution engine that performs no trades."""

    def __init__(self, logger):
        self.logger = logger

    def execute(self, decisions, as_of):
        self.logger.info("Execution engine: noop (no trades executed).")
