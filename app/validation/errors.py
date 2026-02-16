"""Validation error types."""


class ValidationError(RuntimeError):
    """Raised when validation detects an engine mismatch or invalid configuration."""

    pass


class EngineLogicError(RuntimeError):
    """Raised when trade index generation is inconsistent across policies."""

    pass


class DeploymentBlockedException(RuntimeError):
    """Raised when improvement check blocks deployment."""

    pass
