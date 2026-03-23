from typing import Any, Optional


def get_conf(settings: Optional[Any] = None) -> Any:
    """Return provided settings or raise if not supplied (DI root required)."""
    if settings is None:
        raise RuntimeError("Settings must be injected via DI root")
    return settings
