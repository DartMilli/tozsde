class _ConfigProxy:
    def __init__(self):
        object.__setattr__(self, "_overrides", {})
        object.__setattr__(self, "_underlying", None)

    def set_underlying(self, obj):
        object.__setattr__(self, "_underlying", obj)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        underlying = object.__getattribute__(self, "_underlying")
        if underlying is not None and hasattr(underlying, name):
            return getattr(underlying, name)
        # Fallback: if legacy global Config class exists, read attributes from it
        try:
            from app.config.config import Config as GlobalConfig

            if hasattr(GlobalConfig, name):
                return getattr(GlobalConfig, name)
        except Exception:
            pass

        # For backward compatibility with tests that patch attributes on the
        # `Config` object using `mock.patch('module.Config.NAME', ...)`, return
        # None instead of raising AttributeError so the attribute is considered
        # present for patching.
        return None

    def __delattr__(self, name):
        # Support `mock.patch(..., new=...)` cleanup which uses `delattr`.
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            del overrides[name]
            return
        underlying = object.__getattribute__(self, "_underlying")
        if underlying is not None and hasattr(underlying, name):
            try:
                delattr(underlying, name)
            except Exception:
                pass

    def __setattr__(self, name, value):
        if name in ("_overrides", "_underlying"):
            object.__setattr__(self, name, value)
        else:
            self._overrides[name] = value


# Module-level singleton used by legacy code/tests that patch `X.Config`.
Config = _ConfigProxy()
