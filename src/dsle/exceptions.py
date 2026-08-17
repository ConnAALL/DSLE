"""Public exception hierarchy for DSLE."""


class DSLEError(Exception):
    """Base class for errors raised by DSLE."""


class ConfigurationError(DSLEError, ValueError):
    """A DSLE configuration file is missing, inconsistent, or invalid."""


class AssetError(DSLEError, FileNotFoundError):
    """A required game-independent runtime asset is unavailable."""


class RuntimeUnavailableError(DSLEError, RuntimeError):
    """The live-game backend cannot be used in the current process."""


class RuntimeCommunicationError(DSLEError, RuntimeError):
    """Communication with a running game instance failed."""


class ResetError(DSLEError, RuntimeError):
    """A boss encounter could not be prepared for an episode."""
