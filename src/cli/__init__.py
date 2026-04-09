"""GovOn CLI package."""

try:
    from importlib.metadata import version

    __version__ = version("govon")
except Exception:
    __version__ = "0.0.0-dev"
