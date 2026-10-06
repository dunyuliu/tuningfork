"""The HTTP API behind the CLI and the web UI."""

from tuningfork.api.control import control_routes
from tuningfork.api.data import data_routes

__all__ = ["control_routes", "data_routes"]
