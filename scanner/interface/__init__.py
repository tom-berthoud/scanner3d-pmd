"""scanner.interface — Web interface and local display.

Exports:
    create_app: create the configured Flask application.
"""

from scanner.interface.web import create_app

__all__ = ["create_app"]
