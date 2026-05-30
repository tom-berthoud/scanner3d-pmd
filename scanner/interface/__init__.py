"""scanner.interface - Web interface and local display."""


def create_app(*args, **kwargs):
    """Create the Flask app, importing Flask dependencies only when needed."""
    from scanner.interface.web import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
