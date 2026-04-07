"""Dashboard and monitoring module."""

from .tracker import DashboardTracker

try:
    from .app import DashboardApp
except ImportError:
    # Flask is optional
    DashboardApp = None

__all__ = ['DashboardTracker', 'DashboardApp']
