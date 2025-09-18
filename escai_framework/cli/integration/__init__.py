"""
CLI integration module for connecting to ESCAI framework instrumentors.
"""

from .framework_connector import FrameworkConnector, get_framework_connector, framework_context

__all__ = ['FrameworkConnector', 'get_framework_connector', 'framework_context']