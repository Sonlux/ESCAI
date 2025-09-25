"""
GitHub CI/CD Automation Module

This module provides automated CI/CD workflow management that integrates with GitHub Actions
through the MCP GitHub integration. It handles workflow triggering, real-time monitoring,
automatic commits/pushes, and rollback capabilities.
"""

from .models import (
    WorkflowRun, WorkflowJob, WorkflowStep, AutomationSession,
    WorkflowStatus, WorkflowConclusion, JobStatus, StepStatus, AutomationSessionStatus
)
from .interfaces import (
    AutomationConfig, RepositoryConfig,
    GitHubCICDError, WorkflowTriggerError, WorkflowMonitoringError,
    CommitError, RollbackError, AuthenticationError, RateLimitError
)
from . import constants
from . import utils

# Import implemented classes
from .github_mcp_client import GitHubMCPClient
from .status_monitor import StatusMonitor, ProgressReport, MonitoringSession, MonitoringStatus
from .commit_manager import CommitManager, CommitContext, CommitResult, PushResult

# Import placeholder classes (will be implemented in future tasks)
# from .workflow_manager import WorkflowManager
# from .rollback_manager import RollbackManager
# from .error_handler import ErrorHandler

__version__ = "1.0.0"

__all__ = [
    # Data models
    "WorkflowRun",
    "WorkflowJob", 
    "WorkflowStep",
    "AutomationSession",
    "WorkflowStatus",
    "WorkflowConclusion",
    "JobStatus",
    "StepStatus",
    "AutomationSessionStatus",
    
    # Configuration
    "AutomationConfig",
    "RepositoryConfig",
    
    # Exceptions
    "GitHubCICDError",
    "WorkflowTriggerError",
    "WorkflowMonitoringError",
    "CommitError",
    "RollbackError",
    "AuthenticationError",
    "RateLimitError",
    
    # Modules
    "constants",
    "utils",
    
    # Main classes
    "GitHubMCPClient",
    "StatusMonitor",
    "ProgressReport",
    "MonitoringSession", 
    "MonitoringStatus",
    "CommitManager",
    "CommitContext",
    "CommitResult",
    "PushResult",
    
    # Main classes (to be implemented in future tasks)
    # "WorkflowManager",
    # "RollbackManager",
    # "ErrorHandler",
]