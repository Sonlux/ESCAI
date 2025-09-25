"""
Base interfaces and type definitions for the GitHub CI/CD automation system.

This module defines the core interfaces and protocols that components must implement
to ensure consistent behavior across the automation system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol, Union
from datetime import datetime

from .models import (
    WorkflowRun, WorkflowJob, WorkflowStep, AutomationSession,
    WorkflowRunId, JobId, SessionId, CommitSHA, RepositoryName
)


class GitHubMCPClientInterface(Protocol):
    """Protocol defining the interface for GitHub MCP client operations."""
    
    def get_workflows(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get list of available workflows in a repository."""
        ...
    
    def trigger_workflow_dispatch(
        self, 
        owner: str, 
        repo: str, 
        workflow_id: str, 
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger a workflow dispatch event."""
        ...
    
    def get_workflow_run(self, owner: str, repo: str, run_id: int) -> Dict[str, Any]:
        """Get details of a specific workflow run."""
        ...
    
    def get_workflow_run_jobs(self, owner: str, repo: str, run_id: int) -> List[Dict[str, Any]]:
        """Get jobs for a specific workflow run."""
        ...


class StatusMonitorInterface(ABC):
    """Abstract base class for workflow status monitoring."""
    
    @abstractmethod
    async def start_monitoring(self, workflow_run_id: WorkflowRunId) -> None:
        """Start monitoring a workflow run."""
        pass
    
    @abstractmethod
    async def get_current_status(self, workflow_run_id: WorkflowRunId) -> Optional[WorkflowRun]:
        """Get current status of a workflow run."""
        pass
    
    @abstractmethod
    async def stop_monitoring(self, workflow_run_id: WorkflowRunId) -> None:
        """Stop monitoring a workflow run."""
        pass
    
    @abstractmethod
    def generate_progress_report(self, workflow_run: WorkflowRun) -> Dict[str, Any]:
        """Generate a progress report for a workflow run."""
        pass


class CommitManagerInterface(ABC):
    """Abstract base class for commit management operations."""
    
    @abstractmethod
    async def auto_commit(
        self, 
        message: str, 
        files: Optional[List[str]] = None
    ) -> CommitSHA:
        """Create an automatic commit with the specified message and files."""
        pass
    
    @abstractmethod
    async def create_workflow_commit(
        self, 
        workflow_context: Dict[str, Any], 
        changes: List[str]
    ) -> CommitSHA:
        """Create a commit with workflow context information."""
        pass
    
    @abstractmethod
    async def push_changes(self, branch: str = "main") -> bool:
        """Push committed changes to the remote repository."""
        pass
    
    @abstractmethod
    def get_commit_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commit history."""
        pass


class RollbackManagerInterface(ABC):
    """Abstract base class for rollback management operations."""
    
    @abstractmethod
    async def identify_rollback_point(self, workflow_run_id: WorkflowRunId) -> Optional[CommitSHA]:
        """Identify the appropriate rollback point for a failed workflow."""
        pass
    
    @abstractmethod
    async def perform_rollback(self, target_commit: CommitSHA) -> bool:
        """Perform rollback to the specified commit."""
        pass
    
    @abstractmethod
    async def create_revert_commit(self, failed_commit: CommitSHA, reason: str) -> CommitSHA:
        """Create a revert commit for the failed commit."""
        pass
    
    @abstractmethod
    async def verify_rollback_success(self) -> bool:
        """Verify that the rollback operation was successful."""
        pass


class ErrorHandlerInterface(ABC):
    """Abstract base class for error handling operations."""
    
    @abstractmethod
    async def handle_api_rate_limit(self, retry_after: int) -> None:
        """Handle API rate limit errors with appropriate backoff."""
        pass
    
    @abstractmethod
    async def handle_network_error(self, operation: str, retry_count: int) -> bool:
        """Handle network errors with retry logic."""
        pass
    
    @abstractmethod
    def handle_authentication_error(self) -> Dict[str, str]:
        """Handle authentication errors and provide guidance."""
        pass
    
    @abstractmethod
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with contextual information."""
        pass


class WorkflowManagerInterface(ABC):
    """Abstract base class for workflow management operations."""
    
    @abstractmethod
    async def trigger_workflow(
        self, 
        repo_owner: str, 
        repo_name: str, 
        workflow_name: str, 
        inputs: Optional[Dict[str, Any]] = None
    ) -> WorkflowRunId:
        """Trigger a workflow and return the run ID."""
        pass
    
    @abstractmethod
    async def monitor_workflow_progress(self, workflow_run_id: WorkflowRunId) -> WorkflowRun:
        """Monitor workflow progress and return current state."""
        pass
    
    @abstractmethod
    async def list_available_workflows(self, repo_owner: str, repo_name: str) -> List[Dict[str, Any]]:
        """List all available workflows in a repository."""
        pass
    
    @abstractmethod
    async def get_workflow_status(self, workflow_run_id: WorkflowRunId) -> Optional[WorkflowRun]:
        """Get current status of a specific workflow run."""
        pass
    
    @abstractmethod
    async def create_automation_session(
        self, 
        workflow_run_id: WorkflowRunId, 
        repository: RepositoryName
    ) -> AutomationSession:
        """Create a new automation session for tracking."""
        pass


# Configuration types
class AutomationConfig:
    """Configuration class for automation settings."""
    
    def __init__(
        self,
        polling_interval: int = 30,
        timeout: int = 3600,
        auto_commit: bool = True,
        rollback_enabled: bool = True,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        rate_limit_buffer: int = 100
    ):
        self.polling_interval = polling_interval
        self.timeout = timeout
        self.auto_commit = auto_commit
        self.rollback_enabled = rollback_enabled
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.rate_limit_buffer = rate_limit_buffer
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "polling_interval": self.polling_interval,
            "timeout": self.timeout,
            "auto_commit": self.auto_commit,
            "rollback_enabled": self.rollback_enabled,
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "rate_limit_buffer": self.rate_limit_buffer
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AutomationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


class RepositoryConfig:
    """Configuration class for repository-specific settings."""
    
    def __init__(
        self,
        owner: str,
        name: str,
        default_branch: str = "main",
        workflows: Optional[List[Dict[str, Any]]] = None
    ):
        self.owner = owner
        self.name = name
        self.default_branch = default_branch
        self.workflows = workflows or []
    
    @property
    def full_name(self) -> str:
        """Get the full repository name (owner/name)."""
        return f"{self.owner}/{self.name}"
    
    def get_workflow_config(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific workflow."""
        return next(
            (wf for wf in self.workflows if wf.get("name") == workflow_name),
            None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert repository config to dictionary."""
        return {
            "owner": self.owner,
            "name": self.name,
            "default_branch": self.default_branch,
            "workflows": self.workflows
        }


# Event types for the automation system
class AutomationEvent:
    """Base class for automation events."""
    
    def __init__(self, event_type: str, timestamp: Optional[datetime] = None):
        self.event_type = event_type
        self.timestamp = timestamp or datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the event."""
        self.metadata[key] = value


class WorkflowTriggeredEvent(AutomationEvent):
    """Event fired when a workflow is triggered."""
    
    def __init__(self, workflow_run_id: WorkflowRunId, repository: RepositoryName):
        super().__init__("workflow_triggered")
        self.workflow_run_id = workflow_run_id
        self.repository = repository


class WorkflowCompletedEvent(AutomationEvent):
    """Event fired when a workflow completes."""
    
    def __init__(self, workflow_run: WorkflowRun):
        super().__init__("workflow_completed")
        self.workflow_run = workflow_run


class WorkflowFailedEvent(AutomationEvent):
    """Event fired when a workflow fails."""
    
    def __init__(self, workflow_run: WorkflowRun, error_details: Dict[str, Any]):
        super().__init__("workflow_failed")
        self.workflow_run = workflow_run
        self.error_details = error_details


class CommitCreatedEvent(AutomationEvent):
    """Event fired when a commit is created."""
    
    def __init__(self, commit_sha: CommitSHA, message: str, session_id: SessionId):
        super().__init__("commit_created")
        self.commit_sha = commit_sha
        self.message = message
        self.session_id = session_id


class RollbackInitiatedEvent(AutomationEvent):
    """Event fired when a rollback is initiated."""
    
    def __init__(self, session_id: SessionId, target_commit: CommitSHA, reason: str):
        super().__init__("rollback_initiated")
        self.session_id = session_id
        self.target_commit = target_commit
        self.reason = reason


# Exception types for the automation system
class GitHubCICDError(Exception):
    """Base exception for GitHub CI/CD automation errors."""
    pass


class WorkflowTriggerError(GitHubCICDError):
    """Exception raised when workflow triggering fails."""
    pass


class WorkflowMonitoringError(GitHubCICDError):
    """Exception raised when workflow monitoring fails."""
    pass


class CommitError(GitHubCICDError):
    """Exception raised when commit operations fail."""
    pass


class RollbackError(GitHubCICDError):
    """Exception raised when rollback operations fail."""
    pass


class AuthenticationError(GitHubCICDError):
    """Exception raised when authentication fails."""
    pass


class RateLimitError(GitHubCICDError):
    """Exception raised when API rate limits are exceeded."""
    pass