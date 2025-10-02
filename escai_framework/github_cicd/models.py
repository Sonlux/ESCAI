"""
Core data models for GitHub CI/CD automation system.

This module defines the data structures used throughout the GitHub CI/CD automation
system, including workflow runs, jobs, steps, and automation sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class WorkflowStatus(Enum):
    """Enumeration of possible workflow statuses."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowConclusion(Enum):
    """Enumeration of possible workflow conclusions."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"
    NEUTRAL = "neutral"


class JobStatus(Enum):
    """Enumeration of possible job statuses."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class StepStatus(Enum):
    """Enumeration of possible step statuses."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class AutomationSessionStatus(Enum):
    """Enumeration of possible automation session statuses."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """
    Represents a single step within a workflow job.
    
    Attributes:
        name: The name of the step
        status: Current status of the step
        conclusion: Final conclusion of the step (if completed)
        number: Step number within the job
        started_at: When the step started execution
        completed_at: When the step completed execution
    """
    name: str
    status: StepStatus
    number: int
    conclusion: Optional[WorkflowConclusion] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def is_completed(self) -> bool:
        """Check if the step has completed."""
        return self.status == StepStatus.COMPLETED

    def is_successful(self) -> bool:
        """Check if the step completed successfully."""
        return (self.is_completed() and 
                self.conclusion == WorkflowConclusion.SUCCESS)

    def duration_seconds(self) -> Optional[float]:
        """Calculate step duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class WorkflowJob:
    """
    Represents a job within a workflow run.
    
    Attributes:
        id: Unique identifier for the job
        name: The name of the job
        status: Current status of the job
        conclusion: Final conclusion of the job (if completed)
        started_at: When the job started execution
        completed_at: When the job completed execution
        steps: List of steps within this job
    """
    id: int
    name: str
    status: JobStatus
    steps: List[WorkflowStep] = field(default_factory=list)
    conclusion: Optional[WorkflowConclusion] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def is_completed(self) -> bool:
        """Check if the job has completed."""
        return self.status == JobStatus.COMPLETED

    def is_successful(self) -> bool:
        """Check if the job completed successfully."""
        return (self.is_completed() and 
                self.conclusion == WorkflowConclusion.SUCCESS)

    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_failed_steps(self) -> List[WorkflowStep]:
        """Get list of failed steps in this job."""
        return [step for step in self.steps 
                if step.conclusion == WorkflowConclusion.FAILURE]

    def get_step_by_name(self, name: str) -> Optional[WorkflowStep]:
        """Get a step by its name."""
        return next((step for step in self.steps if step.name == name), None)


@dataclass
class WorkflowRun:
    """
    Represents a complete workflow run.
    
    Attributes:
        id: Unique identifier for the workflow run
        workflow_id: ID of the workflow definition
        workflow_name: Name of the workflow
        status: Current status of the workflow run
        conclusion: Final conclusion of the workflow run (if completed)
        created_at: When the workflow run was created
        updated_at: When the workflow run was last updated
        jobs: List of jobs in this workflow run
        repository: Repository name (owner/repo format)
        branch: Branch the workflow ran on
        commit_sha: SHA of the commit that triggered the workflow
        html_url: URL to view the workflow run on GitHub
        run_number: Sequential run number for this workflow
        event: Event that triggered the workflow
        actor: User who triggered the workflow
    """
    id: int
    workflow_id: int
    workflow_name: str
    status: WorkflowStatus
    repository: str
    branch: str
    commit_sha: str
    created_at: datetime
    updated_at: datetime
    jobs: List[WorkflowJob] = field(default_factory=list)
    conclusion: Optional[WorkflowConclusion] = None
    html_url: Optional[str] = None
    run_number: Optional[int] = None
    event: Optional[str] = None
    actor: Optional[str] = None

    def is_completed(self) -> bool:
        """Check if the workflow run has completed."""
        return self.status == WorkflowStatus.COMPLETED

    def is_successful(self) -> bool:
        """Check if the workflow run completed successfully."""
        return (self.is_completed() and 
                self.conclusion == WorkflowConclusion.SUCCESS)

    def is_failed(self) -> bool:
        """Check if the workflow run failed."""
        return (self.status == WorkflowStatus.FAILED or
                self.conclusion == WorkflowConclusion.FAILURE)

    def duration_seconds(self) -> Optional[float]:
        """Calculate workflow run duration in seconds."""
        if self.created_at and self.updated_at and self.is_completed():
            return (self.updated_at - self.created_at).total_seconds()
        return None

    def get_failed_jobs(self) -> List[WorkflowJob]:
        """Get list of failed jobs in this workflow run."""
        return [job for job in self.jobs 
                if job.conclusion == WorkflowConclusion.FAILURE]

    def get_job_by_name(self, name: str) -> Optional[WorkflowJob]:
        """Get a job by its name."""
        return next((job for job in self.jobs if job.name == name), None)

    def get_progress_percentage(self) -> float:
        """Calculate overall progress percentage based on completed jobs."""
        if not self.jobs:
            return 0.0
        
        completed_jobs = sum(1 for job in self.jobs if job.is_completed())
        return (completed_jobs / len(self.jobs)) * 100.0


@dataclass
class AutomationSession:
    """
    Represents an automation session that tracks the complete CI/CD automation process.
    
    Attributes:
        session_id: Unique identifier for the automation session
        workflow_run_id: ID of the associated workflow run
        repository: Repository name (owner/repo format)
        started_at: When the automation session started
        status: Current status of the automation session
        commits_made: List of commit SHAs made during this session
        rollback_point: Commit SHA to rollback to if needed
        error_log: List of error messages encountered during the session
        config: Configuration parameters for this session
        metadata: Additional metadata about the session
    """
    session_id: str
    workflow_run_id: int
    repository: str
    started_at: datetime
    status: AutomationSessionStatus
    commits_made: List[str] = field(default_factory=list)
    rollback_point: Optional[str] = None
    error_log: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[datetime] = None

    def is_active(self) -> bool:
        """Check if the automation session is currently active."""
        return self.status == AutomationSessionStatus.ACTIVE

    def is_completed(self) -> bool:
        """Check if the automation session has completed."""
        return self.status == AutomationSessionStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if the automation session has failed."""
        return self.status == AutomationSessionStatus.FAILED

    def is_rolled_back(self) -> bool:
        """Check if the automation session was rolled back."""
        return self.status == AutomationSessionStatus.ROLLED_BACK

    def add_commit(self, commit_sha: str) -> None:
        """Add a commit SHA to the session's commit history."""
        if commit_sha not in self.commits_made:
            self.commits_made.append(commit_sha)

    def add_error(self, error_message: str) -> None:
        """Add an error message to the session's error log."""
        timestamp = datetime.now().isoformat()
        self.error_log.append(f"[{timestamp}] {error_message}")

    def duration_seconds(self) -> Optional[float]:
        """Calculate session duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_latest_commit(self) -> Optional[str]:
        """Get the most recent commit SHA from this session."""
        return self.commits_made[-1] if self.commits_made else None


# Type aliases for better code readability
WorkflowRunId = int
JobId = int
SessionId = str
CommitSHA = str
RepositoryName = str