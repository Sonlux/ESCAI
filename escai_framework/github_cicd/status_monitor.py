"""
Status Monitor for GitHub CI/CD Automation

This module provides real-time monitoring and reporting of workflow progress.
It implements polling mechanisms with configurable intervals, progress reporting
with job-level and step-level details, and status caching to optimize API usage.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from .models import WorkflowRun, WorkflowJob, WorkflowStep, WorkflowStatus, WorkflowConclusion
from .github_mcp_client import GitHubMCPClient
from .interfaces import RateLimitError, AuthenticationError, WorkflowTriggerError


logger = logging.getLogger(__name__)


class MonitoringStatus(Enum):
    """Status of the monitoring session."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ProgressReport:
    """
    Represents a progress report for a workflow run.
    
    Attributes:
        workflow_run_id: ID of the workflow run
        repository: Repository name (owner/repo format)
        workflow_name: Name of the workflow
        status: Current workflow status
        conclusion: Final conclusion (if completed)
        progress_percentage: Overall progress percentage (0-100)
        jobs_completed: Number of completed jobs
        jobs_total: Total number of jobs
        steps_completed: Number of completed steps across all jobs
        steps_total: Total number of steps across all jobs
        duration_seconds: Duration since workflow started
        estimated_remaining_seconds: Estimated time remaining
        failed_jobs: List of failed job names
        failed_steps: List of failed step names
        last_updated: When this report was generated
        html_url: URL to view the workflow on GitHub
    """
    workflow_run_id: int
    repository: str
    workflow_name: str
    status: WorkflowStatus
    conclusion: Optional[WorkflowConclusion]
    progress_percentage: float
    jobs_completed: int
    jobs_total: int
    steps_completed: int
    steps_total: int
    duration_seconds: Optional[float]
    estimated_remaining_seconds: Optional[float]
    failed_jobs: List[str]
    failed_steps: List[str]
    last_updated: datetime
    html_url: Optional[str] = None
    
    def is_completed(self) -> bool:
        """Check if the workflow is completed."""
        return self.status == WorkflowStatus.COMPLETED
    
    def is_successful(self) -> bool:
        """Check if the workflow completed successfully."""
        return (self.is_completed() and 
                self.conclusion == WorkflowConclusion.SUCCESS)
    
    def has_failures(self) -> bool:
        """Check if there are any failures."""
        return len(self.failed_jobs) > 0 or len(self.failed_steps) > 0


@dataclass
class MonitoringSession:
    """
    Represents a monitoring session for one or more workflow runs.
    
    Attributes:
        session_id: Unique identifier for the monitoring session
        workflow_run_ids: Set of workflow run IDs being monitored
        repository: Repository name (owner/repo format)
        status: Current monitoring status
        poll_interval: Polling interval in seconds
        timeout_seconds: Maximum monitoring duration
        started_at: When monitoring started
        last_poll_at: When the last poll occurred
        error_count: Number of errors encountered
        progress_callbacks: List of callback functions for progress updates
        completion_callbacks: List of callback functions for completion
        cached_reports: Cache of the latest progress reports
    """
    session_id: str
    workflow_run_ids: Set[int]
    repository: str
    status: MonitoringStatus
    poll_interval: int
    timeout_seconds: int
    started_at: datetime
    last_poll_at: Optional[datetime] = None
    error_count: int = 0
    progress_callbacks: List[Callable[[ProgressReport], None]] = field(default_factory=list)
    completion_callbacks: List[Callable[[ProgressReport], None]] = field(default_factory=list)
    cached_reports: Dict[int, ProgressReport] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if the monitoring session is active."""
        return self.status == MonitoringStatus.ACTIVE
    
    def is_timed_out(self) -> bool:
        """Check if the monitoring session has timed out."""
        if self.timeout_seconds <= 0:
            return False
        elapsed = (datetime.now() - self.started_at).total_seconds()
        return elapsed > self.timeout_seconds
    
    def add_workflow_run(self, run_id: int) -> None:
        """Add a workflow run to monitor."""
        self.workflow_run_ids.add(run_id)
    
    def remove_workflow_run(self, run_id: int) -> None:
        """Remove a workflow run from monitoring."""
        self.workflow_run_ids.discard(run_id)
        self.cached_reports.pop(run_id, None)


class StatusMonitor:
    """
    Real-time workflow progress tracking system.
    
    This class provides comprehensive monitoring capabilities for GitHub Actions
    workflows, including polling mechanisms, progress reporting, and status caching
    to optimize API usage and avoid redundant calls.
    """
    
    def __init__(
        self,
        github_client: GitHubMCPClient,
        default_poll_interval: int = 30,
        default_timeout: int = 3600,
        max_concurrent_sessions: int = 10,
        cache_ttl_seconds: int = 60
    ):
        """
        Initialize the status monitor.
        
        Args:
            github_client: GitHub MCP client for API interactions
            default_poll_interval: Default polling interval in seconds
            default_timeout: Default timeout for monitoring sessions in seconds
            max_concurrent_sessions: Maximum number of concurrent monitoring sessions
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.github_client = github_client
        self.default_poll_interval = default_poll_interval
        self.default_timeout = default_timeout
        self.max_concurrent_sessions = max_concurrent_sessions
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Active monitoring sessions
        self.active_sessions: Dict[str, MonitoringSession] = {}
        
        # Status cache to avoid redundant API calls
        self.status_cache: Dict[int, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[int, datetime] = {}
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("StatusMonitor initialized")
    
    async def start_monitoring(
        self,
        workflow_run_id: int,
        repository: str,
        poll_interval: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        progress_callback: Optional[Callable[[ProgressReport], None]] = None,
        completion_callback: Optional[Callable[[ProgressReport], None]] = None
    ) -> str:
        """
        Start monitoring a workflow run.
        
        Args:
            workflow_run_id: ID of the workflow run to monitor
            repository: Repository name in owner/repo format
            poll_interval: Polling interval in seconds (uses default if None)
            timeout_seconds: Timeout in seconds (uses default if None)
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback for completion
            
        Returns:
            Session ID for the monitoring session
            
        Raises:
            ValueError: If maximum concurrent sessions exceeded
            WorkflowTriggerError: If monitoring setup fails
        """
        if len(self.active_sessions) >= self.max_concurrent_sessions:
            raise ValueError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) exceeded")
        
        # Generate unique session ID
        session_id = f"monitor_{workflow_run_id}_{int(datetime.now().timestamp())}"
        
        # Create monitoring session
        session = MonitoringSession(
            session_id=session_id,
            workflow_run_ids={workflow_run_id},
            repository=repository,
            status=MonitoringStatus.ACTIVE,
            poll_interval=poll_interval or self.default_poll_interval,
            timeout_seconds=timeout_seconds or self.default_timeout,
            started_at=datetime.now()
        )
        
        # Add callbacks if provided
        if progress_callback:
            session.progress_callbacks.append(progress_callback)
        if completion_callback:
            session.completion_callbacks.append(completion_callback)
        
        # Store session
        self.active_sessions[session_id] = session
        
        # Start background monitoring if not already running
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Started monitoring workflow run {workflow_run_id} (session: {session_id})")
        return session_id
    
    async def stop_monitoring(self, session_id: str) -> bool:
        """
        Stop monitoring a specific session.
        
        Args:
            session_id: ID of the monitoring session to stop
            
        Returns:
            True if session was stopped, False if session not found
        """
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.status = MonitoringStatus.STOPPED
        
        # Clean up cached data for this session's workflow runs
        for run_id in session.workflow_run_ids:
            self.status_cache.pop(run_id, None)
            self.cache_timestamps.pop(run_id, None)
        
        # Remove session
        del self.active_sessions[session_id]
        
        logger.info(f"Stopped monitoring session {session_id}")
        return True
    
    async def get_current_status(self, workflow_run_id: int, repository: str) -> ProgressReport:
        """
        Get current status of a workflow run.
        
        Args:
            workflow_run_id: ID of the workflow run
            repository: Repository name in owner/repo format
            
        Returns:
            Current progress report for the workflow run
            
        Raises:
            WorkflowTriggerError: If status retrieval fails
        """
        try:
            # Check cache first
            cached_report = self._get_cached_status(workflow_run_id)
            if cached_report:
                return cached_report
            
            # Fetch fresh data from GitHub
            owner, repo = repository.split('/', 1)
            workflow_run = await self.github_client.get_workflow_run(owner, repo, workflow_run_id)
            jobs = await self.github_client.get_workflow_run_jobs(owner, repo, workflow_run_id)
            
            # Update workflow run with jobs
            workflow_run.jobs = jobs
            
            # Generate progress report
            report = self._generate_progress_report(workflow_run)
            
            # Cache the report
            self._cache_status(workflow_run_id, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to get status for workflow run {workflow_run_id}: {str(e)}")
            raise WorkflowTriggerError(f"Failed to get workflow status: {str(e)}")
    
    async def generate_progress_report(self, workflow_run_id: int, repository: str) -> ProgressReport:
        """
        Generate a detailed progress report for a workflow run.
        
        Args:
            workflow_run_id: ID of the workflow run
            repository: Repository name in owner/repo format
            
        Returns:
            Detailed progress report
        """
        return await self.get_current_status(workflow_run_id, repository)
    
    async def wait_for_completion(
        self,
        workflow_run_id: int,
        repository: str,
        timeout: int = 3600,
        poll_interval: int = 30
    ) -> ProgressReport:
        """
        Wait for a workflow run to complete.
        
        Args:
            workflow_run_id: ID of the workflow run to wait for
            repository: Repository name in owner/repo format
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds
            
        Returns:
            Final progress report when workflow completes
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            WorkflowTriggerError: If monitoring fails
        """
        start_time = datetime.now()
        timeout_time = start_time + timedelta(seconds=timeout)
        
        logger.info(f"Waiting for workflow run {workflow_run_id} to complete (timeout: {timeout}s)")
        
        while datetime.now() < timeout_time:
            try:
                report = await self.get_current_status(workflow_run_id, repository)
                
                if report.is_completed():
                    logger.info(f"Workflow run {workflow_run_id} completed with status: {report.conclusion}")
                    return report
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error while waiting for workflow completion: {str(e)}")
                # Continue polling unless it's a critical error
                if isinstance(e, (AuthenticationError, RateLimitError)):
                    raise
                await asyncio.sleep(poll_interval)
        
        # Timeout exceeded
        final_report = await self.get_current_status(workflow_run_id, repository)
        logger.warning(f"Timeout waiting for workflow run {workflow_run_id} to complete")
        raise asyncio.TimeoutError(f"Workflow run {workflow_run_id} did not complete within {timeout} seconds")
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active monitoring session IDs.
        
        Returns:
            List of active session IDs
        """
        return list(self.active_sessions.keys())
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a monitoring session.
        
        Args:
            session_id: ID of the monitoring session
            
        Returns:
            Dictionary with session information, or None if not found
        """
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            'session_id': session.session_id,
            'workflow_run_ids': list(session.workflow_run_ids),
            'repository': session.repository,
            'status': session.status.value,
            'poll_interval': session.poll_interval,
            'timeout_seconds': session.timeout_seconds,
            'started_at': session.started_at.isoformat(),
            'last_poll_at': session.last_poll_at.isoformat() if session.last_poll_at else None,
            'error_count': session.error_count,
            'is_timed_out': session.is_timed_out()
        }
    
    async def shutdown(self) -> None:
        """
        Shutdown the status monitor and clean up resources.
        """
        logger.info("Shutting down StatusMonitor")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.stop_monitoring(session_id)
        
        # Cancel monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Clear caches
        self.status_cache.clear()
        self.cache_timestamps.clear()
        
        logger.info("StatusMonitor shutdown complete")
    
    async def _monitoring_loop(self) -> None:
        """
        Background monitoring loop that polls workflow status for all active sessions.
        """
        logger.info("Starting monitoring loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Process all active sessions
                sessions_to_remove = []
                
                for session_id, session in self.active_sessions.items():
                    try:
                        # Check if session should be stopped
                        if session.status != MonitoringStatus.ACTIVE or session.is_timed_out():
                            sessions_to_remove.append(session_id)
                            continue
                        
                        # Poll each workflow run in the session
                        for run_id in session.workflow_run_ids.copy():
                            await self._poll_workflow_run(session, run_id)
                        
                        # Update last poll time
                        session.last_poll_at = datetime.now()
                        
                    except Exception as e:
                        logger.error(f"Error processing session {session_id}: {str(e)}")
                        session.error_count += 1
                        
                        # Stop session if too many errors
                        if session.error_count >= 5:
                            logger.error(f"Too many errors in session {session_id}, stopping")
                            sessions_to_remove.append(session_id)
                
                # Remove completed or failed sessions
                for session_id in sessions_to_remove:
                    await self.stop_monitoring(session_id)
                
                # Wait before next polling cycle
                if self.active_sessions:
                    min_interval = min(session.poll_interval for session in self.active_sessions.values())
                    await asyncio.sleep(min_interval)
                else:
                    # No active sessions, wait longer
                    await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in monitoring loop: {str(e)}")
                await asyncio.sleep(30)  # Wait before retrying
        
        logger.info("Monitoring loop stopped")
    
    async def _poll_workflow_run(self, session: MonitoringSession, run_id: int) -> None:
        """
        Poll a single workflow run and update progress.
        
        Args:
            session: Monitoring session
            run_id: Workflow run ID to poll
        """
        try:
            # Get current status
            report = await self.get_current_status(run_id, session.repository)
            
            # Update cached report
            session.cached_reports[run_id] = report
            
            # Call progress callbacks
            for callback in session.progress_callbacks:
                try:
                    callback(report)
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")
            
            # Check if workflow is completed
            if report.is_completed():
                # Call completion callbacks
                for callback in session.completion_callbacks:
                    try:
                        callback(report)
                    except Exception as e:
                        logger.error(f"Error in completion callback: {str(e)}")
                
                # Remove completed workflow from session
                session.workflow_run_ids.discard(run_id)
                
                logger.info(f"Workflow run {run_id} completed: {report.conclusion}")
            
        except RateLimitError as e:
            logger.warning(f"Rate limit hit while polling workflow {run_id}: {str(e)}")
            # Don't increment error count for rate limits
        except Exception as e:
            logger.error(f"Error polling workflow run {run_id}: {str(e)}")
            session.error_count += 1
    
    def _get_cached_status(self, workflow_run_id: int) -> Optional[ProgressReport]:
        """
        Get cached status for a workflow run if still valid.
        
        Args:
            workflow_run_id: Workflow run ID
            
        Returns:
            Cached progress report if valid, None otherwise
        """
        if workflow_run_id not in self.cache_timestamps:
            return None
        
        cache_time = self.cache_timestamps[workflow_run_id]
        if (datetime.now() - cache_time).total_seconds() > self.cache_ttl_seconds:
            # Cache expired
            self.status_cache.pop(workflow_run_id, None)
            self.cache_timestamps.pop(workflow_run_id, None)
            return None
        
        cached_data = self.status_cache.get(workflow_run_id)
        if cached_data and 'report' in cached_data:
            return cached_data['report']
        
        return None
    
    def _cache_status(self, workflow_run_id: int, report: ProgressReport) -> None:
        """
        Cache a progress report.
        
        Args:
            workflow_run_id: Workflow run ID
            report: Progress report to cache
        """
        self.status_cache[workflow_run_id] = {'report': report}
        self.cache_timestamps[workflow_run_id] = datetime.now()
    
    def _generate_progress_report(self, workflow_run: WorkflowRun) -> ProgressReport:
        """
        Generate a progress report from a workflow run.
        
        Args:
            workflow_run: WorkflowRun object
            
        Returns:
            Generated progress report
        """
        # Calculate job statistics
        jobs_total = len(workflow_run.jobs)
        jobs_completed = sum(1 for job in workflow_run.jobs if job.is_completed())
        
        # Calculate step statistics
        steps_total = sum(len(job.steps) for job in workflow_run.jobs)
        steps_completed = sum(
            sum(1 for step in job.steps if step.is_completed())
            for job in workflow_run.jobs
        )
        
        # Calculate progress percentage
        if jobs_total > 0:
            progress_percentage = (jobs_completed / jobs_total) * 100.0
        else:
            progress_percentage = 0.0 if workflow_run.status != WorkflowStatus.COMPLETED else 100.0
        
        # Get failed jobs and steps
        failed_jobs = [job.name for job in workflow_run.get_failed_jobs()]
        failed_steps = []
        for job in workflow_run.jobs:
            failed_steps.extend([f"{job.name}: {step.name}" for step in job.get_failed_steps()])
        
        # Calculate duration
        duration_seconds = workflow_run.duration_seconds()
        
        # Estimate remaining time (simple heuristic)
        estimated_remaining_seconds = None
        if (duration_seconds and progress_percentage > 0 and 
            not workflow_run.is_completed() and progress_percentage < 100):
            estimated_total = duration_seconds / (progress_percentage / 100.0)
            estimated_remaining_seconds = max(0, estimated_total - duration_seconds)
        
        return ProgressReport(
            workflow_run_id=workflow_run.id,
            repository=workflow_run.repository,
            workflow_name=workflow_run.workflow_name,
            status=workflow_run.status,
            conclusion=workflow_run.conclusion,
            progress_percentage=progress_percentage,
            jobs_completed=jobs_completed,
            jobs_total=jobs_total,
            steps_completed=steps_completed,
            steps_total=steps_total,
            duration_seconds=duration_seconds,
            estimated_remaining_seconds=estimated_remaining_seconds,
            failed_jobs=failed_jobs,
            failed_steps=failed_steps,
            last_updated=datetime.now(),
            html_url=workflow_run.html_url
        )