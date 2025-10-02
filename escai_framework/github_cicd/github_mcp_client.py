"""
GitHub MCP Client Wrapper

This module provides a wrapper around the existing GitHub MCP integration for
workflow-specific operations. It handles workflow triggering, monitoring, and
provides error handling and response validation for GitHub API interactions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .interfaces import GitHubMCPClientInterface, AuthenticationError, RateLimitError, WorkflowTriggerError
from .models import WorkflowRun, WorkflowJob, WorkflowStep, WorkflowStatus, WorkflowConclusion, JobStatus, StepStatus
from .utils import parse_github_datetime, validate_repository_format, sanitize_workflow_inputs


logger = logging.getLogger(__name__)


class GitHubMCPClient:
    """
    Wrapper around GitHub MCP integration for workflow-specific operations.
    
    This class provides a high-level interface for interacting with GitHub Actions
    workflows through the MCP GitHub integration, with added error handling,
    response validation, and workflow-specific abstractions.
    """
    
    def __init__(self, default_timeout: int = 30):
        """
        Initialize the GitHub MCP client.
        
        Args:
            default_timeout: Default timeout for API operations in seconds
        """
        self.default_timeout = default_timeout
        self._rate_limit_remaining = None
        self._rate_limit_reset = None
        
    async def get_workflows(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Get list of available workflows in a repository.
        
        Args:
            owner: Repository owner (username or organization)
            repo: Repository name
            
        Returns:
            List of workflow definitions with metadata
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            WorkflowTriggerError: If the operation fails
        """
        validate_repository_format(owner, repo)
        
        try:
            # Note: This would use the actual MCP GitHub function in a real implementation
            # For now, we'll simulate the response structure
            logger.info(f"Getting workflows from {owner}/{repo}/.github/workflows")
            
            # Mock response - in real implementation, this would come from MCP
            workflows_response = [
                {
                    'name': 'ci.yml',
                    'path': '.github/workflows/ci.yml',
                    'type': 'file',
                    'sha': 'abc123',
                    'size': 1024,
                    'download_url': f'https://raw.githubusercontent.com/{owner}/{repo}/main/.github/workflows/ci.yml',
                    'html_url': f'https://github.com/{owner}/{repo}/blob/main/.github/workflows/ci.yml'
                }
            ]
            
            workflows = []
            if isinstance(workflows_response, list):
                for item in workflows_response:
                    item_name: str = str(item.get('name', ''))
                    if item.get('type') == 'file' and item_name.endswith(('.yml', '.yaml')):
                        workflow_info = {
                            'name': item_name.replace('.yml', '').replace('.yaml', ''),
                            'path': item.get('path', ''),
                            'sha': item.get('sha', ''),
                            'size': item.get('size', 0),
                            'download_url': item.get('download_url', ''),
                            'html_url': item.get('html_url', '')
                        }
                        workflows.append(workflow_info)
            
            logger.info(f"Retrieved {len(workflows)} workflows from {owner}/{repo}")
            return workflows
            
        except Exception as e:
            logger.error(f"Failed to get workflows for {owner}/{repo}: {str(e)}")
            self._handle_api_error(e)
            raise WorkflowTriggerError(f"Failed to retrieve workflows: {str(e)}")
    
    async def trigger_workflow_dispatch(
        self, 
        owner: str, 
        repo: str, 
        workflow_id: str, 
        inputs: Optional[Dict[str, Any]] = None,
        ref: str = "main"
    ) -> Dict[str, Any]:
        """
        Trigger a workflow dispatch event.
        
        Args:
            owner: Repository owner (username or organization)
            repo: Repository name
            workflow_id: Workflow ID or filename
            inputs: Input parameters for the workflow
            ref: Git reference (branch, tag, or SHA) to run the workflow on
            
        Returns:
            Response from the workflow dispatch trigger
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            WorkflowTriggerError: If the workflow trigger fails
        """
        validate_repository_format(owner, repo)
        
        # Sanitize inputs to prevent injection attacks
        safe_inputs = sanitize_workflow_inputs(inputs) if inputs else {}
        
        try:
            # Note: The GitHub MCP server may not have direct workflow dispatch support
            # This is a placeholder implementation that would need to be adapted
            # based on the actual MCP GitHub server capabilities
            
            logger.info(f"Triggering workflow {workflow_id} in {owner}/{repo} on ref {ref}")
            
            # For now, we'll simulate the workflow dispatch response
            # In a real implementation, this would use the actual MCP function
            response = {
                'message': f'Workflow {workflow_id} triggered successfully',
                'workflow_id': workflow_id,
                'ref': ref,
                'inputs': safe_inputs,
                'triggered_at': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully triggered workflow {workflow_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to trigger workflow {workflow_id} in {owner}/{repo}: {str(e)}")
            self._handle_api_error(e)
            raise WorkflowTriggerError(f"Failed to trigger workflow: {str(e)}")
    
    async def get_workflow_run(self, owner: str, repo: str, run_id: int) -> WorkflowRun:
        """
        Get details of a specific workflow run.
        
        Args:
            owner: Repository owner (username or organization)
            repo: Repository name
            run_id: Workflow run ID
            
        Returns:
            WorkflowRun object with current status and details
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            WorkflowTriggerError: If the operation fails
        """
        validate_repository_format(owner, repo)
        
        try:
            # This would use the actual MCP GitHub function to get workflow run details
            # For now, we'll create a mock response structure
            
            logger.info(f"Getting workflow run {run_id} from {owner}/{repo}")
            
            # Mock workflow run data - in real implementation, this would come from MCP
            run_data = {
                'id': run_id,
                'workflow_id': 12345,
                'name': 'CI Workflow',
                'status': 'in_progress',
                'conclusion': None,
                'created_at': '2024-01-01T10:00:00Z',
                'updated_at': '2024-01-01T10:05:00Z',
                'html_url': f'https://github.com/{owner}/{repo}/actions/runs/{run_id}',
                'run_number': 42,
                'event': 'push',
                'head_branch': 'main',
                'head_sha': 'abc123def456',
                'actor': {'login': 'developer'}
            }
            
            # Convert to WorkflowRun object
            workflow_run = self._parse_workflow_run(run_data, f"{owner}/{repo}")
            
            logger.info(f"Retrieved workflow run {run_id} with status {workflow_run.status.value}")
            return workflow_run
            
        except Exception as e:
            logger.error(f"Failed to get workflow run {run_id} from {owner}/{repo}: {str(e)}")
            self._handle_api_error(e)
            raise WorkflowTriggerError(f"Failed to get workflow run: {str(e)}")
    
    async def get_workflow_run_jobs(self, owner: str, repo: str, run_id: int) -> List[WorkflowJob]:
        """
        Get jobs for a specific workflow run.
        
        Args:
            owner: Repository owner (username or organization)
            repo: Repository name
            run_id: Workflow run ID
            
        Returns:
            List of WorkflowJob objects
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            WorkflowTriggerError: If the operation fails
        """
        validate_repository_format(owner, repo)
        
        try:
            logger.info(f"Getting jobs for workflow run {run_id} from {owner}/{repo}")
            
            # Mock jobs data - in real implementation, this would come from MCP
            jobs_data = [
                {
                    'id': 1001,
                    'name': 'build',
                    'status': 'completed',
                    'conclusion': 'success',
                    'started_at': '2024-01-01T10:01:00Z',
                    'completed_at': '2024-01-01T10:03:00Z',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'status': 'completed',
                            'conclusion': 'success',
                            'number': 1,
                            'started_at': '2024-01-01T10:01:00Z',
                            'completed_at': '2024-01-01T10:01:30Z'
                        },
                        {
                            'name': 'Setup Node.js',
                            'status': 'completed',
                            'conclusion': 'success',
                            'number': 2,
                            'started_at': '2024-01-01T10:01:30Z',
                            'completed_at': '2024-01-01T10:02:00Z'
                        }
                    ]
                },
                {
                    'id': 1002,
                    'name': 'test',
                    'status': 'in_progress',
                    'conclusion': None,
                    'started_at': '2024-01-01T10:03:00Z',
                    'completed_at': None,
                    'steps': [
                        {
                            'name': 'Run tests',
                            'status': 'in_progress',
                            'conclusion': None,
                            'number': 1,
                            'started_at': '2024-01-01T10:03:00Z',
                            'completed_at': None
                        }
                    ]
                }
            ]
            
            # Convert to WorkflowJob objects
            jobs = [self._parse_workflow_job(job_data) for job_data in jobs_data]
            
            logger.info(f"Retrieved {len(jobs)} jobs for workflow run {run_id}")
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get jobs for workflow run {run_id} from {owner}/{repo}: {str(e)}")
            self._handle_api_error(e)
            raise WorkflowTriggerError(f"Failed to get workflow jobs: {str(e)}")
    
    async def get_workflow_run_logs(self, owner: str, repo: str, run_id: int) -> Dict[str, Any]:
        """
        Get logs for a specific workflow run.
        
        Args:
            owner: Repository owner (username or organization)
            repo: Repository name
            run_id: Workflow run ID
            
        Returns:
            Dictionary containing log information and download URLs
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            WorkflowTriggerError: If the operation fails
        """
        validate_repository_format(owner, repo)
        
        try:
            logger.info(f"Getting logs for workflow run {run_id} from {owner}/{repo}")
            
            # Mock logs response - in real implementation, this would come from MCP
            logs_response = {
                'archive_download_url': f'https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs',
                'expires_at': '2024-01-01T11:00:00Z',
                'size_bytes': 1024000
            }
            
            logger.info(f"Retrieved log information for workflow run {run_id}")
            return logs_response
            
        except Exception as e:
            logger.error(f"Failed to get logs for workflow run {run_id} from {owner}/{repo}: {str(e)}")
            self._handle_api_error(e)
            raise WorkflowTriggerError(f"Failed to get workflow logs: {str(e)}")
    
    async def list_workflow_runs(
        self, 
        owner: str, 
        repo: str, 
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        per_page: int = 30
    ) -> List[WorkflowRun]:
        """
        List workflow runs for a repository.
        
        Args:
            owner: Repository owner (username or organization)
            repo: Repository name
            workflow_id: Optional workflow ID to filter by
            status: Optional status to filter by (queued, in_progress, completed)
            per_page: Number of results per page (max 100)
            
        Returns:
            List of WorkflowRun objects
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            WorkflowTriggerError: If the operation fails
        """
        validate_repository_format(owner, repo)
        
        try:
            logger.info(f"Listing workflow runs for {owner}/{repo}")
            
            # Mock workflow runs data - in real implementation, this would come from MCP
            runs_data = [
                {
                    'id': 12345,
                    'workflow_id': 1001,
                    'name': 'CI Workflow',
                    'status': 'completed',
                    'conclusion': 'success',
                    'created_at': '2024-01-01T09:00:00Z',
                    'updated_at': '2024-01-01T09:05:00Z',
                    'html_url': f'https://github.com/{owner}/{repo}/actions/runs/12345',
                    'run_number': 41,
                    'event': 'push',
                    'head_branch': 'main',
                    'head_sha': 'def456abc789',
                    'actor': {'login': 'developer'}
                },
                {
                    'id': 12346,
                    'workflow_id': 1001,
                    'name': 'CI Workflow',
                    'status': 'in_progress',
                    'conclusion': None,
                    'created_at': '2024-01-01T10:00:00Z',
                    'updated_at': '2024-01-01T10:05:00Z',
                    'html_url': f'https://github.com/{owner}/{repo}/actions/runs/12346',
                    'run_number': 42,
                    'event': 'push',
                    'head_branch': 'main',
                    'head_sha': 'abc123def456',
                    'actor': {'login': 'developer'}
                }
            ]
            
            # Filter by status if specified
            if status:
                runs_data = [run for run in runs_data if run.get('status') == status]
            
            # Convert to WorkflowRun objects
            workflow_runs = [
                self._parse_workflow_run(run_data, f"{owner}/{repo}") 
                for run_data in runs_data[:per_page]
            ]
            
            logger.info(f"Retrieved {len(workflow_runs)} workflow runs")
            return workflow_runs
            
        except Exception as e:
            logger.error(f"Failed to list workflow runs for {owner}/{repo}: {str(e)}")
            self._handle_api_error(e)
            raise WorkflowTriggerError(f"Failed to list workflow runs: {str(e)}")
    
    def get_rate_limit_status(self) -> Dict[str, Optional[int]]:
        """
        Get current rate limit status.
        
        Returns:
            Dictionary with rate limit information
        """
        return {
            'remaining': self._rate_limit_remaining,
            'reset': self._rate_limit_reset
        }
    
    async def _call_mcp_function(self, func, **kwargs) -> Any:
        """
        Call an MCP function with error handling and rate limit tracking.
        
        Args:
            func: MCP function to call
            **kwargs: Arguments to pass to the function
            
        Returns:
            Function result
            
        Raises:
            Various exceptions based on the error type
        """
        try:
            # Add timeout to the function call
            result = await asyncio.wait_for(func(**kwargs), timeout=self.default_timeout)
            
            # Update rate limit information if available in response headers
            # This would be implemented based on the actual MCP response format
            
            return result
            
        except asyncio.TimeoutError:
            raise WorkflowTriggerError(f"Operation timed out after {self.default_timeout} seconds")
        except Exception as e:
            self._handle_api_error(e)
            raise
    
    def _handle_api_error(self, error: Exception) -> None:
        """
        Handle API errors and convert them to appropriate exceptions.
        
        Args:
            error: The original exception
            
        Raises:
            AuthenticationError: For authentication-related errors
            RateLimitError: For rate limit errors
        """
        error_str = str(error).lower()
        
        if 'unauthorized' in error_str or 'authentication' in error_str:
            raise AuthenticationError(f"GitHub authentication failed: {str(error)}")
        elif 'rate limit' in error_str or 'forbidden' in error_str:
            raise RateLimitError(f"GitHub API rate limit exceeded: {str(error)}")
        elif 'not found' in error_str:
            raise WorkflowTriggerError(f"Resource not found: {str(error)}")
    
    def _parse_workflow_run(self, run_data: Dict[str, Any], repository: str) -> WorkflowRun:
        """
        Parse GitHub API workflow run data into WorkflowRun object.
        
        Args:
            run_data: Raw workflow run data from GitHub API
            repository: Repository name in owner/repo format
            
        Returns:
            WorkflowRun object
        """
        # Parse status and conclusion
        status = WorkflowStatus(run_data.get('status', 'queued'))
        conclusion = None
        if run_data.get('conclusion'):
            conclusion = WorkflowConclusion(run_data['conclusion'])
        
        # Parse timestamps
        created_at = parse_github_datetime(run_data.get('created_at'))
        updated_at = parse_github_datetime(run_data.get('updated_at'))
        
        return WorkflowRun(
            id=run_data['id'],
            workflow_id=run_data.get('workflow_id', 0),
            workflow_name=run_data.get('name', 'Unknown Workflow'),
            status=status,
            conclusion=conclusion,
            repository=repository,
            branch=run_data.get('head_branch', 'main'),
            commit_sha=run_data.get('head_sha', ''),
            created_at=created_at,
            updated_at=updated_at,
            html_url=run_data.get('html_url'),
            run_number=run_data.get('run_number'),
            event=run_data.get('event'),
            actor=run_data.get('actor', {}).get('login')
        )
    
    def _parse_workflow_job(self, job_data: Dict[str, Any]) -> WorkflowJob:
        """
        Parse GitHub API job data into WorkflowJob object.
        
        Args:
            job_data: Raw job data from GitHub API
            
        Returns:
            WorkflowJob object
        """
        # Parse status and conclusion
        status = JobStatus(job_data.get('status', 'queued'))
        conclusion = None
        if job_data.get('conclusion'):
            conclusion = WorkflowConclusion(job_data['conclusion'])
        
        # Parse timestamps
        started_at = parse_github_datetime(job_data.get('started_at'))
        completed_at = parse_github_datetime(job_data.get('completed_at'))
        
        # Parse steps
        steps = []
        for step_data in job_data.get('steps', []):
            step = self._parse_workflow_step(step_data)
            steps.append(step)
        
        return WorkflowJob(
            id=job_data['id'],
            name=job_data.get('name', 'Unknown Job'),
            status=status,
            conclusion=conclusion,
            started_at=started_at,
            completed_at=completed_at,
            steps=steps
        )
    
    def _parse_workflow_step(self, step_data: Dict[str, Any]) -> WorkflowStep:
        """
        Parse GitHub API step data into WorkflowStep object.
        
        Args:
            step_data: Raw step data from GitHub API
            
        Returns:
            WorkflowStep object
        """
        # Parse status and conclusion
        status = StepStatus(step_data.get('status', 'queued'))
        conclusion = None
        if step_data.get('conclusion'):
            conclusion = WorkflowConclusion(step_data['conclusion'])
        
        # Parse timestamps
        started_at = parse_github_datetime(step_data.get('started_at'))
        completed_at = parse_github_datetime(step_data.get('completed_at'))
        
        return WorkflowStep(
            name=step_data.get('name', 'Unknown Step'),
            status=status,
            conclusion=conclusion,
            number=step_data.get('number', 0),
            started_at=started_at,
            completed_at=completed_at
        )
    
    async def cancel_workflow_run(self, owner: str, repo: str, run_id: int) -> bool:
        """
        Cancel a running workflow.
        
        Args:
            owner: Repository owner (username or organization)
            repo: Repository name
            run_id: Workflow run ID to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
            
        Raises:
            AuthenticationError: If GitHub authentication fails
            RateLimitError: If API rate limit is exceeded
            WorkflowTriggerError: If cancellation request fails
        """
        try:
            validate_repository_format(owner, repo)
            
            logger.info(f"Cancelling workflow run {run_id} for {owner}/{repo}")
            
            # Note: This would use the actual MCP GitHub integration
            # For now, we'll simulate the cancellation
            # In a real implementation, this would call:
            # result = await mcp_github_cancel_workflow_run(owner, repo, run_id)
            
            # Simulate successful cancellation
            logger.info(f"Successfully cancelled workflow run {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow run {run_id}: {str(e)}")
            raise WorkflowTriggerError(f"Failed to cancel workflow run: {str(e)}")