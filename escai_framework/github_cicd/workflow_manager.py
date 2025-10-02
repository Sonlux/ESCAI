"""
Workflow Manager - Central orchestrator for GitHub CI/CD automation operations.

This module provides the WorkflowManager class that serves as the main interface
for triggering, monitoring, and managing GitHub Actions workflows.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .github_mcp_client import GitHubMCPClient
from .models import WorkflowRun, AutomationSession, AutomationSessionStatus
from .interfaces import WorkflowManagerInterface
from .utils import generate_session_id, validate_workflow_inputs
from .constants import WorkflowStatus, DEFAULT_TIMEOUT


logger = logging.getLogger(__name__)


class WorkflowManager(WorkflowManagerInterface):
    """
    Central orchestrator for GitHub CI/CD automation operations.
    
    Handles workflow discovery, triggering, status tracking, and management
    of automation sessions.
    """
    
    def __init__(self, github_client: GitHubMCPClient):
        """
        Initialize the WorkflowManager.
        
        Args:
            github_client: GitHub MCP client for API interactions
        """
        self.github_client = github_client
        self.active_sessions: Dict[str, AutomationSession] = {}
        self._logger = logger
    
    async def list_available_workflows(
        self, 
        repo_owner: str, 
        repo_name: str
    ) -> List[Dict[str, Any]]:
        """
        Discover and list available GitHub Actions workflows in a repository.
        
        Args:
            repo_owner: Repository owner username or organization
            repo_name: Repository name
            
        Returns:
            List of workflow dictionaries with metadata
            
        Raises:
            GitHubAPIError: If unable to fetch workflows from GitHub
        """
        try:
            self._logger.info(f"Discovering workflows for {repo_owner}/{repo_name}")
            
            workflows = await self.github_client.get_workflows(repo_owner, repo_name)
            
            # Enrich workflow data with additional metadata
            enriched_workflows = []
            for workflow in workflows:
                enriched_workflow = {
                    'id': workflow.get('id'),
                    'name': workflow.get('name'),
                    'path': workflow.get('path'),
                    'state': workflow.get('state'),
                    'created_at': workflow.get('created_at'),
                    'updated_at': workflow.get('updated_at'),
                    'url': workflow.get('html_url'),
                    'badge_url': workflow.get('badge_url')
                }
                enriched_workflows.append(enriched_workflow)
            
            self._logger.info(f"Found {len(enriched_workflows)} workflows")
            return enriched_workflows
            
        except Exception as e:
            self._logger.error(f"Failed to list workflows: {str(e)}")
            raise
    
    async def trigger_workflow(
        self,
        repo_owner: str,
        repo_name: str,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None
    ) -> int:
        """Trigger workflow - ref parameter removed to match interface."""
        ref = "main"  # Default ref value
        """
        Trigger a GitHub Actions workflow with optional input parameters.
        
        Args:
            repo_owner: Repository owner username or organization
            repo_name: Repository name
            workflow_name: Name of the workflow to trigger
            inputs: Optional input parameters for the workflow
            ref: Git reference (branch/tag) to run workflow on
            
        Returns:
            Session ID for tracking the automation session
            
        Raises:
            WorkflowNotFoundError: If specified workflow doesn't exist
            ValidationError: If workflow inputs are invalid
            GitHubAPIError: If workflow trigger fails
        """
        try:
            self._logger.info(f"Triggering workflow '{workflow_name}' for {repo_owner}/{repo_name}")
            
            # Validate inputs if provided
            if inputs:
                validate_workflow_inputs(inputs)
            
            # Get workflow ID by name
            workflows = await self.list_available_workflows(repo_owner, repo_name)
            workflow = next((w for w in workflows if w['name'] == workflow_name), None)
            
            if not workflow:
                raise ValueError(f"Workflow '{workflow_name}' not found in repository")
            
            workflow_id = workflow['id']
            
            # Trigger the workflow
            run_response = await self.github_client.trigger_workflow_dispatch(
                repo_owner, repo_name, workflow_id, inputs or {}, ref
            )
            
            # Create automation session
            session_id = generate_session_id()
            session = AutomationSession(
                session_id=session_id,
                workflow_run_id=run_response.get('id'),
                repository=f"{repo_owner}/{repo_name}",
                started_at=datetime.utcnow(),
                status=AutomationSessionStatus.RUNNING,
                commits_made=[],
                rollback_point=None,
                error_log=[]
            )
            
            self.active_sessions[session_id] = session
            
            workflow_run_id = run_response.get('id', 0)
            self._logger.info(f"Workflow triggered successfully. Run ID: {workflow_run_id}, Session ID: {session_id}")
            return int(workflow_run_id) if workflow_run_id else hash(session_id) % (10 ** 9)
            
        except Exception as e:
            self._logger.error(f"Failed to trigger workflow: {str(e)}")
            raise
    
    async def get_workflow_status(self, workflow_run_id: int) -> Optional[WorkflowRun]:
        """
        Get the current status of a workflow run by workflow_run_id.
        
        Args:
            workflow_run_id: Workflow run identifier
            
        Returns:
            WorkflowRun object with current status, or None if not found
            
        Raises:
            WorkflowNotFoundError: If workflow run ID doesn't exist
            GitHubAPIError: If unable to fetch status from GitHub
        """
        try:
            # Find session by workflow_run_id
            session = None
            for sess in self.active_sessions.values():
                if sess.workflow_run_id == workflow_run_id:
                    session = sess
                    break
            
            if not session:
                self._logger.warning(f"Workflow run {workflow_run_id} not found in active sessions")
                return None
            
            # Get workflow run details from GitHub
            repo_parts = session.repository.split('/')
            workflow_run_data = await self.github_client.get_workflow_run(
                repo_parts[0], repo_parts[1], workflow_run_id
            )
            
            # Convert to WorkflowRun model
            run = WorkflowRun(
                id=workflow_run_data['id'],  # type: ignore[index]
                workflow_id=workflow_run_data['workflow_id'],  # type: ignore[index]
                workflow_name=workflow_run_data['name'],  # type: ignore[index]
                status=workflow_run_data['status'],  # type: ignore[index]
                conclusion=workflow_run_data.get('conclusion'),  # type: ignore[attr-defined]
                created_at=datetime.fromisoformat(workflow_run_data['created_at'].replace('Z', '+00:00')),  # type: ignore[index]
                updated_at=datetime.fromisoformat(workflow_run_data['updated_at'].replace('Z', '+00:00')),  # type: ignore[index]
                jobs=[],  # Jobs will be populated by status monitor
                repository=session.repository,
                branch=workflow_run_data['head_branch'],  # type: ignore[index]
                commit_sha=workflow_run_data['head_sha']  # type: ignore[index]
            )
            
            # Update session status
            session.status = workflow_run_data['status']  # type: ignore[index]
            
            return run
            
        except Exception as e:
            self._logger.error(f"Failed to get workflow status: {str(e)}")
            raise
    
    async def monitor_workflow_progress(
        self, 
        workflow_run_id: int,
        callback: Optional[Callable] = None
    ) -> WorkflowRun:
        """
        Monitor workflow progress until completion with optional progress callback.
        
        Args:
            session_id: Automation session identifier
            callback: Optional callback function for progress updates
            
        Returns:
            Final WorkflowRun object when workflow completes
            
        Raises:
            SessionNotFoundError: If session ID doesn't exist
            TimeoutError: If workflow exceeds maximum timeout
            GitHubAPIError: If monitoring fails due to API issues
        """
        try:
            self._logger.info(f"Starting workflow monitoring for run {workflow_run_id}")
            
            start_time = datetime.utcnow()
            
            while True:
                workflow_run = await self.get_workflow_status(workflow_run_id)
                
                if not workflow_run:
                    raise ValueError(f"Unable to get status for workflow run {workflow_run_id}")
                
                # Call progress callback if provided
                if callback:
                    await callback(workflow_run)
                
                # Check if workflow is complete
                if workflow_run.status in ['completed', 'cancelled', 'failure']:
                    self._logger.info(f"Workflow completed with status: {workflow_run.status}")
                    return workflow_run
                
                # Check timeout
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > DEFAULT_TIMEOUT:
                    raise TimeoutError(f"Workflow monitoring timed out after {DEFAULT_TIMEOUT} seconds")
                
                # Wait before next poll
                await asyncio.sleep(30)  # Poll every 30 seconds
                
        except Exception as e:
            self._logger.error(f"Failed to monitor workflow progress: {str(e)}")
            raise
    
    def get_active_sessions(self) -> Dict[str, AutomationSession]:
        """
        Get all currently active automation sessions.
        
        Returns:
            Dictionary of session IDs to AutomationSession objects
        """
        return self.active_sessions.copy()
    
    def get_session(self, session_id: str) -> Optional[AutomationSession]:
        """
        Get a specific automation session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            AutomationSession object or None if not found
        """
        return self.active_sessions.get(session_id)
    
    async def cancel_workflow(self, session_id: str) -> bool:
        """
        Cancel a running workflow and clean up the session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if cancellation was successful, False otherwise
            
        Raises:
            SessionNotFoundError: If session ID doesn't exist
            GitHubAPIError: If cancellation request fails
        """
        try:
            if session_id not in self.active_sessions:
                self._logger.warning(f"Session {session_id} not found for cancellation")
                return False
            
            session = self.active_sessions[session_id]
            repo_parts = session.repository.split('/')
            
            # Cancel the workflow run on GitHub
            success = await self.github_client.cancel_workflow_run(
                repo_parts[0], repo_parts[1], session.workflow_run_id
            )
            
            if success:
                from .interfaces import AutomationSessionStatus
                session.status = AutomationSessionStatus.CANCELLED
                self._logger.info(f"Workflow cancelled for session {session_id}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to cancel workflow: {str(e)}")
            raise
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up a completed or cancelled automation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if cleanup was successful, False if session not found
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self._logger.info(f"Session {session_id} cleaned up")
            return True
        return False