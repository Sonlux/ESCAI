"""
Commit management system for GitHub CI/CD automation.

This module provides automated commit and push operations with contextual
commit message generation, retry logic, and workflow context integration.
"""

import os
import subprocess
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .models import AutomationSession, WorkflowRun, CommitSHA
from .interfaces import GitHubMCPClientInterface
from .utils import format_timestamp, sanitize_commit_message


@dataclass
class CommitContext:
    """
    Context information for generating commit messages.
    
    Attributes:
        workflow_run: Associated workflow run information
        session: Automation session details
        files_changed: List of files that were modified
        operation_type: Type of operation (e.g., 'automation', 'rollback', 'fix')
        custom_message: Optional custom message to include
    """
    workflow_run: Optional[WorkflowRun] = None
    session: Optional[AutomationSession] = None
    files_changed: List[str] = None
    operation_type: str = "automation"
    custom_message: Optional[str] = None

    def __post_init__(self):
        if self.files_changed is None:
            self.files_changed = []


@dataclass
class CommitResult:
    """
    Result of a commit operation.
    
    Attributes:
        success: Whether the commit was successful
        commit_sha: SHA of the created commit (if successful)
        message: Commit message that was used
        files_committed: List of files that were committed
        error: Error message if the commit failed
        timestamp: When the commit was made
    """
    success: bool
    commit_sha: Optional[str] = None
    message: Optional[str] = None
    files_committed: List[str] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.files_committed is None:
            self.files_committed = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PushResult:
    """
    Result of a push operation.
    
    Attributes:
        success: Whether the push was successful
        branch: Branch that was pushed to
        commits_pushed: Number of commits pushed
        error: Error message if the push failed
        retry_count: Number of retries attempted
        timestamp: When the push was completed
    """
    success: bool
    branch: str
    commits_pushed: int = 0
    error: Optional[str] = None
    retry_count: int = 0
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CommitManager:
    """
    Manages automated commit and push operations for GitHub CI/CD workflows.
    
    This class provides functionality for:
    - Automatic file staging and commit operations
    - Contextual commit message generation based on workflow information
    - Push operations with retry logic and error handling
    - Commit history tracking and workflow context integration
    """

    def __init__(
        self,
        github_client: GitHubMCPClientInterface,
        repository_path: str = ".",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_stage: bool = True
    ):
        """
        Initialize the CommitManager.
        
        Args:
            github_client: GitHub MCP client for API operations
            repository_path: Path to the local git repository
            max_retries: Maximum number of retry attempts for push operations
            retry_delay: Initial delay between retries (exponential backoff)
            auto_stage: Whether to automatically stage files before committing
        """
        self.github_client = github_client
        self.repository_path = Path(repository_path).resolve()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_stage = auto_stage
        self._commit_history: List[CommitResult] = []

    def auto_commit(
        self,
        message: str,
        files: Optional[List[str]] = None,
        context: Optional[CommitContext] = None
    ) -> CommitResult:
        """
        Automatically commit changes with optional file specification.
        
        Args:
            message: Base commit message
            files: Specific files to commit (None for all staged files)
            context: Additional context for commit message generation
            
        Returns:
            CommitResult with details of the commit operation
        """
        try:
            # Generate contextual commit message
            final_message = self._generate_commit_message(message, context)
            
            # Stage files if auto_stage is enabled
            if self.auto_stage:
                staged_files = self._stage_files(files)
            else:
                staged_files = files or []
            
            # Perform the commit
            commit_sha = self._execute_commit(final_message)
            
            result = CommitResult(
                success=True,
                commit_sha=commit_sha,
                message=final_message,
                files_committed=staged_files
            )
            
            # Track commit in history
            self._commit_history.append(result)
            
            # Update session if provided
            if context and context.session:
                context.session.add_commit(commit_sha)
            
            return result
            
        except Exception as e:
            error_msg = f"Commit failed: {str(e)}"
            result = CommitResult(
                success=False,
                error=error_msg,
                message=message
            )
            self._commit_history.append(result)
            return result

    def create_workflow_commit(
        self,
        workflow_context: WorkflowRun,
        changes: List[str],
        session: Optional[AutomationSession] = None,
        custom_message: Optional[str] = None
    ) -> CommitResult:
        """
        Create a commit with workflow-specific context and messaging.
        
        Args:
            workflow_context: Workflow run information for context
            changes: List of files that were changed
            session: Optional automation session for tracking
            custom_message: Optional custom message to include
            
        Returns:
            CommitResult with details of the commit operation
        """
        context = CommitContext(
            workflow_run=workflow_context,
            session=session,
            files_changed=changes,
            operation_type="workflow_automation",
            custom_message=custom_message
        )
        
        base_message = f"Automated changes from workflow: {workflow_context.workflow_name}"
        if custom_message:
            base_message = f"{base_message} - {custom_message}"
        
        return self.auto_commit(base_message, changes, context)

    def push_changes(
        self,
        branch: str = "main",
        force: bool = False
    ) -> PushResult:
        """
        Push committed changes to the remote repository with retry logic.
        
        Args:
            branch: Target branch for the push operation
            force: Whether to force push (use with caution)
            
        Returns:
            PushResult with details of the push operation
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Get number of commits to push
                commits_to_push = self._count_unpushed_commits(branch)
                
                # Execute push command
                self._execute_push(branch, force)
                
                return PushResult(
                    success=True,
                    branch=branch,
                    commits_pushed=commits_to_push,
                    retry_count=retry_count
                )
                
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                
                if retry_count <= self.max_retries:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    time.sleep(delay)
                else:
                    break
        
        return PushResult(
            success=False,
            branch=branch,
            error=f"Push failed after {retry_count} retries: {last_error}",
            retry_count=retry_count
        )

    def get_commit_history(self, limit: int = 10) -> List[CommitResult]:
        """
        Get the commit history from this session.
        
        Args:
            limit: Maximum number of commits to return
            
        Returns:
            List of recent CommitResult objects
        """
        return self._commit_history[-limit:] if limit > 0 else self._commit_history

    def get_repository_commit_history(
        self,
        limit: int = 10,
        branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        Get commit history from the git repository.
        
        Args:
            limit: Maximum number of commits to return
            branch: Branch to get history from
            
        Returns:
            List of commit information dictionaries
        """
        try:
            cmd = [
                "git", "log", f"--max-count={limit}", "--pretty=format:%H|%s|%an|%ad",
                "--date=iso", branch
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 3)
                    if len(parts) == 4:
                        commits.append({
                            'sha': parts[0],
                            'message': parts[1],
                            'author': parts[2],
                            'date': parts[3]
                        })
            
            return commits
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get commit history: {e}")

    def get_current_commit_sha(self) -> str:
        """
        Get the SHA of the current HEAD commit.
        
        Returns:
            SHA of the current commit
            
        Raises:
            RuntimeError: If unable to get current commit SHA
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get current commit SHA: {e}")

    def has_uncommitted_changes(self) -> bool:
        """
        Check if there are uncommitted changes in the repository.
        
        Returns:
            True if there are uncommitted changes, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def get_modified_files(self) -> List[str]:
        """
        Get list of modified files in the repository.
        
        Returns:
            List of file paths that have been modified
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            modified_files = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    # Parse git status output (format: XY filename)
                    status = line[:2]
                    filename = line[3:]
                    if status.strip():  # Any status indicates modification
                        modified_files.append(filename)
            
            return modified_files
            
        except subprocess.CalledProcessError:
            return []

    def _generate_commit_message(
        self,
        base_message: str,
        context: Optional[CommitContext] = None
    ) -> str:
        """
        Generate a contextual commit message based on workflow information.
        
        Args:
            base_message: Base commit message
            context: Additional context for message generation
            
        Returns:
            Enhanced commit message with context
        """
        message_parts = [base_message]
        
        if context:
            # Add workflow context
            if context.workflow_run:
                workflow_info = f"Workflow: {context.workflow_run.workflow_name}"
                if context.workflow_run.run_number:
                    workflow_info += f" (#{context.workflow_run.run_number})"
                message_parts.append(workflow_info)
            
            # Add session context
            if context.session:
                session_info = f"Session: {context.session.session_id}"
                message_parts.append(session_info)
            
            # Add file information
            if context.files_changed:
                if len(context.files_changed) <= 3:
                    files_info = f"Files: {', '.join(context.files_changed)}"
                else:
                    files_info = f"Files: {len(context.files_changed)} files modified"
                message_parts.append(files_info)
            
            # Add custom message
            if context.custom_message:
                message_parts.append(context.custom_message)
        
        # Add timestamp
        timestamp = format_timestamp(datetime.now())
        message_parts.append(f"Timestamp: {timestamp}")
        
        # Join parts with newlines for multi-line commit message
        full_message = message_parts[0]
        if len(message_parts) > 1:
            full_message += "\n\n" + "\n".join(message_parts[1:])
        
        return sanitize_commit_message(full_message)

    def _stage_files(self, files: Optional[List[str]] = None) -> List[str]:
        """
        Stage files for commit.
        
        Args:
            files: Specific files to stage (None for all modified files)
            
        Returns:
            List of files that were staged
            
        Raises:
            RuntimeError: If staging fails
        """
        try:
            if files:
                # Stage specific files
                for file_path in files:
                    subprocess.run(
                        ["git", "add", file_path],
                        cwd=self.repository_path,
                        check=True
                    )
                return files
            else:
                # Stage all modified files
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=self.repository_path,
                    check=True
                )
                return self.get_modified_files()
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to stage files: {e}")

    def _execute_commit(self, message: str) -> str:
        """
        Execute the git commit command.
        
        Args:
            message: Commit message
            
        Returns:
            SHA of the created commit
            
        Raises:
            RuntimeError: If commit fails
        """
        try:
            # Perform the commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repository_path,
                check=True
            )
            
            # Get the commit SHA
            return self.get_current_commit_sha()
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to commit: {e}")

    def _execute_push(self, branch: str, force: bool = False) -> None:
        """
        Execute the git push command.
        
        Args:
            branch: Target branch
            force: Whether to force push
            
        Raises:
            RuntimeError: If push fails
        """
        try:
            cmd = ["git", "push", "origin", branch]
            if force:
                cmd.append("--force")
            
            subprocess.run(
                cmd,
                cwd=self.repository_path,
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to push to {branch}: {e}")

    def _count_unpushed_commits(self, branch: str) -> int:
        """
        Count the number of commits that haven't been pushed.
        
        Args:
            branch: Branch to check
            
        Returns:
            Number of unpushed commits
        """
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", f"origin/{branch}..HEAD"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return 0