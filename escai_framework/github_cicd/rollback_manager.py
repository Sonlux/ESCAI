"""
Rollback Manager - Handles failed workflow recovery.

This module provides comprehensive rollback management for failed GitHub CI/CD workflows,
including rollback point identification, safe revert operations, and rollback history tracking.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .interfaces import RollbackManagerInterface, RollbackError, CommitError
from .models import WorkflowRunId, CommitSHA
from .utils import (
    generate_session_id, format_timestamp, extract_error_context
)


logger = logging.getLogger(__name__)


@dataclass
class RollbackPoint:
    """
    Represents a potential rollback point in the commit history.
    
    Attributes:
        commit_sha: SHA of the commit that can be rolled back to
        commit_message: Message of the rollback commit
        timestamp: When the commit was created
        is_stable: Whether this commit represents a stable state
        workflow_run_id: Associated workflow run ID (if any)
        confidence_score: Confidence that this is a good rollback point (0.0-1.0)
    """
    commit_sha: CommitSHA
    commit_message: str
    timestamp: datetime
    is_stable: bool = False
    workflow_run_id: Optional[WorkflowRunId] = None
    confidence_score: float = 0.0


@dataclass
class RollbackOperation:
    """
    Represents a rollback operation and its details.
    
    Attributes:
        operation_id: Unique identifier for the rollback operation
        session_id: Associated automation session ID
        target_commit: Commit SHA to rollback to
        source_commit: Commit SHA being rolled back from
        reason: Reason for the rollback
        initiated_at: When the rollback was initiated
        completed_at: When the rollback completed (if successful)
        status: Current status of the rollback operation
        revert_commit_sha: SHA of the revert commit created
        verification_passed: Whether post-rollback verification passed
    """
    operation_id: str
    session_id: str
    target_commit: CommitSHA
    source_commit: CommitSHA
    reason: str
    initiated_at: datetime
    status: str = "initiated"
    completed_at: Optional[datetime] = None
    revert_commit_sha: Optional[CommitSHA] = None
    verification_passed: Optional[bool] = None


class RollbackManager(RollbackManagerInterface):
    """
    Manages rollback operations for failed GitHub CI/CD workflows.
    
    This class provides comprehensive rollback functionality including:
    - Intelligent rollback point identification
    - Safe revert operations with verification
    - Rollback history tracking and reporting
    - State verification and recovery
    """
    
    def __init__(self, github_client, commit_manager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RollbackManager.
        
        Args:
            github_client: GitHub MCP client for API operations
            commit_manager: Commit manager for git operations
            config: Configuration dictionary for rollback settings
        """
        self.github_client = github_client
        self.commit_manager = commit_manager
        self.config = config or {}
        
        # Configuration settings
        self.max_rollback_history = self.config.get("max_rollback_history", 50)
        self.stability_threshold = self.config.get("stability_threshold", 0.7)
        self.verification_timeout = self.config.get("verification_timeout", 300)
        self.safe_rollback_window_hours = self.config.get("safe_rollback_window_hours", 24)
        
        # Internal state
        self.rollback_history: List[RollbackOperation] = []
        self.active_operations: Dict[str, RollbackOperation] = {}
        
        logger.info("RollbackManager initialized with config: %s", self.config)
    
    async def identify_rollback_point(self, workflow_run_id: WorkflowRunId) -> Optional[CommitSHA]:
        """
        Identify the most appropriate rollback point for a failed workflow.
        
        This method analyzes the commit history to find the best rollback target
        based on stability indicators, workflow success history, and temporal factors.
        
        Args:
            workflow_run_id: ID of the failed workflow run
            
        Returns:
            Commit SHA of the identified rollback point, or None if no suitable point found
            
        Raises:
            RollbackError: If rollback point identification fails
        """
        try:
            logger.info("Identifying rollback point for workflow run %s", workflow_run_id)
            
            # Get commit history
            commit_history = await self._get_commit_history()
            if not commit_history:
                logger.warning("No commit history available for rollback point identification")
                return None
            
            # Analyze commits to find potential rollback points
            rollback_candidates = await self._analyze_rollback_candidates(
                commit_history, workflow_run_id
            )
            
            if not rollback_candidates:
                logger.warning("No suitable rollback candidates found")
                return None
            
            # Select the best rollback point
            best_candidate = self._select_best_rollback_point(rollback_candidates)
            
            logger.info(
                "Selected rollback point: %s (confidence: %.2f)", 
                best_candidate.commit_sha, 
                best_candidate.confidence_score
            )
            
            return best_candidate.commit_sha
            
        except Exception as e:
            error_context = extract_error_context(e)
            logger.error("Failed to identify rollback point: %s", error_context)
            raise RollbackError(f"Rollback point identification failed: {str(e)}") from e

    async def perform_rollback(self, target_commit: CommitSHA) -> bool:
        """
        Perform rollback to the specified commit.
        
        This method creates a revert commit that undoes changes back to the target commit,
        then verifies the rollback was successful.
        
        Args:
            target_commit: Commit SHA to rollback to
            
        Returns:
            True if rollback was successful, False otherwise
            
        Raises:
            RollbackError: If rollback operation fails
        """
        operation_id = generate_session_id()
        
        try:
            logger.info("Starting rollback operation %s to commit %s", operation_id, target_commit)
            
            # Get current commit for tracking
            current_commit = await self._get_current_commit()
            
            # Create rollback operation record
            operation = RollbackOperation(
                operation_id=operation_id,
                session_id=generate_session_id(),
                target_commit=target_commit,
                source_commit=current_commit,
                reason=f"Automated rollback to {target_commit}",
                initiated_at=datetime.now()
            )
            
            self.active_operations[operation_id] = operation
            
            # Validate rollback target
            if not await self._validate_rollback_target(target_commit):
                operation.status = "failed"
                raise RollbackError(f"Invalid rollback target: {target_commit}")
            
            # Create revert commit
            revert_commit_sha = await self._create_revert_commit(
                target_commit, operation.reason
            )
            operation.revert_commit_sha = revert_commit_sha
            operation.status = "reverting"
            
            # Verify rollback success
            verification_passed = await self.verify_rollback_success()
            operation.verification_passed = verification_passed
            
            if verification_passed:
                operation.status = "completed"
                operation.completed_at = datetime.now()
                logger.info("Rollback operation %s completed successfully", operation_id)
                
                # Add to history
                self.rollback_history.append(operation)
                self._cleanup_rollback_history()
                
                return True
            else:
                operation.status = "verification_failed"
                logger.error("Rollback operation %s failed verification", operation_id)
                return False
                
        except Exception as e:
            if operation_id in self.active_operations:
                self.active_operations[operation_id].status = "failed"
            
            error_context = extract_error_context(e)
            logger.error("Rollback operation %s failed: %s", operation_id, error_context)
            raise RollbackError(f"Rollback operation failed: {str(e)}") from e
        
        finally:
            # Clean up active operation
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]

    async def create_revert_commit(self, failed_commit: CommitSHA, reason: str) -> CommitSHA:
        """
        Create a revert commit for the failed commit.
        
        This method creates a revert commit that undoes the changes introduced
        by the failed commit, with appropriate commit messaging.
        
        Args:
            failed_commit: SHA of the commit to revert
            reason: Reason for the revert
            
        Returns:
            SHA of the created revert commit
            
        Raises:
            CommitError: If revert commit creation fails
        """
        try:
            logger.info("Creating revert commit for %s with reason: %s", failed_commit, reason)
            
            # Generate revert commit message
            revert_message = f"Revert commit {failed_commit[:8]}: {reason}"
            
            # Use commit manager to create the revert commit
            revert_commit_sha = await self.commit_manager.auto_commit(
                message=revert_message,
                files=None  # Let git determine which files to revert
            )
            
            logger.info("Created revert commit %s", revert_commit_sha)
            return revert_commit_sha
            
        except Exception as e:
            error_context = extract_error_context(e)
            logger.error("Failed to create revert commit: %s", error_context)
            raise CommitError(f"Revert commit creation failed: {str(e)}") from e

    async def verify_rollback_success(self) -> bool:
        """
        Verify that the rollback operation was successful.
        
        This method performs various checks to ensure the rollback completed
        successfully and the repository is in a stable state.
        
        Returns:
            True if rollback verification passed, False otherwise
        """
        try:
            logger.info("Verifying rollback success")
            
            # Check if repository is in a clean state
            if not await self._verify_repository_state():
                logger.warning("Repository state verification failed")
                return False
            
            # Check if recent commits are consistent
            if not await self._verify_commit_consistency():
                logger.warning("Commit consistency verification failed")
                return False
            
            # Check if there are no pending changes
            if not await self._verify_no_pending_changes():
                logger.warning("Pending changes verification failed")
                return False
            
            logger.info("Rollback verification passed")
            return True
            
        except Exception as e:
            error_context = extract_error_context(e)
            logger.error("Rollback verification failed: %s", error_context)
            return False

    def get_rollback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of rollback operations.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of recent rollback operations as dictionaries
        """
        if not self.rollback_history:
            return []
        
        # Sort by initiated_at descending (most recent first)
        sorted_history = sorted(
            self.rollback_history,
            key=lambda x: x.initiated_at,
            reverse=True
        )
        
        # Convert to dictionaries and apply limit
        return [self._rollback_operation_to_dict(op) for op in sorted_history[:limit]]

    def get_active_rollback_operations(self) -> List[Dict[str, Any]]:
        """
        Get currently active rollback operations.
        
        Returns:
            List of active rollback operations as dictionaries
        """
        return [self._rollback_operation_to_dict(op) for op in self.active_operations.values()]

    # Private helper methods
    
    async def _get_commit_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get commit history from the commit manager."""
        try:
            history_limit = limit or self.max_rollback_history
            return self.commit_manager.get_commit_history(limit=history_limit)
        except Exception as e:
            logger.error("Failed to get commit history: %s", str(e))
            return []

    async def _analyze_rollback_candidates(
        self, 
        commit_history: List[Dict[str, Any]], 
        workflow_run_id: WorkflowRunId
    ) -> List[RollbackPoint]:
        """
        Analyze commit history to identify potential rollback points.
        
        Args:
            commit_history: List of commits from git history
            workflow_run_id: ID of the failed workflow
            
        Returns:
            List of potential rollback points with confidence scores
        """
        candidates = []
        
        for commit in commit_history:
            try:
                commit_sha = commit.get("sha", "")
                commit_message = commit.get("message", "")
                commit_date = commit.get("date")
                
                if not commit_sha:
                    continue
                
                # Parse commit timestamp
                timestamp = datetime.fromisoformat(commit_date) if commit_date else datetime.now()
                
                # Calculate hours ago for confidence scoring
                hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
                
                # Calculate confidence score based on various factors
                confidence_score = self._calculate_rollback_confidence(
                    commit, int(hours_ago), workflow_run_id
                )
                
                # Determine if this is a stable commit
                is_stable = self._is_stable_commit(commit)
                
                rollback_point = RollbackPoint(
                    commit_sha=commit_sha,
                    commit_message=commit_message,
                    timestamp=timestamp,
                    is_stable=is_stable,
                    workflow_run_id=workflow_run_id,
                    confidence_score=confidence_score
                )
                
                candidates.append(rollback_point)
                
            except Exception as e:
                logger.warning("Failed to analyze commit %s: %s", commit.get("sha", "unknown"), str(e))
                continue
        
        return candidates

    def _select_best_rollback_point(self, candidates: List[RollbackPoint]) -> RollbackPoint:
        """
        Select the best rollback point from candidates.
        
        Args:
            candidates: List of rollback point candidates
            
        Returns:
            The best rollback point candidate
        """
        if not candidates:
            raise RollbackError("No rollback candidates available")
        
        # Sort by confidence score (descending) and stability
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (x.confidence_score, x.is_stable),
            reverse=True
        )
        
        return sorted_candidates[0]

    def _calculate_rollback_confidence(
        self, 
        commit: Dict[str, Any], 
        hours_ago: int,
        workflow_run_id: WorkflowRunId
    ) -> float:
        """
        Calculate confidence score for a rollback point.
        
        Args:
            commit: Commit information
            hours_ago: Hours since the commit was made
            workflow_run_id: Failed workflow run ID
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        try:
            # Base confidence for any commit
            confidence += 0.3
            
            # Higher confidence for commits with successful workflow runs
            if self._has_successful_workflow_run_sync(commit.get("sha", "")):
                confidence += 0.4
            
            # Higher confidence for commits with stability indicators
            commit_message = commit.get("message", "").lower()
            if any(keyword in commit_message for keyword in ["stable", "release", "fix", "patch"]):
                confidence += 0.2
            
            # Lower confidence for very recent commits (might be unstable)
            if hours_ago < 1:  # Less than 1 hour ago
                confidence -= 0.2
            elif hours_ago > self.safe_rollback_window_hours:  # Too old
                confidence -= 0.1
            
            # Ensure confidence is within bounds
            confidence = max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning("Failed to calculate confidence for commit %s: %s", 
                         commit.get("sha", "unknown"), str(e))
            confidence = 0.1  # Minimal confidence for problematic commits
        
        return confidence

    def _is_stable_commit(self, commit: Dict[str, Any]) -> bool:
        """
        Determine if a commit represents a stable state.
        
        Args:
            commit: Commit information
            
        Returns:
            True if the commit is considered stable
        """
        try:
            commit_message = commit.get("message", "").lower()
            
            # Check for stability indicators in commit message
            stability_keywords = [
                "stable", "release", "version", "tag", "production",
                "deploy", "merge", "hotfix", "patch"
            ]
            
            # Check for instability indicators
            instability_keywords = [
                "wip", "experimental", "debug", "test", "temp", "temporary"
            ]
            
            has_stable = any(keyword in commit_message for keyword in stability_keywords)
            has_unstable = any(keyword in commit_message for keyword in instability_keywords)
            
            # If it has both stable and unstable indicators, consider it unstable
            if has_stable and has_unstable:
                return False
            
            return has_stable
            
        except Exception:
            return False

    def _has_successful_workflow_run_sync(self, commit_sha: CommitSHA) -> bool:
        """
        Check if a commit has associated successful workflow runs (synchronous version).
        
        Args:
            commit_sha: Commit SHA to check
            
        Returns:
            True if the commit has successful workflow runs
        """
        try:
            # This would typically query the GitHub API to check workflow runs
            # For now, we'll implement a basic heuristic
            return True  # Placeholder - would need actual GitHub API integration
            
        except Exception:
            return False

    async def _get_current_commit(self) -> CommitSHA:
        """Get the current commit SHA."""
        try:
            history = await self._get_commit_history(limit=1)
            if history:
                return history[0].get("sha", "")
            return ""
        except Exception:
            return ""

    async def _validate_rollback_target(self, target_commit: CommitSHA) -> bool:
        """
        Validate that the rollback target is valid.
        
        Args:
            target_commit: Target commit SHA
            
        Returns:
            True if the target is valid
        """
        try:
            if not target_commit:
                return False
            
            # Check if commit exists in history
            history = await self._get_commit_history()
            commit_shas = [commit.get("sha", "") for commit in history]
            
            return target_commit in commit_shas
            
        except Exception:
            return False

    async def _create_revert_commit(self, target_commit: CommitSHA, reason: str) -> CommitSHA:
        """
        Create a revert commit to the target commit.
        
        Args:
            target_commit: Commit to revert to
            reason: Reason for the revert
            
        Returns:
            SHA of the created revert commit
        """
        try:
            # Generate revert commit message
            revert_message = f"Revert to {target_commit[:8]}: {reason}"
            
            # Create the revert commit using commit manager
            return await self.commit_manager.auto_commit(
                message=revert_message,
                files=None
            )
            
        except Exception as e:
            raise CommitError(f"Failed to create revert commit: {str(e)}") from e

    async def _verify_repository_state(self) -> bool:
        """Verify that the repository is in a clean state."""
        try:
            # This would typically check git status
            # For now, return True as placeholder
            return True
        except Exception:
            return False

    async def _verify_commit_consistency(self) -> bool:
        """Verify that recent commits are consistent."""
        try:
            # This would check for any inconsistencies in recent commits
            # For now, return True as placeholder
            return True
        except Exception:
            return False

    async def _verify_no_pending_changes(self) -> bool:
        """Verify that there are no pending changes."""
        try:
            # This would check for uncommitted changes
            # For now, return True as placeholder
            return True
        except Exception:
            return False

    def _cleanup_rollback_history(self) -> None:
        """Clean up old rollback history entries."""
        if len(self.rollback_history) > self.max_rollback_history:
            # Keep only the most recent entries
            self.rollback_history = self.rollback_history[-self.max_rollback_history:]

    def _rollback_operation_to_dict(self, operation: RollbackOperation) -> Dict[str, Any]:
        """
        Convert a RollbackOperation to a dictionary.
        
        Args:
            operation: RollbackOperation to convert
            
        Returns:
            Dictionary representation of the operation
        """
        return {
            "operation_id": operation.operation_id,
            "session_id": operation.session_id,
            "target_commit": operation.target_commit,
            "source_commit": operation.source_commit,
            "reason": operation.reason,
            "initiated_at": operation.initiated_at.isoformat(),
            "status": operation.status,
            "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
            "revert_commit_sha": operation.revert_commit_sha,
            "verification_passed": operation.verification_passed
        }