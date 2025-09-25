"""
Utility functions for GitHub CI/CD automation system.

This module provides common utility functions used across the automation system.
"""

import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from .models import WorkflowStatus, WorkflowConclusion, JobStatus, StepStatus


def generate_session_id() -> str:
    """Generate a unique session ID for automation sessions."""
    return str(uuid.uuid4())


def parse_repository_name(repo_full_name: str) -> tuple[str, str]:
    """
    Parse a full repository name into owner and repo name.
    
    Args:
        repo_full_name: Repository name in format "owner/repo"
        
    Returns:
        Tuple of (owner, repo_name)
        
    Raises:
        ValueError: If repository name format is invalid
    """
    if "/" not in repo_full_name:
        raise ValueError(f"Invalid repository name format: {repo_full_name}")
    
    parts = repo_full_name.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid repository name format: {repo_full_name}")
    
    return parts[0], parts[1]


def format_duration(seconds: Optional[float]) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2m 30s", "1h 15m")
    """
    if seconds is None:
        return "N/A"
    
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def create_commit_message(
    workflow_name: str,
    workflow_run_id: int,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a standardized commit message for workflow automation.
    
    Args:
        workflow_name: Name of the workflow
        workflow_run_id: ID of the workflow run
        context: Additional context information
        
    Returns:
        Formatted commit message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_message = f"[CI/CD] Automated commit from {workflow_name} (run #{workflow_run_id})"
    
    if context:
        context_info = []
        for key, value in context.items():
            context_info.append(f"{key}: {value}")
        
        if context_info:
            base_message += f"\n\nContext:\n" + "\n".join(f"- {info}" for info in context_info)
    
    base_message += f"\n\nTimestamp: {timestamp}"
    return base_message


def is_workflow_terminal_state(status: WorkflowStatus) -> bool:
    """
    Check if a workflow status represents a terminal state.
    
    Args:
        status: Workflow status to check
        
    Returns:
        True if the status is terminal (completed or failed)
    """
    return status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]


def is_successful_conclusion(conclusion: Optional[WorkflowConclusion]) -> bool:
    """
    Check if a workflow conclusion represents success.
    
    Args:
        conclusion: Workflow conclusion to check
        
    Returns:
        True if the conclusion represents success
    """
    return conclusion == WorkflowConclusion.SUCCESS


def calculate_success_rate(
    successful_runs: int, 
    total_runs: int
) -> float:
    """
    Calculate success rate percentage.
    
    Args:
        successful_runs: Number of successful runs
        total_runs: Total number of runs
        
    Returns:
        Success rate as percentage (0.0 to 100.0)
    """
    if total_runs == 0:
        return 0.0
    return (successful_runs / total_runs) * 100.0


def sanitize_workflow_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize workflow inputs to ensure they are safe for GitHub API.
    
    Args:
        inputs: Raw workflow inputs
        
    Returns:
        Sanitized workflow inputs
    """
    sanitized = {}
    
    for key, value in inputs.items():
        # Convert key to string and remove any potentially dangerous characters
        clean_key = str(key).replace(" ", "_").replace("-", "_")
        clean_key = "".join(c for c in clean_key if c.isalnum() or c == "_")
        
        # Convert value to string and limit length
        if isinstance(value, (str, int, float, bool)):
            clean_value = str(value)[:1000]  # Limit to 1000 characters
        else:
            clean_value = str(value)[:1000]
        
        sanitized[clean_key] = clean_value
    
    return sanitized


def validate_repository_access(owner: str, repo: str) -> Dict[str, str]:
    """
    Validate repository access parameters.
    
    Args:
        owner: Repository owner
        repo: Repository name
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ValueError: If parameters are invalid
    """
    errors = {}
    
    if not owner or not isinstance(owner, str):
        errors["owner"] = "Owner must be a non-empty string"
    elif not owner.replace("-", "").replace("_", "").isalnum():
        errors["owner"] = "Owner contains invalid characters"
    
    if not repo or not isinstance(repo, str):
        errors["repo"] = "Repository name must be a non-empty string"
    elif not repo.replace("-", "").replace("_", "").replace(".", "").isalnum():
        errors["repo"] = "Repository name contains invalid characters"
    
    if errors:
        raise ValueError(f"Invalid repository parameters: {errors}")
    
    return {"owner": owner, "repo": repo}


def extract_error_context(error: Exception) -> Dict[str, Any]:
    """
    Extract contextual information from an exception.
    
    Args:
        error: Exception to extract context from
        
    Returns:
        Dictionary with error context information
    """
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
        "module": getattr(error, "__module__", "unknown")
    }


def parse_github_datetime(datetime_str: Optional[str]) -> Optional[datetime]:
    """
    Parse GitHub API datetime string to datetime object.
    
    Args:
        datetime_str: ISO 8601 datetime string from GitHub API
        
    Returns:
        Parsed datetime object or None if input is None/invalid
    """
    if not datetime_str:
        return None
    
    try:
        # GitHub API returns ISO 8601 format: 2024-01-01T10:00:00Z
        if datetime_str.endswith('Z'):
            datetime_str = datetime_str[:-1] + '+00:00'
        return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None


def validate_repository_format(owner: str, repo: str) -> None:
    """
    Validate repository owner and name format.
    
    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        
    Raises:
        ValueError: If owner or repo format is invalid
    """
    if not owner or not isinstance(owner, str):
        raise ValueError("Owner must be a non-empty string")
    
    if not repo or not isinstance(repo, str):
        raise ValueError("Repository name must be a non-empty string")
    
    # GitHub username/org name validation
    if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$', owner):
        raise ValueError(f"Invalid owner format: {owner}")
    
    # GitHub repository name validation
    if not re.match(r'^[a-zA-Z0-9._-]+$', repo):
        raise ValueError(f"Invalid repository name format: {repo}")
    
    if len(owner) > 39:
        raise ValueError(f"Owner name too long: {owner}")
    
    if len(repo) > 100:
        raise ValueError(f"Repository name too long: {repo}")


def validate_workflow_inputs(inputs: Dict[str, Any]) -> bool:
    """
    Validate workflow input parameters.
    
    Args:
        inputs: Dictionary of workflow inputs to validate
        
    Returns:
        True if inputs are valid, False otherwise
        
    Raises:
        ValueError: If inputs contain invalid values
    """
    if not isinstance(inputs, dict):
        raise ValueError("Workflow inputs must be a dictionary")
    
    # Check for valid input types
    for key, value in inputs.items():
        if not isinstance(key, str):
            raise ValueError(f"Input key must be string, got {type(key)}")
        
        # GitHub workflow inputs should be strings, numbers, or booleans
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Input value for '{key}' must be string, number, or boolean")
    
    return True


def sanitize_workflow_inputs(inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitize workflow inputs to prevent injection attacks and ensure API compatibility.
    
    Args:
        inputs: Raw workflow inputs dictionary
        
    Returns:
        Sanitized workflow inputs dictionary
    """
    if not inputs:
        return {}
    
    sanitized = {}
    
    for key, value in inputs.items():
        # Sanitize key
        if not isinstance(key, str):
            continue
        
        # Remove potentially dangerous characters from key
        clean_key = re.sub(r'[^a-zA-Z0-9_-]', '_', key)
        clean_key = clean_key.strip('_-')[:50]  # Limit key length
        
        if not clean_key:
            continue
        
        # Sanitize value
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, str):
                # Remove control characters and limit length
                clean_value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str(value))
                clean_value = clean_value[:1000]  # Limit value length
            else:
                clean_value = str(value)
        else:
            # Convert complex types to string representation
            clean_value = str(value)[:1000]
        
        sanitized[clean_key] = clean_value
    
    return sanitized


def format_timestamp(dt: datetime) -> str:
    """
    Format a datetime object to a standardized timestamp string.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        Formatted timestamp string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def sanitize_commit_message(message: str) -> str:
    """
    Sanitize a commit message to ensure it's safe for git operations.
    
    Args:
        message: Raw commit message
        
    Returns:
        Sanitized commit message
    """
    if not message:
        return "Automated commit"
    
    # Remove control characters and normalize whitespace
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', message)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Limit message length (git has practical limits)
    if len(sanitized) > 2000:
        sanitized = sanitized[:1997] + "..."
    
    # Ensure message is not empty after sanitization
    if not sanitized:
        return "Automated commit"
    
    return sanitized