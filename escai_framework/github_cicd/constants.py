"""
Constants for GitHub CI/CD automation system.

This module defines constants used throughout the automation system.
"""

# Default configuration values
DEFAULT_POLLING_INTERVAL = 30  # seconds
DEFAULT_TIMEOUT = 3600  # seconds (1 hour)
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_RATE_LIMIT_BUFFER = 100  # requests

# GitHub API limits
GITHUB_API_RATE_LIMIT = 5000  # requests per hour for authenticated users
GITHUB_API_SECONDARY_RATE_LIMIT = 100  # requests per minute

# Workflow monitoring limits
MAX_CONCURRENT_WORKFLOWS = 10
MAX_MONITORING_DURATION = 86400  # seconds (24 hours)
MAX_COMMIT_MESSAGE_LENGTH = 2000
MAX_ERROR_LOG_ENTRIES = 100

# Retry configuration
MIN_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 300  # seconds (5 minutes)
EXPONENTIAL_BACKOFF_MULTIPLIER = 2

# File and directory limits
MAX_FILES_PER_COMMIT = 100
MAX_FILE_SIZE_MB = 100

# Session configuration
SESSION_CLEANUP_INTERVAL = 3600  # seconds (1 hour)
MAX_ACTIVE_SESSIONS = 50

# Logging configuration
LOG_ROTATION_SIZE_MB = 10
LOG_RETENTION_DAYS = 30

# GitHub webhook events that can trigger workflows
SUPPORTED_WORKFLOW_EVENTS = [
    "push",
    "pull_request",
    "workflow_dispatch",
    "schedule",
    "release",
    "issues",
    "issue_comment",
    "pull_request_review",
    "pull_request_review_comment",
    "repository_dispatch"
]

# Workflow run conclusions that indicate failure
FAILURE_CONCLUSIONS = [
    "failure",
    "cancelled",
    "timed_out"
]

# Workflow run conclusions that indicate success
SUCCESS_CONCLUSIONS = [
    "success"
]

# Environment variable names
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"
ENV_GITHUB_API_BASE_URL = "GITHUB_API_BASE_URL"
ENV_WORKFLOW_POLL_INTERVAL = "WORKFLOW_POLL_INTERVAL"
ENV_WORKFLOW_TIMEOUT = "WORKFLOW_TIMEOUT"
ENV_AUTO_COMMIT_ENABLED = "AUTO_COMMIT_ENABLED"
ENV_ROLLBACK_ENABLED = "ROLLBACK_ENABLED"

# Default GitHub API base URL
DEFAULT_GITHUB_API_BASE_URL = "https://api.github.com"

# Status display symbols
STATUS_SYMBOLS = {
    "queued": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "failed": "❌",
    "success": "✅",
    "failure": "❌",
    "cancelled": "🚫",
    "skipped": "⏭️",
    "timed_out": "⏰",
    "action_required": "⚠️",
    "neutral": "⚪"
}

# Progress bar configuration
PROGRESS_BAR_WIDTH = 50
PROGRESS_BAR_FILL_CHAR = "█"
PROGRESS_BAR_EMPTY_CHAR = "░"

# Error message templates
ERROR_MESSAGES = {
    "workflow_not_found": "Workflow '{workflow_name}' not found in repository {repository}",
    "workflow_trigger_failed": "Failed to trigger workflow '{workflow_name}': {error}",
    "monitoring_failed": "Failed to monitor workflow run {run_id}: {error}",
    "commit_failed": "Failed to create commit: {error}",
    "push_failed": "Failed to push changes: {error}",
    "rollback_failed": "Failed to rollback to commit {commit_sha}: {error}",
    "authentication_failed": "GitHub authentication failed. Please check your token.",
    "rate_limit_exceeded": "GitHub API rate limit exceeded. Retrying in {retry_after} seconds.",
    "network_error": "Network error occurred: {error}. Retrying...",
    "invalid_repository": "Invalid repository format. Expected 'owner/repo', got '{repository}'",
    "insufficient_permissions": "Insufficient permissions to access repository {repository}",
    "workflow_timeout": "Workflow run {run_id} timed out after {timeout} seconds"
}

# Success message templates
SUCCESS_MESSAGES = {
    "workflow_triggered": "Successfully triggered workflow '{workflow_name}' (run #{run_id})",
    "workflow_completed": "Workflow '{workflow_name}' completed successfully in {duration}",
    "commit_created": "Successfully created commit {commit_sha}: {message}",
    "changes_pushed": "Successfully pushed changes to {branch}",
    "rollback_completed": "Successfully rolled back to commit {commit_sha}",
    "monitoring_started": "Started monitoring workflow run {run_id}",
    "session_created": "Created automation session {session_id}"
}

# Configuration file names
CONFIG_FILE_NAME = "github_cicd_config.yaml"
SECRETS_FILE_NAME = ".github_cicd_secrets"

# Default branch names
DEFAULT_BRANCH_NAMES = ["main", "master", "develop"]

# Commit message prefixes
COMMIT_MESSAGE_PREFIXES = {
    "automation": "[CI/CD]",
    "rollback": "[ROLLBACK]",
    "fix": "[FIX]",
    "feature": "[FEATURE]",
    "docs": "[DOCS]",
    "test": "[TEST]"
}