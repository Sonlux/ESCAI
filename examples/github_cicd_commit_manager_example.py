#!/usr/bin/env python3
"""
Example demonstrating the GitHub CI/CD CommitManager functionality.

This example shows how to use the CommitManager for automated commit and push
operations with workflow context integration.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from escai_framework.github_cicd import (
    CommitManager,
    CommitContext,
    WorkflowRun,
    AutomationSession,
    WorkflowStatus,
    WorkflowConclusion,
    AutomationSessionStatus,
    GitHubMCPClient
)


class MockGitHubMCPClient:
    """Mock GitHub MCP client for demonstration purposes."""
    
    def __init__(self):
        self.calls = []
    
    def log_call(self, method, *args, **kwargs):
        """Log method calls for demonstration."""
        self.calls.append(f"{method}({args}, {kwargs})")


def create_sample_workflow_run() -> WorkflowRun:
    """Create a sample workflow run for demonstration."""
    return WorkflowRun(
        id=12345,
        workflow_id=67890,
        workflow_name="CI/CD Pipeline",
        status=WorkflowStatus.IN_PROGRESS,
        repository="owner/repo",
        branch="main",
        commit_sha="abc123def456",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        run_number=42,
        event="push",
        actor="developer"
    )


def create_sample_automation_session() -> AutomationSession:
    """Create a sample automation session for demonstration."""
    return AutomationSession(
        session_id="session-12345",
        workflow_run_id=12345,
        repository="owner/repo",
        started_at=datetime.now(),
        status=AutomationSessionStatus.ACTIVE
    )


def demonstrate_basic_commit_operations():
    """Demonstrate basic commit operations."""
    print("=== Basic Commit Operations Demo ===\n")
    
    # Initialize CommitManager
    github_client = MockGitHubMCPClient()
    commit_manager = CommitManager(
        github_client=github_client,
        repository_path=".",  # Current directory
        max_retries=3,
        retry_delay=1.0
    )
    
    print("1. CommitManager initialized")
    print(f"   Repository path: {commit_manager.repository_path}")
    print(f"   Max retries: {commit_manager.max_retries}")
    print(f"   Auto-stage enabled: {commit_manager.auto_stage}")
    
    # Check for uncommitted changes
    try:
        has_changes = commit_manager.has_uncommitted_changes()
        print(f"\n2. Repository status check:")
        print(f"   Has uncommitted changes: {has_changes}")
        
        if has_changes:
            modified_files = commit_manager.get_modified_files()
            print(f"   Modified files: {modified_files[:5]}...")  # Show first 5 files
    except Exception as e:
        print(f"   Error checking repository status: {e}")
    
    # Get current commit SHA
    try:
        current_sha = commit_manager.get_current_commit_sha()
        print(f"   Current commit SHA: {current_sha[:8]}...")
    except Exception as e:
        print(f"   Error getting current commit SHA: {e}")
    
    # Get repository commit history
    try:
        history = commit_manager.get_repository_commit_history(limit=3)
        print(f"\n3. Recent commit history ({len(history)} commits):")
        for i, commit in enumerate(history):
            print(f"   {i+1}. {commit['sha'][:8]} - {commit['message'][:50]}...")
            print(f"      Author: {commit['author']} | Date: {commit['date']}")
    except Exception as e:
        print(f"   Error getting commit history: {e}")


def demonstrate_contextual_commit_messages():
    """Demonstrate contextual commit message generation."""
    print("\n=== Contextual Commit Messages Demo ===\n")
    
    github_client = MockGitHubMCPClient()
    commit_manager = CommitManager(github_client=github_client, repository_path=".")
    
    # Create sample workflow context
    workflow_run = create_sample_workflow_run()
    session = create_sample_automation_session()
    
    # Test different commit contexts
    contexts = [
        CommitContext(
            workflow_run=workflow_run,
            session=session,
            files_changed=['src/main.py', 'tests/test_main.py'],
            operation_type="automation",
            custom_message="Fixed failing tests"
        ),
        CommitContext(
            workflow_run=workflow_run,
            files_changed=['deploy.yaml', 'config.json', 'scripts/deploy.sh'],
            operation_type="deployment",
            custom_message="Updated deployment configuration"
        ),
        CommitContext(
            files_changed=[f"file_{i}.py" for i in range(10)],
            operation_type="refactoring",
            custom_message="Large refactoring changes"
        )
    ]
    
    print("Generated commit messages with different contexts:\n")
    
    for i, context in enumerate(contexts, 1):
        message = commit_manager._generate_commit_message(
            f"Automated commit #{i}",
            context
        )
        print(f"{i}. Context: {context.operation_type}")
        print(f"   Files: {len(context.files_changed)} files")
        if context.workflow_run:
            print(f"   Workflow: {context.workflow_run.workflow_name}")
        print(f"   Message preview:")
        # Show first few lines of the message
        lines = message.split('\n')
        for line in lines[:4]:
            print(f"     {line}")
        if len(lines) > 4:
            print(f"     ... ({len(lines) - 4} more lines)")
        print()


def demonstrate_commit_history_tracking():
    """Demonstrate commit history tracking."""
    print("=== Commit History Tracking Demo ===\n")
    
    github_client = MockGitHubMCPClient()
    commit_manager = CommitManager(github_client=github_client, repository_path=".")
    
    # Simulate some commit operations by adding to history
    from escai_framework.github_cicd.commit_manager import CommitResult
    
    sample_commits = [
        CommitResult(
            success=True,
            commit_sha="abc123def",
            message="Initial commit",
            files_committed=['README.md'],
            timestamp=datetime.now()
        ),
        CommitResult(
            success=True,
            commit_sha="def456abc",
            message="Add feature X",
            files_committed=['src/feature_x.py', 'tests/test_feature_x.py'],
            timestamp=datetime.now()
        ),
        CommitResult(
            success=False,
            error="Commit failed: nothing to commit",
            message="Empty commit attempt",
            timestamp=datetime.now()
        )
    ]
    
    # Add to commit manager history
    commit_manager._commit_history.extend(sample_commits)
    
    print("Commit history from this session:")
    history = commit_manager.get_commit_history()
    
    for i, commit in enumerate(history, 1):
        print(f"{i}. {'✓' if commit.success else '✗'} {commit.message}")
        if commit.success:
            print(f"   SHA: {commit.commit_sha}")
            print(f"   Files: {commit.files_committed}")
        else:
            print(f"   Error: {commit.error}")
        print(f"   Time: {commit.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print()


def demonstrate_workflow_integration():
    """Demonstrate workflow context integration."""
    print("=== Workflow Integration Demo ===\n")
    
    github_client = MockGitHubMCPClient()
    commit_manager = CommitManager(github_client=github_client, repository_path=".")
    
    # Create workflow context
    workflow_run = create_sample_workflow_run()
    session = create_sample_automation_session()
    
    print("Workflow context:")
    print(f"  Workflow: {workflow_run.workflow_name} (#{workflow_run.run_number})")
    print(f"  Repository: {workflow_run.repository}")
    print(f"  Branch: {workflow_run.branch}")
    print(f"  Status: {workflow_run.status.value}")
    print(f"  Session: {session.session_id}")
    print()
    
    # Demonstrate workflow commit message generation
    changes = ['src/api.py', 'src/models.py', 'tests/test_api.py']
    
    print("Creating workflow-specific commit:")
    print(f"  Changed files: {changes}")
    
    # This would normally perform the actual commit, but we'll just show the message
    context = CommitContext(
        workflow_run=workflow_run,
        session=session,
        files_changed=changes,
        operation_type="workflow_automation",
        custom_message="API improvements and tests"
    )
    
    message = commit_manager._generate_commit_message(
        f"Automated changes from workflow: {workflow_run.workflow_name}",
        context
    )
    
    print("  Generated commit message:")
    for line in message.split('\n'):
        print(f"    {line}")


def main():
    """Run all demonstrations."""
    print("GitHub CI/CD CommitManager Example")
    print("=" * 50)
    
    try:
        demonstrate_basic_commit_operations()
        demonstrate_contextual_commit_messages()
        demonstrate_commit_history_tracking()
        demonstrate_workflow_integration()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nNote: This example demonstrates the CommitManager API")
        print("without performing actual git operations. In a real scenario,")
        print("the CommitManager would interact with your git repository.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("This might be expected if not running in a git repository.")


if __name__ == "__main__":
    main()