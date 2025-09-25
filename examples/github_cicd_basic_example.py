"""
Basic example demonstrating GitHub CI/CD automation system.

This example shows how to create and work with the core data models
and the GitHubMCPClient for the GitHub CI/CD automation system.
"""

import asyncio
from datetime import datetime, timedelta
from escai_framework.github_cicd import (
    WorkflowRun, WorkflowJob, WorkflowStep, AutomationSession,
    WorkflowStatus, WorkflowConclusion, JobStatus, StepStatus, AutomationSessionStatus,
    GitHubMCPClient, utils, constants
)


def create_sample_workflow_run():
    """Create a sample workflow run with jobs and steps."""
    
    # Create workflow steps
    setup_step = WorkflowStep(
        name="Setup Node.js",
        status=StepStatus.COMPLETED,
        number=1,
        conclusion=WorkflowConclusion.SUCCESS,
        started_at=datetime.now() - timedelta(minutes=10),
        completed_at=datetime.now() - timedelta(minutes=9)
    )
    
    build_step = WorkflowStep(
        name="Build application",
        status=StepStatus.COMPLETED,
        number=2,
        conclusion=WorkflowConclusion.SUCCESS,
        started_at=datetime.now() - timedelta(minutes=9),
        completed_at=datetime.now() - timedelta(minutes=7)
    )
    
    test_step = WorkflowStep(
        name="Run tests",
        status=StepStatus.COMPLETED,
        number=3,
        conclusion=WorkflowConclusion.SUCCESS,
        started_at=datetime.now() - timedelta(minutes=7),
        completed_at=datetime.now() - timedelta(minutes=5)
    )
    
    # Create workflow jobs
    build_job = WorkflowJob(
        id=12345,
        name="build",
        status=JobStatus.COMPLETED,
        conclusion=WorkflowConclusion.SUCCESS,
        started_at=datetime.now() - timedelta(minutes=10),
        completed_at=datetime.now() - timedelta(minutes=5),
        steps=[setup_step, build_step, test_step]
    )
    
    deploy_step = WorkflowStep(
        name="Deploy to staging",
        status=StepStatus.COMPLETED,
        number=1,
        conclusion=WorkflowConclusion.SUCCESS,
        started_at=datetime.now() - timedelta(minutes=5),
        completed_at=datetime.now() - timedelta(minutes=2)
    )
    
    deploy_job = WorkflowJob(
        id=12346,
        name="deploy",
        status=JobStatus.COMPLETED,
        conclusion=WorkflowConclusion.SUCCESS,
        started_at=datetime.now() - timedelta(minutes=5),
        completed_at=datetime.now() - timedelta(minutes=2),
        steps=[deploy_step]
    )
    
    # Create workflow run
    workflow_run = WorkflowRun(
        id=67890,
        workflow_id=111,
        workflow_name="CI/CD Pipeline",
        status=WorkflowStatus.COMPLETED,
        conclusion=WorkflowConclusion.SUCCESS,
        repository="myorg/myapp",
        branch="main",
        commit_sha="abc123def456",
        created_at=datetime.now() - timedelta(minutes=10),
        updated_at=datetime.now() - timedelta(minutes=2),
        jobs=[build_job, deploy_job],
        html_url="https://github.com/myorg/myapp/actions/runs/67890",
        run_number=42,
        event="push",
        actor="developer"
    )
    
    return workflow_run


def create_sample_automation_session():
    """Create a sample automation session."""
    
    session_id = utils.generate_session_id()
    
    session = AutomationSession(
        session_id=session_id,
        workflow_run_id=67890,
        repository="myorg/myapp",
        started_at=datetime.now() - timedelta(minutes=10),
        status=AutomationSessionStatus.COMPLETED,
        completed_at=datetime.now() - timedelta(minutes=2),
        rollback_point="def456abc123"
    )
    
    # Add some commits made during the session
    session.add_commit("abc123def456")
    session.add_commit("def456abc123")
    
    # Add some configuration
    session.config = {
        "auto_commit": True,
        "rollback_enabled": True,
        "polling_interval": 30
    }
    
    return session


def demonstrate_workflow_analysis():
    """Demonstrate workflow analysis capabilities."""
    
    print("=== GitHub CI/CD Automation Example ===\n")
    
    # Create sample data
    workflow_run = create_sample_workflow_run()
    session = create_sample_automation_session()
    
    # Display workflow information
    print(f"Workflow: {workflow_run.workflow_name}")
    print(f"Repository: {workflow_run.repository}")
    print(f"Status: {workflow_run.status.value} ({constants.STATUS_SYMBOLS.get(workflow_run.status.value, '?')})")
    print(f"Conclusion: {workflow_run.conclusion.value if workflow_run.conclusion else 'N/A'}")
    print(f"Duration: {utils.format_duration(workflow_run.duration_seconds())}")
    print(f"Progress: {workflow_run.get_progress_percentage():.1f}%")
    print(f"Run URL: {workflow_run.html_url}")
    print()
    
    # Display job information
    print("Jobs:")
    for job in workflow_run.jobs:
        job_symbol = constants.STATUS_SYMBOLS.get(job.conclusion.value if job.conclusion else job.status.value, '?')
        print(f"  {job_symbol} {job.name} - {utils.format_duration(job.duration_seconds())}")
        
        # Display steps for each job
        for step in job.steps:
            step_symbol = constants.STATUS_SYMBOLS.get(step.conclusion.value if step.conclusion else step.status.value, '?')
            print(f"    {step_symbol} {step.name} - {utils.format_duration(step.duration_seconds())}")
    print()
    
    # Display session information
    print(f"Automation Session: {session.session_id}")
    print(f"Status: {session.status.value}")
    print(f"Duration: {utils.format_duration(session.duration_seconds())}")
    print(f"Commits made: {len(session.commits_made)}")
    if session.commits_made:
        print(f"Latest commit: {session.get_latest_commit()}")
    print(f"Rollback point: {session.rollback_point}")
    print()
    
    # Demonstrate utility functions
    print("=== Utility Functions Demo ===")
    
    # Repository parsing
    owner, repo = utils.parse_repository_name(workflow_run.repository)
    print(f"Parsed repository - Owner: {owner}, Repo: {repo}")
    
    # Commit message generation
    commit_msg = utils.create_commit_message(
        workflow_run.workflow_name,
        workflow_run.id,
        {"branch": workflow_run.branch, "actor": workflow_run.actor}
    )
    print(f"Generated commit message:\n{commit_msg}")
    print()
    
    # Success rate calculation
    success_rate = utils.calculate_success_rate(8, 10)
    print(f"Example success rate: {success_rate}% (8 out of 10 successful runs)")
    
    # Input sanitization
    raw_inputs = {
        "deploy env": "production",
        "version-tag": "v1.2.3",
        "enable_tests": True
    }
    sanitized = utils.sanitize_workflow_input(raw_inputs)
    print(f"Sanitized inputs: {sanitized}")
    
    print("\n=== Analysis Results ===")
    
    # Check if workflow was successful
    if workflow_run.is_successful():
        print("✅ Workflow completed successfully!")
    elif workflow_run.is_failed():
        print("❌ Workflow failed!")
        failed_jobs = workflow_run.get_failed_jobs()
        if failed_jobs:
            print(f"Failed jobs: {[job.name for job in failed_jobs]}")
    
    # Check session status
    if session.is_completed():
        print("✅ Automation session completed successfully!")
    elif session.is_failed():
        print("❌ Automation session failed!")
    elif session.is_rolled_back():
        print("🔄 Automation session was rolled back!")
    
    print(f"\nTotal workflow duration: {utils.format_duration(workflow_run.duration_seconds())}")
    print(f"Session managed {len(session.commits_made)} commits")


async def demonstrate_github_mcp_client():
    """Demonstrate GitHubMCPClient usage."""
    
    print("\n=== GitHub MCP Client Demo ===")
    
    # Create client instance
    client = GitHubMCPClient(default_timeout=30)
    
    try:
        # Get available workflows
        print("Getting available workflows...")
        workflows = await client.get_workflows("myorg", "myapp")
        print(f"Found {len(workflows)} workflows:")
        for workflow in workflows:
            print(f"  - {workflow['name']} ({workflow['path']})")
        
        # Trigger a workflow
        print("\nTriggering workflow...")
        inputs = {"environment": "staging", "version": "1.2.3"}
        trigger_result = await client.trigger_workflow_dispatch(
            "myorg", "myapp", "ci.yml", inputs
        )
        print(f"Workflow triggered: {trigger_result['workflow_id']}")
        
        # Get workflow run details
        print("\nGetting workflow run details...")
        workflow_run = await client.get_workflow_run("myorg", "myapp", 12345)
        print(f"Workflow: {workflow_run.workflow_name}")
        print(f"Status: {workflow_run.status.value}")
        print(f"Repository: {workflow_run.repository}")
        print(f"Branch: {workflow_run.branch}")
        print(f"Commit: {workflow_run.commit_sha}")
        
        # Get workflow jobs
        print("\nGetting workflow jobs...")
        jobs = await client.get_workflow_run_jobs("myorg", "myapp", 12345)
        print(f"Found {len(jobs)} jobs:")
        for job in jobs:
            print(f"  - {job.name}: {job.status.value}")
            for step in job.steps:
                print(f"    • {step.name}: {step.status.value}")
        
        # List recent workflow runs
        print("\nListing recent workflow runs...")
        runs = await client.list_workflow_runs("myorg", "myapp", per_page=5)
        print(f"Found {len(runs)} recent runs:")
        for run in runs:
            status_symbol = constants.STATUS_SYMBOLS.get(run.status.value, '?')
            print(f"  {status_symbol} #{run.run_number} - {run.workflow_name} ({run.status.value})")
        
        # Check rate limit status
        rate_limit = client.get_rate_limit_status()
        print(f"\nRate limit status: {rate_limit}")
        
    except Exception as e:
        print(f"Error during MCP client demo: {e}")


def main():
    """Run all demonstrations."""
    # Run synchronous demo
    demonstrate_workflow_analysis()
    
    # Run asynchronous demo
    asyncio.run(demonstrate_github_mcp_client())


if __name__ == "__main__":
    main()