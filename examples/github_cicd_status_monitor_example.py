#!/usr/bin/env python3
"""
GitHub CI/CD Status Monitor Example

This example demonstrates how to use the StatusMonitor class to track
workflow progress in real-time with configurable polling intervals,
progress reporting, and status caching.
"""

import asyncio
import logging
from datetime import datetime

from escai_framework.github_cicd import (
    StatusMonitor, ProgressReport, GitHubMCPClient
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_callback(report: ProgressReport) -> None:
    """
    Callback function for progress updates.
    
    Args:
        report: Progress report with current status
    """
    print(f"\n📊 Progress Update for Workflow {report.workflow_run_id}")
    print(f"   Repository: {report.repository}")
    print(f"   Workflow: {report.workflow_name}")
    print(f"   Status: {report.status.value}")
    print(f"   Progress: {report.progress_percentage:.1f}%")
    print(f"   Jobs: {report.jobs_completed}/{report.jobs_total}")
    print(f"   Steps: {report.steps_completed}/{report.steps_total}")
    
    if report.duration_seconds:
        print(f"   Duration: {report.duration_seconds:.0f}s")
    
    if report.estimated_remaining_seconds:
        print(f"   Estimated remaining: {report.estimated_remaining_seconds:.0f}s")
    
    if report.failed_jobs:
        print(f"   ❌ Failed jobs: {', '.join(report.failed_jobs)}")
    
    if report.failed_steps:
        print(f"   ❌ Failed steps: {', '.join(report.failed_steps)}")


def completion_callback(report: ProgressReport) -> None:
    """
    Callback function for workflow completion.
    
    Args:
        report: Final progress report
    """
    print(f"\n🎉 Workflow {report.workflow_run_id} Completed!")
    print(f"   Status: {report.conclusion.value if report.conclusion else 'Unknown'}")
    print(f"   Duration: {report.duration_seconds:.0f}s" if report.duration_seconds else "   Duration: Unknown")
    
    if report.is_successful():
        print("   ✅ All jobs completed successfully!")
    elif report.has_failures():
        print("   ❌ Some jobs or steps failed:")
        for job in report.failed_jobs:
            print(f"      - Job: {job}")
        for step in report.failed_steps:
            print(f"      - Step: {step}")


async def monitor_single_workflow():
    """Example: Monitor a single workflow run."""
    print("🚀 Example 1: Monitor Single Workflow")
    print("=" * 50)
    
    # Initialize GitHub client and status monitor
    github_client = GitHubMCPClient()
    monitor = StatusMonitor(
        github_client=github_client,
        default_poll_interval=10,  # Poll every 10 seconds
        default_timeout=1800,      # 30 minute timeout
        cache_ttl_seconds=30       # Cache for 30 seconds
    )
    
    try:
        # Start monitoring a workflow run
        session_id = await monitor.start_monitoring(
            workflow_run_id=12345,
            repository="owner/repo",
            poll_interval=5,  # Override default to poll every 5 seconds
            progress_callback=progress_callback,
            completion_callback=completion_callback
        )
        
        print(f"Started monitoring session: {session_id}")
        
        # Get current status manually
        report = await monitor.get_current_status(12345, "owner/repo")
        print(f"\nInitial status: {report.status.value}")
        print(f"Progress: {report.progress_percentage:.1f}%")
        
        # Wait for completion (with timeout)
        try:
            final_report = await monitor.wait_for_completion(
                workflow_run_id=12345,
                repository="owner/repo",
                timeout=300,  # 5 minute timeout
                poll_interval=10
            )
            
            print(f"\nWorkflow completed with status: {final_report.conclusion}")
            
        except asyncio.TimeoutError:
            print("\nTimeout waiting for workflow completion")
        
        # Stop monitoring
        await monitor.stop_monitoring(session_id)
        
    except Exception as e:
        logger.error(f"Error monitoring workflow: {e}")
    
    finally:
        await monitor.shutdown()


async def monitor_multiple_workflows():
    """Example: Monitor multiple workflow runs concurrently."""
    print("\n🚀 Example 2: Monitor Multiple Workflows")
    print("=" * 50)
    
    # Initialize status monitor
    github_client = GitHubMCPClient()
    monitor = StatusMonitor(
        github_client=github_client,
        max_concurrent_sessions=5,
        cache_ttl_seconds=60
    )
    
    workflow_runs = [12345, 12346, 12347]
    session_ids = []
    
    try:
        # Start monitoring multiple workflows
        for run_id in workflow_runs:
            session_id = await monitor.start_monitoring(
                workflow_run_id=run_id,
                repository="owner/repo",
                progress_callback=lambda report, rid=run_id: print(f"Workflow {rid}: {report.progress_percentage:.1f}%"),
                completion_callback=lambda report, rid=run_id: print(f"Workflow {rid} completed: {report.conclusion}")
            )
            session_ids.append(session_id)
            print(f"Started monitoring workflow {run_id} (session: {session_id})")
        
        # Check active sessions
        active_sessions = monitor.get_active_sessions()
        print(f"\nActive monitoring sessions: {len(active_sessions)}")
        
        # Get session information
        for session_id in session_ids:
            info = monitor.get_session_info(session_id)
            if info:
                print(f"Session {session_id}: monitoring {len(info['workflow_run_ids'])} workflows")
        
        # Wait a bit for monitoring to process
        await asyncio.sleep(30)
        
        # Stop all monitoring sessions
        for session_id in session_ids:
            await monitor.stop_monitoring(session_id)
        
    except Exception as e:
        logger.error(f"Error monitoring multiple workflows: {e}")
    
    finally:
        await monitor.shutdown()


async def demonstrate_caching():
    """Example: Demonstrate status caching functionality."""
    print("\n🚀 Example 3: Status Caching")
    print("=" * 50)
    
    github_client = GitHubMCPClient()
    monitor = StatusMonitor(
        github_client=github_client,
        cache_ttl_seconds=10  # Very short cache for demonstration
    )
    
    try:
        workflow_run_id = 12345
        repository = "owner/repo"
        
        # First call - will hit the API
        print("First call (will hit API)...")
        start_time = datetime.now()
        report1 = await monitor.get_current_status(workflow_run_id, repository)
        duration1 = (datetime.now() - start_time).total_seconds()
        print(f"Duration: {duration1:.3f}s")
        
        # Second call immediately - should use cache
        print("\nSecond call (should use cache)...")
        start_time = datetime.now()
        report2 = await monitor.get_current_status(workflow_run_id, repository)
        duration2 = (datetime.now() - start_time).total_seconds()
        print(f"Duration: {duration2:.3f}s")
        
        # Verify same data
        print(f"Same data: {report1.last_updated == report2.last_updated}")
        print(f"Cache speedup: {duration1/duration2:.1f}x faster")
        
        # Wait for cache to expire
        print("\nWaiting for cache to expire...")
        await asyncio.sleep(11)
        
        # Third call - should hit API again
        print("Third call (cache expired, will hit API)...")
        start_time = datetime.now()
        report3 = await monitor.get_current_status(workflow_run_id, repository)
        duration3 = (datetime.now() - start_time).total_seconds()
        print(f"Duration: {duration3:.3f}s")
        
    except Exception as e:
        logger.error(f"Error demonstrating caching: {e}")
    
    finally:
        await monitor.shutdown()


async def main():
    """Main example function."""
    print("🔍 GitHub CI/CD Status Monitor Examples")
    print("=" * 60)
    
    # Note: These examples use mock data since we don't have a real GitHub connection
    print("Note: These examples demonstrate the API but use mock data")
    print("In a real implementation, connect to actual GitHub workflows\n")
    
    try:
        # Run examples
        await monitor_single_workflow()
        await monitor_multiple_workflows()
        await demonstrate_caching()
        
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    print("\n✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())