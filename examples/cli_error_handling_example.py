"""
Example demonstrating the CLI error handling framework.

This example shows how the error handling system works with different
types of errors and recovery mechanisms.
"""

import asyncio
import time
from typing import Optional

from escai_framework.cli.utils.error_handling import (
    CLIError, CLIErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext,
    ValidationError, NetworkError, FrameworkError, ConfigurationError,
    RetryConfig, create_error_with_suggestions
)
from escai_framework.cli.utils.error_decorators import (
    handle_cli_errors, retry_on_network_error, validate_input,
    require_framework, graceful_degradation, log_performance,
    create_command_context
)
from rich.console import Console


def main():
    """Demonstrate error handling capabilities."""
    console = Console()
    console.print("[bold blue]ESCAI CLI Error Handling Framework Demo[/bold blue]\n")
    
    # Initialize error handler
    error_handler = CLIErrorHandler(console=console)
    
    # Demo 1: Basic error handling
    console.print("[bold]Demo 1: Basic Error Handling[/bold]")
    demo_basic_error_handling(error_handler)
    
    # Demo 2: Retry mechanism
    console.print("\n[bold]Demo 2: Retry Mechanism[/bold]")
    demo_retry_mechanism()
    
    # Demo 3: Graceful degradation
    console.print("\n[bold]Demo 3: Graceful Degradation[/bold]")
    demo_graceful_degradation()
    
    # Demo 4: Input validation
    console.print("\n[bold]Demo 4: Input Validation[/bold]")
    demo_input_validation()
    
    # Demo 5: Framework requirements
    console.print("\n[bold]Demo 5: Framework Requirements[/bold]")
    demo_framework_requirements()
    
    # Demo 6: Performance monitoring
    console.print("\n[bold]Demo 6: Performance Monitoring[/bold]")
    demo_performance_monitoring()
    
    # Demo 7: Error statistics
    console.print("\n[bold]Demo 7: Error Statistics[/bold]")
    demo_error_statistics(error_handler)


def demo_basic_error_handling(error_handler: CLIErrorHandler):
    """Demonstrate basic error handling."""
    console = error_handler.console
    
    # Create different types of errors
    errors = [
        ValidationError(
            "Invalid email format: 'not-an-email'",
            field="email"
        ),
        NetworkError(
            "Connection timeout after 30 seconds",
            endpoint="https://api.escai.framework/monitor"
        ),
        FrameworkError(
            "LangChain framework not properly configured",
            framework="langchain",
            severity=ErrorSeverity.HIGH
        ),
        ConfigurationError(
            "Missing required configuration: database_url",
            config_key="database_url"
        )
    ]
    
    console.print("Handling various error types:")
    for error in errors:
        console.print(f"\n[dim]Handling {type(error).__name__}...[/dim]")
        error_handler.handle_error(error)
        time.sleep(1)  # Pause for demonstration


def demo_retry_mechanism():
    """Demonstrate retry mechanism with exponential backoff."""
    console = Console()
    
    # Simulate a flaky network function
    attempt_count = 0
    
    @retry_on_network_error(max_attempts=3, base_delay=0.5)
    def flaky_api_call():
        nonlocal attempt_count
        attempt_count += 1
        console.print(f"[dim]API call attempt {attempt_count}[/dim]")
        
        if attempt_count < 3:
            raise NetworkError("Temporary network issue")
        return {"status": "success", "data": "API response"}
    
    console.print("Calling flaky API with retry mechanism:")
    try:
        result = flaky_api_call()
        console.print(f"[green]✅ Success: {result}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed after all retries: {e}[/red]")


def demo_graceful_degradation():
    """Demonstrate graceful degradation with fallback."""
    console = Console()
    
    def get_live_data():
        """Simulate getting live data that might fail."""
        raise NetworkError("Live data service unavailable")
    
    def get_cached_data():
        """Fallback to cached data."""
        return {"data": "cached_results", "timestamp": "2024-01-01T12:00:00Z"}
    
    @graceful_degradation(
        fallback_func=get_cached_data,
        degraded_message="Live data unavailable, using cached data"
    )
    def get_monitoring_data():
        return get_live_data()
    
    console.print("Attempting to get monitoring data:")
    result = get_monitoring_data()
    console.print(f"[yellow]📊 Data retrieved: {result}[/yellow]")


def demo_input_validation():
    """Demonstrate input validation."""
    console = Console()
    
    @validate_input(
        lambda x: isinstance(x, str) and len(x) > 0,
        "Agent ID must be a non-empty string",
        "agent_id"
    )
    def start_monitoring(agent_id: str):
        return f"Monitoring started for agent: {agent_id}"
    
    console.print("Testing input validation:")
    
    # Valid input
    try:
        result = start_monitoring("valid_agent_123")
        console.print(f"[green]✅ {result}[/green]")
    except Exception as e:
        console.print(f"[red]❌ {e}[/red]")
    
    # Invalid input
    try:
        result = start_monitoring("")
        console.print(f"[green]✅ {result}[/green]")
    except ValidationError as e:
        console.print(f"[red]❌ Validation failed: {e.message}[/red]")


def demo_framework_requirements():
    """Demonstrate framework requirement checking."""
    console = Console()
    
    @require_framework("json")  # Using built-in module that should be available
    def process_with_json():
        import json
        return json.dumps({"status": "success"})
    
    @require_framework("nonexistent_framework")
    def process_with_nonexistent():
        return "This should not execute"
    
    console.print("Testing framework requirements:")
    
    # Available framework
    try:
        result = process_with_json()
        console.print(f"[green]✅ JSON processing: {result}[/green]")
    except Exception as e:
        console.print(f"[red]❌ {e}[/red]")
    
    # Unavailable framework
    try:
        result = process_with_nonexistent()
        console.print(f"[green]✅ {result}[/green]")
    except FrameworkError as e:
        console.print(f"[red]❌ Framework error: {e.message}[/red]")


def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    console = Console()
    
    @log_performance(threshold_seconds=0.1)
    def fast_operation():
        time.sleep(0.05)
        return "Fast operation completed"
    
    @log_performance(threshold_seconds=0.1)
    def slow_operation():
        time.sleep(0.2)
        return "Slow operation completed"
    
    console.print("Testing performance monitoring:")
    
    # Fast operation (no warning)
    result = fast_operation()
    console.print(f"[green]✅ {result}[/green]")
    
    # Slow operation (should trigger warning)
    result = slow_operation()
    console.print(f"[yellow]⚠️  {result}[/yellow]")


def demo_error_statistics(error_handler: CLIErrorHandler):
    """Demonstrate error statistics tracking."""
    console = error_handler.console
    
    # Generate some errors for statistics
    test_errors = [
        ValidationError("Test validation error 1"),
        ValidationError("Test validation error 2"),
        NetworkError("Test network error 1"),
        FrameworkError("Test framework error", framework="test"),
        NetworkError("Test network error 2"),
    ]
    
    console.print("Generating errors for statistics:")
    for error in test_errors:
        error_handler.handle_error(error)
    
    # Display statistics
    stats = error_handler.get_error_statistics()
    console.print(f"\n[bold]Error Statistics:[/bold]")
    console.print(f"Total errors: {stats['total_errors']}")
    console.print(f"Error counts by category:")
    for category, count in stats['error_counts'].items():
        console.print(f"  - {category.value}: {count}")
    
    if stats['degraded_features']:
        console.print(f"Degraded features: {', '.join(stats['degraded_features'])}")
    
    # Reset error state
    console.print("\n[dim]Resetting error state...[/dim]")
    error_handler.reset_error_state()


@handle_cli_errors(context_factory=create_command_context("demo_command"))
def demo_command_with_context(command_arg: str):
    """Demonstrate command with error context."""
    if command_arg == "fail":
        raise ValidationError("Demo command intentionally failed")
    return f"Command executed successfully with arg: {command_arg}"


async def demo_async_retry():
    """Demonstrate async retry mechanism."""
    console = Console()
    
    attempt_count = 0
    
    async def async_flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        console.print(f"[dim]Async attempt {attempt_count}[/dim]")
        
        if attempt_count < 2:
            raise NetworkError("Async network error")
        return f"Async success on attempt {attempt_count}"
    
    error_handler = CLIErrorHandler(console=console)
    config = RetryConfig(max_attempts=3, base_delay=0.1)
    
    console.print("Testing async retry mechanism:")
    try:
        result = await error_handler.retry_with_backoff(
            async_flaky_function,
            retry_config=config
        )
        console.print(f"[green]✅ {result}[/green]")
    except Exception as e:
        console.print(f"[red]❌ {e}[/red]")


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Run async demo
    print("\n" + "="*50)
    print("Async Demo:")
    asyncio.run(demo_async_retry())