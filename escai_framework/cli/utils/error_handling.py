"""
Comprehensive error handling framework for ESCAI CLI.

This module provides centralized error handling with categorized error types,
user-friendly messages, retry mechanisms, and graceful degradation.
"""

import asyncio
import logging
import sys
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from datetime import datetime
import time
import random

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class ErrorCategory(Enum):
    """Categories of errors that can occur in the CLI."""
    VALIDATION = "validation"
    NETWORK = "network"
    FRAMEWORK = "framework"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    DATA_PROCESSING = "data_processing"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE = "resource"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for an error."""
    command: Optional[str] = None
    session_id: Optional[str] = None
    user_input: Optional[str] = None
    timestamp: Optional[datetime] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorSuggestion:
    """Suggestion for resolving an error."""
    action: str
    description: str
    command_example: Optional[str] = None
    documentation_link: Optional[str] = None


class CLIError(Exception):
    """Base exception class for CLI errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suggestions: Optional[List[ErrorSuggestion]] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.suggestions = suggestions or []
        self.context = context or ErrorContext()
        self.original_error = original_error
        self.timestamp = datetime.now()


class ValidationError(CLIError):
    """Error for input validation failures."""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        # Set default severity if not provided
        if 'severity' not in kwargs:
            kwargs['severity'] = ErrorSeverity.LOW
        super().__init__(
            message,
            ErrorCategory.VALIDATION,
            **kwargs
        )
        self.field = field


class NetworkError(CLIError):
    """Error for network-related failures."""
    
    def __init__(self, message: str, endpoint: str = None, **kwargs):
        # Set default severity if not provided
        if 'severity' not in kwargs:
            kwargs['severity'] = ErrorSeverity.MEDIUM
        super().__init__(
            message,
            ErrorCategory.NETWORK,
            **kwargs
        )
        self.endpoint = endpoint


class FrameworkError(CLIError):
    """Error for agent framework integration failures."""
    
    def __init__(self, message: str, framework: str = None, **kwargs):
        # Set default severity if not provided
        if 'severity' not in kwargs:
            kwargs['severity'] = ErrorSeverity.HIGH
        super().__init__(
            message,
            ErrorCategory.FRAMEWORK,
            **kwargs
        )
        self.framework = framework


class ConfigurationError(CLIError):
    """Error for configuration-related failures."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        # Set default severity if not provided
        if 'severity' not in kwargs:
            kwargs['severity'] = ErrorSeverity.MEDIUM
        super().__init__(
            message,
            ErrorCategory.CONFIGURATION,
            **kwargs
        )
        self.config_key = config_key


class SystemError(CLIError):
    """Error for system-level failures."""
    
    def __init__(self, message: str, **kwargs):
        # Set default severity if not provided
        if 'severity' not in kwargs:
            kwargs['severity'] = ErrorSeverity.CRITICAL
        super().__init__(
            message,
            ErrorCategory.SYSTEM,
            **kwargs
        )


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class CLIErrorHandler:
    """Centralized error handler for the CLI."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.logger = logging.getLogger(__name__)
        self._error_counts: Dict[ErrorCategory, int] = {}
        self._degraded_features: set = set()
        
    def handle_error(
        self,
        error: Union[CLIError, Exception],
        context: Optional[ErrorContext] = None
    ) -> bool:
        """
        Handle an error with appropriate user messaging and logging.
        
        Args:
            error: The error to handle
            context: Additional context information
            
        Returns:
            bool: True if the error was handled gracefully, False if critical
        """
        if isinstance(error, CLIError):
            cli_error = error
        else:
            cli_error = self._convert_to_cli_error(error, context)
            
        # Update error context if provided
        if context:
            cli_error.context = context
            
        # Log the error
        self._log_error(cli_error)
        
        # Update error counts
        self._update_error_counts(cli_error)
        
        # Display user-friendly error message
        self._display_error_message(cli_error)
        
        # Check if graceful degradation is needed
        if cli_error.severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(cli_error)
            return False
        elif cli_error.severity == ErrorSeverity.HIGH:
            self._handle_high_severity_error(cli_error)
            
        return True
        
    def _convert_to_cli_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None
    ) -> CLIError:
        """Convert a generic exception to a CLIError."""
        error_type = type(error).__name__
        
        # Map common exception types to CLI error categories
        category_mapping = {
            'ConnectionError': ErrorCategory.NETWORK,
            'TimeoutError': ErrorCategory.NETWORK,
            'ValueError': ErrorCategory.VALIDATION,
            'TypeError': ErrorCategory.VALIDATION,
            'FileNotFoundError': ErrorCategory.SYSTEM,
            'PermissionError': ErrorCategory.PERMISSION,
            'ImportError': ErrorCategory.FRAMEWORK,
            'ModuleNotFoundError': ErrorCategory.FRAMEWORK,
        }
        
        category = category_mapping.get(error_type, ErrorCategory.SYSTEM)
        
        return CLIError(
            message=str(error),
            category=category,
            context=context,
            original_error=error
        )
        
    def _log_error(self, error: CLIError) -> None:
        """Log error details for debugging."""
        log_data = {
            'category': error.category.value,
            'severity': error.severity.value,
            'error_message': error.message,  # Changed from 'message' to avoid conflict
            'timestamp': error.timestamp.isoformat(),
        }
        
        if error.context:
            if error.context.command:
                log_data['command'] = error.context.command
            if error.context.session_id:
                log_data['session_id'] = error.context.session_id
            if error.context.additional_data:
                log_data.update(error.context.additional_data)
                
        if error.original_error:
            log_data['original_error'] = str(error.original_error)
            log_data['traceback'] = traceback.format_exception(
                type(error.original_error),
                error.original_error,
                error.original_error.__traceback__
            )
            
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error("CLI Error", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("CLI Warning", extra=log_data)
        else:
            self.logger.info("CLI Info", extra=log_data)
            
    def _update_error_counts(self, error: CLIError) -> None:
        """Update error counts for monitoring."""
        self._error_counts[error.category] = self._error_counts.get(error.category, 0) + 1
        
    def _display_error_message(self, error: CLIError) -> None:
        """Display user-friendly error message."""
        # Choose color based on severity
        color_mapping = {
            ErrorSeverity.LOW: "yellow",
            ErrorSeverity.MEDIUM: "orange3",
            ErrorSeverity.HIGH: "red",
            ErrorSeverity.CRITICAL: "bright_red"
        }
        
        color = color_mapping.get(error.severity, "red")
        
        # Create error message panel
        title = f"{error.category.value.title()} Error"
        if error.severity == ErrorSeverity.CRITICAL:
            title = f"🚨 Critical {title}"
        elif error.severity == ErrorSeverity.HIGH:
            title = f"⚠️  {title}"
            
        message_text = Text(error.message, style=color)
        
        # Add suggestions if available
        if error.suggestions:
            message_text.append("\n\n💡 Suggestions:\n", style="bold cyan")
            for i, suggestion in enumerate(error.suggestions, 1):
                message_text.append(f"{i}. {suggestion.action}\n", style="cyan")
                message_text.append(f"   {suggestion.description}\n", style="dim cyan")
                if suggestion.command_example:
                    message_text.append(f"   Example: {suggestion.command_example}\n", style="green")
                    
        panel = Panel(
            message_text,
            title=title,
            border_style=color,
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
    def _handle_critical_error(self, error: CLIError) -> None:
        """Handle critical errors that may require system shutdown."""
        self.console.print(
            "\n[bright_red]Critical error detected. The CLI may need to shut down.[/bright_red]"
        )
        
        # Add to degraded features
        self._degraded_features.add("core_functionality")
        
        # Offer recovery options
        self.console.print("\n[yellow]Recovery options:[/yellow]")
        self.console.print("1. Restart the CLI")
        self.console.print("2. Check system requirements")
        self.console.print("3. Contact support with error details")
        
    def _handle_high_severity_error(self, error: CLIError) -> None:
        """Handle high severity errors with graceful degradation."""
        if error.category == ErrorCategory.FRAMEWORK:
            self._degraded_features.add(f"framework_{getattr(error, 'framework', 'unknown')}")
            self.console.print(
                f"\n[yellow]Framework integration degraded. "
                f"Some features may be unavailable.[/yellow]"
            )
        elif error.category == ErrorCategory.NETWORK:
            self._degraded_features.add("network_features")
            self.console.print(
                f"\n[yellow]Network connectivity issues detected. "
                f"Operating in offline mode.[/yellow]"
            )
            
    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            retry_config: Retry configuration
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            The last exception if all retries fail
        """
        config = retry_config or RetryConfig()
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt == config.max_attempts - 1:
                    # Last attempt failed
                    break
                    
                # Calculate delay with exponential backoff
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                
                # Add jitter to prevent thundering herd
                if config.jitter:
                    delay *= (0.5 + random.random() * 0.5)
                    
                self.logger.debug(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
                )
                
                if asyncio.iscoroutinefunction(func):
                    await asyncio.sleep(delay)
                else:
                    time.sleep(delay)
                    
        # All retries failed
        if isinstance(last_exception, Exception):
            raise last_exception
        else:
            raise RuntimeError("All retry attempts failed")
            
    def is_feature_degraded(self, feature: str) -> bool:
        """Check if a feature is currently degraded."""
        return feature in self._degraded_features
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            'error_counts': dict(self._error_counts),
            'degraded_features': list(self._degraded_features),
            'total_errors': sum(self._error_counts.values())
        }
        
    def reset_error_state(self) -> None:
        """Reset error state and degraded features."""
        self._error_counts.clear()
        self._degraded_features.clear()
        self.console.print("[green]Error state reset. All features restored.[/green]")


# Predefined error suggestions
COMMON_SUGGESTIONS = {
    ErrorCategory.VALIDATION: [
        ErrorSuggestion(
            action="Check input format",
            description="Verify that your input matches the expected format",
            command_example="escai --help <command>"
        ),
        ErrorSuggestion(
            action="Use interactive mode",
            description="Try using interactive mode for guided input",
            command_example="escai --interactive"
        )
    ],
    ErrorCategory.NETWORK: [
        ErrorSuggestion(
            action="Check network connection",
            description="Verify your internet connection is working"
        ),
        ErrorSuggestion(
            action="Try offline mode",
            description="Use cached data and local analysis",
            command_example="escai --offline"
        ),
        ErrorSuggestion(
            action="Check firewall settings",
            description="Ensure ESCAI endpoints are not blocked"
        )
    ],
    ErrorCategory.FRAMEWORK: [
        ErrorSuggestion(
            action="Install framework dependencies",
            description="Ensure the required agent framework is installed",
            command_example="pip install langchain autogen crewai"
        ),
        ErrorSuggestion(
            action="Check framework version",
            description="Verify you're using a compatible framework version"
        ),
        ErrorSuggestion(
            action="Use alternative framework",
            description="Try monitoring with a different supported framework"
        )
    ],
    ErrorCategory.CONFIGURATION: [
        ErrorSuggestion(
            action="Run configuration wizard",
            description="Set up your configuration interactively",
            command_example="escai config setup"
        ),
        ErrorSuggestion(
            action="Check configuration file",
            description="Verify your configuration file syntax and values"
        ),
        ErrorSuggestion(
            action="Reset to defaults",
            description="Reset configuration to default values",
            command_example="escai config reset"
        )
    ]
}


def create_error_with_suggestions(
    message: str,
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    additional_suggestions: Optional[List[ErrorSuggestion]] = None,
    **kwargs
) -> CLIError:
    """Create a CLIError with appropriate suggestions."""
    suggestions = COMMON_SUGGESTIONS.get(category, []).copy()
    if additional_suggestions:
        suggestions.extend(additional_suggestions)
        
    return CLIError(
        message=message,
        category=category,
        severity=severity,
        suggestions=suggestions,
        **kwargs
    )