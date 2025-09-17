"""
Error handling decorators for CLI commands.

This module provides decorators to automatically handle errors in CLI commands
with appropriate user messaging and logging.
"""

import functools
import asyncio
from typing import Any, Callable, Optional, Type, Union

from .error_handling import (
    CLIError, CLIErrorHandler, ErrorCategory, ErrorContext, ErrorSeverity,
    NetworkError, FrameworkError, ValidationError, ConfigurationError,
    RetryConfig, create_error_with_suggestions
)


def handle_cli_errors(
    error_handler: Optional[CLIErrorHandler] = None,
    context_factory: Optional[Callable[..., ErrorContext]] = None
):
    """
    Decorator to handle CLI errors automatically.
    
    Args:
        error_handler: Custom error handler instance
        context_factory: Function to create error context from function args
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or CLIErrorHandler()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    # For async functions, we need to handle them differently
                    # Return a coroutine that can be awaited
                    return _async_wrapper(func, handler, context_factory, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except CLIError as e:
                context = None
                if context_factory:
                    try:
                        context = context_factory(*args, **kwargs)
                    except Exception:
                        pass  # Don't fail on context creation
                        
                handler.handle_error(e, context)
                return None
                
            except Exception as e:
                context = None
                if context_factory:
                    try:
                        context = context_factory(*args, **kwargs)
                    except Exception:
                        pass
                        
                handler.handle_error(e, context)
                return None
                
        return wrapper
    return decorator


async def _async_wrapper(func, handler, context_factory, *args, **kwargs):
    """Async wrapper for error handling."""
    try:
        return await func(*args, **kwargs)
    except CLIError as e:
        context = None
        if context_factory:
            try:
                context = context_factory(*args, **kwargs)
            except Exception:
                pass
                
        handler.handle_error(e, context)
        return None
        
    except Exception as e:
        context = None
        if context_factory:
            try:
                context = context_factory(*args, **kwargs)
            except Exception:
                pass
                
        handler.handle_error(e, context)
        return None


def retry_on_network_error(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """
    Decorator to retry functions on network errors.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = CLIErrorHandler()
            retry_config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay
            )
            
            return await handler.retry_with_backoff(
                func, *args, retry_config=retry_config, **kwargs
            )
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def validate_input(
    validation_func: Callable[[Any], bool],
    error_message: str,
    field_name: Optional[str] = None
):
    """
    Decorator to validate function input.
    
    Args:
        validation_func: Function that returns True if input is valid
        error_message: Error message to display if validation fails
        field_name: Name of the field being validated
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate all arguments
            all_args = list(args) + list(kwargs.values())
            for arg in all_args:
                if not validation_func(arg):
                    raise ValidationError(
                        message=error_message,
                        field=field_name
                    )
                    
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


def require_framework(framework_name: str):
    """
    Decorator to ensure a specific framework is available.
    
    Args:
        framework_name: Name of the required framework
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Try to import the framework
                if framework_name.lower() == 'langchain':
                    __import__('langchain')
                elif framework_name.lower() == 'autogen':
                    __import__('autogen')
                elif framework_name.lower() == 'crewai':
                    __import__('crewai')
                elif framework_name.lower() == 'openai':
                    __import__('openai')
                elif framework_name.lower() == 'json':  # For testing
                    __import__('json')
                else:
                    raise ImportError(f"Unknown framework: {framework_name}")
                    
            except ImportError as e:
                raise FrameworkError(
                    message=f"Framework '{framework_name}' is not available: {e}",
                    framework=framework_name
                )
                
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


def graceful_degradation(
    fallback_func: Optional[Callable] = None,
    degraded_message: Optional[str] = None
):
    """
    Decorator to provide graceful degradation on errors.
    
    Args:
        fallback_func: Function to call if the main function fails
        degraded_message: Message to display when using degraded functionality
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = CLIErrorHandler()
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                if fallback_func:
                    if degraded_message:
                        handler.console.print(f"[yellow]{degraded_message}[/yellow]")
                        
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        # If fallback also fails, handle the original error
                        handler.handle_error(e)
                        return None
                else:
                    handler.handle_error(e)
                    return None
                    
        return wrapper
    return decorator


def log_performance(threshold_seconds: float = 1.0):
    """
    Decorator to log performance warnings for slow operations.
    
    Args:
        threshold_seconds: Time threshold in seconds to trigger warning
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time > threshold_seconds:
                    handler = CLIErrorHandler()
                    handler.console.print(
                        f"[yellow]Performance warning: {func.__name__} took "
                        f"{execution_time:.2f}s (threshold: {threshold_seconds}s)[/yellow]"
                    )
                    
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                # Add execution time to error context
                if hasattr(e, 'context') and e.context:
                    if not e.context.additional_data:
                        e.context.additional_data = {}
                    e.context.additional_data['execution_time'] = execution_time
                    
                raise e
                
        return wrapper
    return decorator


# Context factory functions for common CLI patterns
def create_command_context(command_name: str, session_id: Optional[str] = None):
    """Create error context for CLI commands."""
    def context_factory(*args, **kwargs):
        return ErrorContext(
            command=command_name,
            session_id=session_id,
            user_input=str(kwargs) if kwargs else str(args),
            additional_data={
                'args': args,
                'kwargs': kwargs
            }
        )
    return context_factory


def create_monitoring_context(agent_type: str, session_id: Optional[str] = None):
    """Create error context for monitoring operations."""
    def context_factory(*args, **kwargs):
        return ErrorContext(
            command=f"monitor_{agent_type}",
            session_id=session_id,
            additional_data={
                'agent_type': agent_type,
                'args': args,
                'kwargs': kwargs
            }
        )
    return context_factory


def create_analysis_context(analysis_type: str, session_id: Optional[str] = None):
    """Create error context for analysis operations."""
    def context_factory(*args, **kwargs):
        return ErrorContext(
            command=f"analyze_{analysis_type}",
            session_id=session_id,
            additional_data={
                'analysis_type': analysis_type,
                'args': args,
                'kwargs': kwargs
            }
        )
    return context_factory