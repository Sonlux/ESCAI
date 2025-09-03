"""
Retry mechanisms with exponential backoff for external services.

This module provides decorators and utilities for implementing robust retry logic
with configurable backoff strategies, jitter, and failure handling.
"""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, List, Optional, Type, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .exceptions import NetworkError, ServiceUnavailableError, ConnectionTimeoutError


logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry mechanisms."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    jitter_range: float = 0.1
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        NetworkError,
        ServiceUnavailableError,
        ConnectionTimeoutError,
        ConnectionError,
        TimeoutError
    )
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()


class RetryExhaustedException(Exception):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. Last error: {last_exception}"
        )


class RetryManager:
    """Manages retry logic with configurable backoff strategies."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = self.config.base_delay * (2 ** (attempt - 1))
            jitter = base_delay * self.config.jitter_range * (2 * random.random() - 1)  # nosec B311
            delay = base_delay + jitter
        else:
            delay = self.config.base_delay
        
        return min(delay, self.config.max_delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check non-retryable exceptions first
        if self.config.non_retryable_exceptions and isinstance(exception, self.config.non_retryable_exceptions):
            return False
        
        # Check retryable exceptions
        return isinstance(exception, self.config.retryable_exceptions)
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute an async function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.debug(f"Executing {func.__name__}, attempt {attempt}/{self.config.max_attempts}")
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise e
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts exhausted for {func.__name__}")
        
        raise RetryExhaustedException(self.config.max_attempts, last_exception)
    
    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a synchronous function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.debug(f"Executing {func.__name__}, attempt {attempt}/{self.config.max_attempts}")
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise e
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts exhausted for {func.__name__}")
        
        raise RetryExhaustedException(self.config.max_attempts, last_exception)


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
    retryable_exceptions: Tuple[Type[Exception], ...] = None,
    non_retryable_exceptions: Tuple[Type[Exception], ...] = None
):
    """
    Decorator for async functions with retry logic.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_strategy: Strategy for calculating retry delays
        retryable_exceptions: Tuple of exception types that should trigger retries
        non_retryable_exceptions: Tuple of exception types that should not trigger retries
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_strategy=backoff_strategy,
                retryable_exceptions=retryable_exceptions or RetryConfig().retryable_exceptions,
                non_retryable_exceptions=non_retryable_exceptions or ()
            )
            retry_manager = RetryManager(config)
            return await retry_manager.execute_async(func, *args, **kwargs)
        
        return wrapper
    return decorator


def retry_sync(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
    retryable_exceptions: Tuple[Type[Exception], ...] = None,
    non_retryable_exceptions: Tuple[Type[Exception], ...] = None
):
    """
    Decorator for synchronous functions with retry logic.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_strategy: Strategy for calculating retry delays
        retryable_exceptions: Tuple of exception types that should trigger retries
        non_retryable_exceptions: Tuple of exception types that should not trigger retries
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_strategy=backoff_strategy,
                retryable_exceptions=retryable_exceptions or RetryConfig().retryable_exceptions,
                non_retryable_exceptions=non_retryable_exceptions or ()
            )
            retry_manager = RetryManager(config)
            return retry_manager.execute_sync(func, *args, **kwargs)
        
        return wrapper
    return decorator


# Predefined retry configurations for common scenarios
DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError)
)

API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
    retryable_exceptions=(NetworkError, ServiceUnavailableError, ConnectionTimeoutError)
)

ML_MODEL_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=2.0,
    max_delay=20.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    retryable_exceptions=(RuntimeError, OSError)
)


# Convenience decorators with predefined configurations
def retry_database(func: Callable):
    """Decorator for database operations with appropriate retry configuration."""
    if asyncio.iscoroutinefunction(func):
        return retry_async(**DATABASE_RETRY_CONFIG.__dict__)(func)
    else:
        return retry_sync(**DATABASE_RETRY_CONFIG.__dict__)(func)


def retry_api(func: Callable):
    """Decorator for API calls with appropriate retry configuration."""
    if asyncio.iscoroutinefunction(func):
        return retry_async(**API_RETRY_CONFIG.__dict__)(func)
    else:
        return retry_sync(**API_RETRY_CONFIG.__dict__)(func)


def retry_ml_model(func: Callable):
    """Decorator for ML model operations with appropriate retry configuration."""
    if asyncio.iscoroutinefunction(func):
        return retry_async(**ML_MODEL_RETRY_CONFIG.__dict__)(func)
    else:
        return retry_sync(**ML_MODEL_RETRY_CONFIG.__dict__)(func)