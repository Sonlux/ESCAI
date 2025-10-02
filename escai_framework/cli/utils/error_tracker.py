"""
Error tracking system with stack trace preservation and analysis.

This module provides comprehensive error tracking, categorization,
and analysis capabilities for the CLI system.
"""

import sys
import traceback
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import threading
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .logging_system import get_logger


@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    error_id: str
    timestamp: str
    error_type: str
    error_message: str
    stack_trace: str
    module: str
    function: str
    line_number: int
    session_id: Optional[str]
    command: Optional[str]
    user_input: Optional[str]
    context: Dict[str, Any]
    severity: str
    category: str
    fingerprint: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ErrorTracker:
    """Centralized error tracking and analysis system."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".escai" / "error_tracking.db"
        self.console = Console()
        self.logger = get_logger("error_tracker")
        self._lock = threading.Lock()
        
        # Error categories
        self.error_categories = {
            'user_input': ['ValidationError', 'ValueError', 'TypeError'],
            'system': ['OSError', 'IOError', 'PermissionError', 'FileNotFoundError'],
            'network': ['ConnectionError', 'TimeoutError', 'HTTPError'],
            'framework': ['ImportError', 'ModuleNotFoundError', 'AttributeError'],
            'memory': ['MemoryError', 'OverflowError'],
            'runtime': ['RuntimeError', 'KeyError', 'IndexError'],
            'unknown': []
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the error tracking database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    error_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    stack_trace TEXT NOT NULL,
                    module TEXT,
                    function TEXT,
                    line_number INTEGER,
                    session_id TEXT,
                    command TEXT,
                    user_input TEXT,
                    context TEXT,
                    severity TEXT,
                    category TEXT,
                    fingerprint TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON errors(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fingerprint ON errors(fingerprint)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON errors(category)
            """)
    
    def track_error(self, 
                   exc_type: type, 
                   exc_value: Union[Exception, BaseException], 
                   exc_traceback,
                   session_id: Optional[str] = None,
                   command: Optional[str] = None,
                   user_input: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None) -> str:
        """Track an error with full context."""
        
        # Extract error information
        error_type = exc_type.__name__
        error_message = str(exc_value)
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Get caller information from traceback
        tb_frame = exc_traceback.tb_frame
        module = tb_frame.f_code.co_filename
        function = tb_frame.f_code.co_name
        line_number = exc_traceback.tb_lineno
        
        # Generate error fingerprint for deduplication
        fingerprint = self._generate_fingerprint(error_type, error_message, module, function)
        
        # Categorize error
        category = self._categorize_error(error_type)
        
        # Determine severity
        severity = self._determine_severity(exc_type, context)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=self._generate_error_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            module=Path(module).name,
            function=function,
            line_number=line_number,
            session_id=session_id,
            command=command,
            user_input=user_input,
            context=context or {},
            severity=severity,
            category=category,
            fingerprint=fingerprint
        )
        
        # Store in database
        self._store_error(error_info)
        
        # Log the error
        self.logger.error(
            f"Tracked error: {error_type}: {error_message}",
            extra={
                "error_id": error_info.error_id,
                "fingerprint": fingerprint,
                "category": category,
                "severity": severity
            }
        )
        
        return error_info.error_id  
  
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"err_{timestamp}"
    
    def _generate_fingerprint(self, error_type: str, error_message: str, 
                             module: str, function: str) -> str:
        """Generate error fingerprint for deduplication."""
        # Normalize error message (remove dynamic parts)
        normalized_message = self._normalize_error_message(error_message)
        
        # Create fingerprint from stable components
        fingerprint_data = f"{error_type}:{normalized_message}:{Path(module).name}:{function}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]
    
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message by removing dynamic parts."""
        import re
        
        # Remove file paths
        message = re.sub(r'/[^\s]+', '<path>', message)
        message = re.sub(r'[A-Z]:\\[^\s]+', '<path>', message)
        
        # Remove numbers that might be dynamic
        message = re.sub(r'\b\d+\b', '<num>', message)
        
        # Remove memory addresses
        message = re.sub(r'0x[0-9a-fA-F]+', '<addr>', message)
        
        # Remove timestamps
        message = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', '<timestamp>', message)
        
        return message.strip()
    
    def _categorize_error(self, error_type: str) -> str:
        """Categorize error based on type."""
        for category, error_types in self.error_categories.items():
            if error_type in error_types:
                return category
        return 'unknown'
    
    def _determine_severity(self, exc_type: type, context: Optional[Dict[str, Any]]) -> str:
        """Determine error severity."""
        if issubclass(exc_type, (SystemExit, KeyboardInterrupt)):
            return 'info'
        elif issubclass(exc_type, (MemoryError, SystemError)):
            return 'critical'
        elif issubclass(exc_type, (ConnectionError, TimeoutError, OSError)):
            return 'high'
        elif issubclass(exc_type, (ValueError, TypeError, AttributeError)):
            return 'medium'
        else:
            return 'low'
    
    def _store_error(self, error_info: ErrorInfo):
        """Store error information in database."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO errors (
                        error_id, timestamp, error_type, error_message, stack_trace,
                        module, function, line_number, session_id, command, user_input,
                        context, severity, category, fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    error_info.error_id,
                    error_info.timestamp,
                    error_info.error_type,
                    error_info.error_message,
                    error_info.stack_trace,
                    error_info.module,
                    error_info.function,
                    error_info.line_number,
                    error_info.session_id,
                    error_info.command,
                    error_info.user_input,
                    json.dumps(error_info.context),
                    error_info.severity,
                    error_info.category,
                    error_info.fingerprint
                ))
    
    def get_error_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get error statistics for the specified period."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Total errors
            total_errors = conn.execute(
                "SELECT COUNT(*) as count FROM errors WHERE timestamp >= ?",
                (cutoff_date,)
            ).fetchone()['count']
            
            # Errors by category
            category_stats = conn.execute("""
                SELECT category, COUNT(*) as count 
                FROM errors 
                WHERE timestamp >= ? 
                GROUP BY category 
                ORDER BY count DESC
            """, (cutoff_date,)).fetchall()
            
            # Errors by severity
            severity_stats = conn.execute("""
                SELECT severity, COUNT(*) as count 
                FROM errors 
                WHERE timestamp >= ? 
                GROUP BY severity 
                ORDER BY count DESC
            """, (cutoff_date,)).fetchall()
            
            # Top error types
            type_stats = conn.execute("""
                SELECT error_type, COUNT(*) as count 
                FROM errors 
                WHERE timestamp >= ? 
                GROUP BY error_type 
                ORDER BY count DESC 
                LIMIT 10
            """, (cutoff_date,)).fetchall()
            
            # Most frequent errors (by fingerprint)
            frequent_errors = conn.execute("""
                SELECT fingerprint, error_type, error_message, COUNT(*) as count,
                       MAX(timestamp) as last_seen
                FROM errors 
                WHERE timestamp >= ? 
                GROUP BY fingerprint 
                ORDER BY count DESC 
                LIMIT 10
            """, (cutoff_date,)).fetchall()
        
        return {
            'total_errors': total_errors,
            'period_days': days,
            'by_category': [dict(row) for row in category_stats],
            'by_severity': [dict(row) for row in severity_stats],
            'by_type': [dict(row) for row in type_stats],
            'most_frequent': [dict(row) for row in frequent_errors]
        }
    
    def get_similar_errors(self, fingerprint: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar errors based on fingerprint."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            similar_errors = conn.execute("""
                SELECT error_id, timestamp, error_message, command, session_id
                FROM errors 
                WHERE fingerprint = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (fingerprint, limit)).fetchall()
        
        return [dict(row) for row in similar_errors]
    
    def get_error_trends(self, days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Get error trends over time."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Daily error counts
            daily_counts = conn.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM errors 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (cutoff_date,)).fetchall()
            
            # Category trends
            category_trends = conn.execute("""
                SELECT DATE(timestamp) as date, category, COUNT(*) as count
                FROM errors 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp), category
                ORDER BY date, category
            """, (cutoff_date,)).fetchall()
        
        return {
            'daily_counts': [dict(row) for row in daily_counts],
            'category_trends': [dict(row) for row in category_trends]
        }
    
    def generate_error_report(self, days: int = 7) -> None:
        """Generate and display comprehensive error report."""
        stats = self.get_error_stats(days)
        
        # Summary panel
        summary_table = Table(title=f"Error Summary (Last {days} days)")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        
        summary_table.add_row("Total Errors", str(stats['total_errors']))
        
        if stats['by_severity']:
            top_severity = stats['by_severity'][0]
            summary_table.add_row("Top Severity", f"{top_severity['severity']} ({top_severity['count']})")
        
        if stats['by_category']:
            top_category = stats['by_category'][0]
            summary_table.add_row("Top Category", f"{top_category['category']} ({top_category['count']})")
        
        self.console.print(Panel(summary_table, title="[bold red]Error Report[/bold red]"))
        
        # Category breakdown
        if stats['by_category']:
            cat_table = Table(title="Errors by Category")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="yellow")
            cat_table.add_column("Percentage", style="green")
            
            total = stats['total_errors']
            for cat in stats['by_category']:
                percentage = (cat['count'] / total * 100) if total > 0 else 0
                cat_table.add_row(
                    cat['category'],
                    str(cat['count']),
                    f"{percentage:.1f}%"
                )
            
            self.console.print(Panel(cat_table, title="[bold blue]Category Breakdown[/bold blue]"))
        
        # Most frequent errors
        if stats['most_frequent']:
            freq_table = Table(title="Most Frequent Errors")
            freq_table.add_column("Error Type", style="red")
            freq_table.add_column("Message", style="yellow")
            freq_table.add_column("Count", style="cyan")
            freq_table.add_column("Last Seen", style="dim")
            
            for error in stats['most_frequent'][:5]:
                message = error['error_message']
                if len(message) > 50:
                    message = message[:47] + "..."
                
                last_seen = datetime.fromisoformat(error['last_seen']).strftime("%m-%d %H:%M")
                
                freq_table.add_row(
                    error['error_type'],
                    message,
                    str(error['count']),
                    last_seen
                )
            
            self.console.print(Panel(freq_table, title="[bold yellow]Frequent Errors[/bold yellow]"))


# Global error tracker instance
_error_tracker: Optional[ErrorTracker] = None


def initialize_error_tracking(db_path: Optional[Path] = None) -> ErrorTracker:
    """Initialize global error tracking."""
    global _error_tracker
    _error_tracker = ErrorTracker(db_path)
    return _error_tracker


def track_error(exc_type: type, exc_value: Union[Exception, BaseException], exc_traceback,
               session_id: Optional[str] = None,
               command: Optional[str] = None,
               user_input: Optional[str] = None,
               context: Optional[Dict[str, Any]] = None) -> str:
    """Track an error using the global tracker."""
    if _error_tracker is None:
        initialize_error_tracking()
    
    return _error_tracker.track_error(
        exc_type, exc_value, exc_traceback,
        session_id, command, user_input, context
    )


@contextmanager
def error_tracking_context(session_id: Optional[str] = None,
                          command: Optional[str] = None,
                          user_input: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None):
    """Context manager for automatic error tracking."""
    try:
        yield
    except (Exception, BaseException) as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        track_error(exc_type, exc_value, exc_traceback, session_id, command, user_input, context)
        raise


def get_error_stats(days: int = 7) -> Dict[str, Any]:
    """Get error statistics from global tracker."""
    if _error_tracker is None:
        initialize_error_tracking()
    return _error_tracker.get_error_stats(days)


def generate_error_report(days: int = 7) -> None:
    """Generate error report from global tracker."""
    if _error_tracker is None:
        initialize_error_tracking()
    _error_tracker.generate_error_report(days)


# Import required for datetime operations
from datetime import timedelta