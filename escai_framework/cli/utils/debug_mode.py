"""
Debug mode utilities for enhanced CLI debugging and troubleshooting.

This module provides debug mode functionality with verbose output,
interactive debugging, and enhanced error reporting.
"""

import sys
import os
import traceback
import inspect
import pdb
import time
import psutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax
from rich.traceback import Traceback
from rich.progress import Progress, SpinnerColumn, TextColumn

from .logging_system import get_logger, CLILogger


class DebugContext:
    """Context manager for debug operations."""
    
    def __init__(self, operation: str, logger: CLILogger):
        self.operation = operation
        self.logger = logger
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss
        self.console = Console()
    
    def __enter__(self):
        self.logger.debug(f"Starting debug context: {self.operation}")
        if DebugMode.is_verbose():
            self.console.print(f"[dim]🔍 Debug: Starting {self.operation}[/dim]")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        duration = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        if exc_type is not None:
            self.logger.error(
                f"Debug context failed: {self.operation}",
                extra={
                    "duration": duration,
                    "memory_delta_mb": memory_delta / (1024 * 1024),
                    "exception_type": exc_type.__name__,
                    "exception_message": str(exc_val)
                }
            )
            
            if DebugMode.is_interactive():
                self._handle_interactive_debug(exc_type, exc_val, exc_tb)
        else:
            self.logger.debug(
                f"Debug context completed: {self.operation}",
                extra={
                    "duration": duration,
                    "memory_delta_mb": memory_delta / (1024 * 1024)
                }
            )
            
            if DebugMode.is_verbose():
                self.console.print(
                    f"[dim]✅ Debug: Completed {self.operation} "
                    f"({duration:.3f}s, {memory_delta/1024/1024:+.1f}MB)[/dim]"
                )
    
    def _handle_interactive_debug(self, exc_type, exc_val, exc_tb):
        """Handle interactive debugging on exception."""
        console = Console()
        
        console.print("\n[bold red]🚨 Exception occurred in debug mode![/bold red]")
        console.print(f"[red]Operation: {self.operation}[/red]")
        console.print(f"[red]Exception: {exc_type.__name__}: {exc_val}[/red]")
        
        # Show rich traceback
        tb = Traceback.from_exception(exc_type, exc_val, exc_tb)
        console.print(tb)
        
        # Interactive options
        console.print("\n[yellow]Debug options:[/yellow]")
        console.print("1. Enter interactive debugger (pdb)")
        console.print("2. Show local variables")
        console.print("3. Show call stack")
        console.print("4. Continue execution")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                console.print("[cyan]Entering interactive debugger...[/cyan]")
                pdb.post_mortem(exc_tb)
            elif choice == "2":
                self._show_local_variables(exc_tb)
            elif choice == "3":
                self._show_call_stack(exc_tb)
            # Choice 4 or any other input continues execution
            
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Continuing execution...[/yellow]")
    
    def _show_local_variables(self, tb):
        """Show local variables from the traceback."""
        console = Console()
        
        frame = tb.tb_frame
        locals_table = Table(title="Local Variables")
        locals_table.add_column("Variable", style="cyan")
        locals_table.add_column("Type", style="yellow")
        locals_table.add_column("Value", style="green")
        
        for name, value in frame.f_locals.items():
            if not name.startswith('_'):
                value_str = repr(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                
                locals_table.add_row(
                    name,
                    type(value).__name__,
                    value_str
                )
        
        console.print(Panel(locals_table, title="[bold blue]Local Variables[/bold blue]"))
    
    def _show_call_stack(self, tb):
        """Show the call stack."""
        console = Console()
        
        stack_tree = Tree("Call Stack")
        
        current_tb = tb
        while current_tb:
            frame = current_tb.tb_frame
            filename = frame.f_code.co_filename
            line_no = current_tb.tb_lineno
            func_name = frame.f_code.co_name
            
            # Get the source line if possible
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    if 0 <= line_no - 1 < len(lines):
                        source_line = lines[line_no - 1].strip()
                    else:
                        source_line = "Source not available"
            except:
                source_line = "Source not available"
            
            node_text = f"{Path(filename).name}:{line_no} in {func_name}()\n{source_line}"
            stack_tree.add(node_text)
            
            current_tb = current_tb.tb_next
        
        console.print(Panel(stack_tree, title="[bold red]Call Stack[/bold red]"))


class DebugMode:
    """Global debug mode configuration and utilities."""
    
    _enabled = False
    _verbose = False
    _interactive = False
    _trace_calls = False
    _profile_memory = False
    _log_sql = False
    _console = Console()
    _logger = None
    
    @classmethod
    def enable(cls, 
              verbose: bool = True,
              interactive: bool = False,
              trace_calls: bool = False,
              profile_memory: bool = False,
              log_sql: bool = False):
        """Enable debug mode with specified options."""
        cls._enabled = True
        cls._verbose = verbose
        cls._interactive = interactive
        cls._trace_calls = trace_calls
        cls._profile_memory = profile_memory
        cls._log_sql = log_sql
        
        # Initialize logger
        cls._logger = get_logger("debug_mode")
        
        # Set up call tracing if requested
        if trace_calls:
            sys.settrace(cls._trace_function)
        
        cls._console.print("[bold green]Debug mode enabled[/bold green]")
        cls._console.print(f"[dim]Verbose: {verbose}, Interactive: {interactive}, "
                          f"Trace: {trace_calls}, Memory: {profile_memory}[/dim]")
    
    @classmethod
    def disable(cls):
        """Disable debug mode."""
        cls._enabled = False
        cls._verbose = False
        cls._interactive = False
        cls._trace_calls = False
        cls._profile_memory = False
        cls._log_sql = False
        
        # Remove call tracing
        sys.settrace(None)
        
        cls._console.print("[bold yellow]Debug mode disabled[/bold yellow]")
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if debug mode is enabled."""
        return cls._enabled
    
    @classmethod
    def is_verbose(cls) -> bool:
        """Check if verbose mode is enabled."""
        return cls._enabled and cls._verbose
    
    @classmethod
    def is_interactive(cls) -> bool:
        """Check if interactive mode is enabled."""
        return cls._enabled and cls._interactive
    
    @classmethod
    def is_tracing(cls) -> bool:
        """Check if call tracing is enabled."""
        return cls._enabled and cls._trace_calls
    
    @classmethod
    def debug_print(cls, message: str, style: str = "dim"):
        """Print debug message if verbose mode is enabled."""
        if cls.is_verbose():
            cls._console.print(f"[{style}]🔍 {message}[/{style}]")
    
    @classmethod
    def debug_context(cls, operation: str) -> DebugContext:
        """Create a debug context manager."""
        if not cls._logger:
            cls._logger = get_logger("debug_mode")
        return DebugContext(operation, cls._logger)
    
    @classmethod
    def _trace_function(cls, frame, event, arg):
        """Function tracer for call tracing."""
        if event == 'call':
            filename = frame.f_code.co_filename
            func_name = frame.f_code.co_name
            line_no = frame.f_lineno
            
            # Only trace our own code
            if 'escai_framework' in filename and not filename.endswith('debug_mode.py'):
                cls.debug_print(f"CALL: {Path(filename).name}:{line_no} {func_name}()", "blue")
        
        elif event == 'return':
            filename = frame.f_code.co_filename
            func_name = frame.f_code.co_name
            
            if 'escai_framework' in filename and not filename.endswith('debug_mode.py'):
                cls.debug_print(f"RETURN: {func_name}() -> {repr(arg)[:50]}", "green")
        
        return cls._trace_function


def debug_decorator(operation_name: Optional[str] = None):
    """Decorator to add debug context to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if DebugMode.is_enabled():
                with DebugMode.debug_context(op_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def debug_breakpoint(condition: bool = True, message: str = "Debug breakpoint"):
    """Conditional debug breakpoint."""
    if DebugMode.is_enabled() and condition:
        console = Console()
        console.print(f"[bold yellow]🛑 {message}[/bold yellow]")
        
        if DebugMode.is_interactive():
            console.print("[cyan]Entering interactive debugger...[/cyan]")
            pdb.set_trace()
        else:
            console.print("[dim]Set interactive=True to enter debugger[/dim]")


def debug_inspect(obj: Any, name: str = "object", max_depth: int = 2):
    """Inspect and display object details in debug mode."""
    if not DebugMode.is_verbose():
        return
    
    console = Console()
    
    # Basic info
    info_table = Table(title=f"Debug Inspection: {name}")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="yellow")
    
    info_table.add_row("Type", type(obj).__name__)
    info_table.add_row("Module", getattr(type(obj), '__module__', 'unknown'))
    info_table.add_row("ID", str(id(obj)))
    info_table.add_row("Size", f"{sys.getsizeof(obj)} bytes")
    
    # String representation
    try:
        str_repr = str(obj)
        if len(str_repr) > 100:
            str_repr = str_repr[:97] + "..."
        info_table.add_row("String", str_repr)
    except:
        info_table.add_row("String", "Cannot convert to string")
    
    console.print(Panel(info_table, title="[bold blue]Object Inspection[/bold blue]"))
    
    # Attributes (if depth allows)
    if max_depth > 0 and hasattr(obj, '__dict__'):
        attr_table = Table(title="Attributes")
        attr_table.add_column("Name", style="cyan")
        attr_table.add_column("Type", style="yellow")
        attr_table.add_column("Value", style="green")
        
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(obj, attr_name)
                    attr_type = type(attr_value).__name__
                    
                    # Limit value display
                    value_str = repr(attr_value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    
                    attr_table.add_row(attr_name, attr_type, value_str)
                except:
                    attr_table.add_row(attr_name, "unknown", "Cannot access")
        
        console.print(Panel(attr_table, title="[bold green]Attributes[/bold green]"))


def debug_performance(func: Callable) -> Callable:
    """Decorator to measure and log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not DebugMode.is_enabled():
            return func(*args, **kwargs)
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss if DebugMode._profile_memory else 0
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss if DebugMode._profile_memory else 0
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory if DebugMode._profile_memory else 0
            
            DebugMode.debug_print(
                f"PERF: {func.__name__}() took {duration:.3f}s"
                + (f", {memory_delta/1024/1024:+.1f}MB" if DebugMode._profile_memory else ""),
                "magenta"
            )
            
            return result
            
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            DebugMode.debug_print(
                f"PERF: {func.__name__}() failed after {duration:.3f}s: {e}",
                "red"
            )
            raise
    
    return wrapper


@contextmanager
def debug_timer(operation: str):
    """Context manager for timing operations."""
    if not DebugMode.is_verbose():
        yield
        return
    
    console = Console()
    start_time = time.perf_counter()
    
    with console.status(f"[cyan]⏱️  {operation}...[/cyan]"):
        try:
            yield
            end_time = time.perf_counter()
            duration = end_time - start_time
            console.print(f"[green]✅ {operation} completed in {duration:.3f}s[/green]")
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            console.print(f"[red]❌ {operation} failed after {duration:.3f}s: {e}[/red]")
            raise


def debug_memory_usage():
    """Display current memory usage."""
    if not DebugMode.is_verbose():
        return
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    console = Console()
    memory_table = Table(title="Memory Usage")
    memory_table.add_column("Metric", style="cyan")
    memory_table.add_column("Value", style="yellow")
    
    memory_table.add_row("RSS", f"{memory_info.rss / 1024 / 1024:.1f} MB")
    memory_table.add_row("VMS", f"{memory_info.vms / 1024 / 1024:.1f} MB")
    memory_table.add_row("CPU %", f"{process.cpu_percent():.1f}%")
    
    console.print(Panel(memory_table, title="[bold blue]Memory Usage[/bold blue]"))


def debug_system_info():
    """Display system information for debugging."""
    if not DebugMode.is_verbose():
        return
    
    console = Console()
    
    # System info
    system_table = Table(title="System Information")
    system_table.add_column("Property", style="cyan")
    system_table.add_column("Value", style="yellow")
    
    system_table.add_row("Python Version", sys.version.split()[0])
    system_table.add_row("Platform", sys.platform)
    system_table.add_row("CPU Count", str(os.cpu_count()))
    system_table.add_row("Working Directory", str(Path.cwd()))
    
    # Environment variables (selected)
    env_vars = ['PATH', 'PYTHONPATH', 'HOME', 'USER']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if len(value) > 50:
            value = value[:47] + "..."
        system_table.add_row(f"ENV: {var}", value)
    
    console.print(Panel(system_table, title="[bold blue]System Information[/bold blue]"))


# Convenience functions
def enable_debug(verbose: bool = True, interactive: bool = False):
    """Enable debug mode with common settings."""
    DebugMode.enable(verbose=verbose, interactive=interactive)


def disable_debug():
    """Disable debug mode."""
    DebugMode.disable()


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return DebugMode.is_enabled()