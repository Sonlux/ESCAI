"""
Log analysis tools for troubleshooting and debugging.

This module provides tools to analyze log files, identify patterns,
and generate troubleshooting reports.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, TaskID


@dataclass
class LogPattern:
    """Represents a pattern found in logs."""
    pattern: str
    count: int
    first_occurrence: datetime
    last_occurrence: datetime
    severity: str
    examples: List[str]


@dataclass
class ErrorSummary:
    """Summary of errors found in logs."""
    error_type: str
    count: int
    first_seen: datetime
    last_seen: datetime
    affected_commands: Set[str]
    stack_traces: List[str]
    suggested_fixes: List[str]


@dataclass
class PerformanceMetrics:
    """Performance metrics extracted from logs."""
    operation: str
    total_executions: int
    avg_duration: float
    min_duration: float
    max_duration: float
    p95_duration: float
    failure_rate: float


class LogAnalyzer:
    """Comprehensive log analysis tool."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.console = Console()
        
        # Common error patterns
        self.error_patterns = {
            'connection_error': r'(connection|network|timeout|unreachable)',
            'permission_error': r'(permission|access|denied|unauthorized)',
            'file_error': r'(file not found|no such file|directory)',
            'validation_error': r'(validation|invalid|malformed)',
            'framework_error': r'(langchain|autogen|crewai|openai).*error',
            'memory_error': r'(memory|out of memory|allocation)',
            'import_error': r'(import|module.*not found)',
        }
        
        # Performance thresholds (in seconds)
        self.performance_thresholds = {
            'fast': 0.1,
            'acceptable': 1.0,
            'slow': 5.0,
            'very_slow': 10.0
        }
    
    def analyze_logs(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    log_level: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive log analysis."""
        
        self.console.print("[bold blue]Starting log analysis...[/bold blue]")
        
        with Progress() as progress:
            task = progress.add_task("Analyzing logs...", total=100)
            
            # Find log files
            log_files = self._find_log_files(start_date, end_date)
            progress.update(task, advance=10)
            
            # Parse log entries
            entries = self._parse_log_entries(log_files, log_level)
            progress.update(task, advance=30)
            
            # Analyze patterns
            patterns = self._analyze_patterns(entries)
            progress.update(task, advance=20)
            
            # Analyze errors
            errors = self._analyze_errors(entries)
            progress.update(task, advance=20)
            
            # Analyze performance
            performance = self._analyze_performance(entries)
            progress.update(task, advance=20)
        
        return {
            'summary': self._generate_summary(entries, patterns, errors, performance),
            'patterns': patterns,
            'errors': errors,
            'performance': performance,
            'recommendations': self._generate_recommendations(patterns, errors, performance)
        }
    
    def _find_log_files(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> List[Path]:
        """Find relevant log files based on date range."""
        log_files = []
        
        for log_file in self.log_dir.glob("*.log*"):
            if start_date or end_date:
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
            
            log_files.append(log_file)
        
        return sorted(log_files, key=lambda f: f.stat().st_mtime)
    
    def _parse_log_entries(self, log_files: List[Path], log_level: Optional[str]) -> List[Dict[str, Any]]:
        """Parse log entries from files."""
        entries = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            # Try to parse as JSON (structured logs)
                            entry = json.loads(line)
                            if log_level and entry.get('level') != log_level:
                                continue
                            entry['source_file'] = log_file.name
                            entry['line_number'] = line_num
                            entries.append(entry)
                        except json.JSONDecodeError:
                            # Handle plain text logs
                            entry = self._parse_plain_text_log(line, log_file.name, line_num)
                            if entry and (not log_level or entry.get('level') == log_level):
                                entries.append(entry)
            
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not parse {log_file.name}: {e}[/yellow]")
        
        return entries
    
    def _parse_plain_text_log(self, line: str, source_file: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse plain text log line."""
        # Basic pattern for common log formats
        pattern = r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\s]*)\s+(\w+)\s+(.+)'
        match = re.match(pattern, line)
        
        if match:
            timestamp_str, level, message = match.groups()
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                return {
                    'timestamp': timestamp_str,
                    'level': level,
                    'message': message,
                    'source_file': source_file,
                    'line_number': line_num
                }
            except ValueError:
                pass
        
        return None
    
    def _analyze_patterns(self, entries: List[Dict[str, Any]]) -> List[LogPattern]:
        """Analyze log patterns and frequencies."""
        patterns = defaultdict(list)
        
        for entry in entries:
            message = entry.get('message', '')
            
            # Extract common patterns
            for pattern_name, pattern_regex in self.error_patterns.items():
                if re.search(pattern_regex, message, re.IGNORECASE):
                    patterns[pattern_name].append(entry)
        
        # Convert to LogPattern objects
        log_patterns = []
        for pattern_name, pattern_entries in patterns.items():
            if pattern_entries:
                timestamps = [
                    datetime.fromisoformat(entry['timestamp']) 
                    for entry in pattern_entries 
                    if 'timestamp' in entry
                ]
                
                if timestamps:
                    log_patterns.append(LogPattern(
                        pattern=pattern_name,
                        count=len(pattern_entries),
                        first_occurrence=min(timestamps),
                        last_occurrence=max(timestamps),
                        severity=self._determine_severity(pattern_entries),
                        examples=[entry.get('message', '')[:100] for entry in pattern_entries[:3]]
                    ))
        
        return sorted(log_patterns, key=lambda p: p.count, reverse=True)
    
    def _analyze_errors(self, entries: List[Dict[str, Any]]) -> List[ErrorSummary]:
        """Analyze errors and generate summaries."""
        error_groups = defaultdict(list)
        
        for entry in entries:
            if entry.get('level') in ['ERROR', 'CRITICAL']:
                message = entry.get('message', '')
                
                # Group similar errors
                error_type = self._classify_error(message)
                error_groups[error_type].append(entry)
        
        # Generate error summaries
        error_summaries = []
        for error_type, error_entries in error_groups.items():
            timestamps = [
                datetime.fromisoformat(entry['timestamp']) 
                for entry in error_entries 
                if 'timestamp' in entry
            ]
            
            commands = {entry.get('command', 'unknown') for entry in error_entries}
            stack_traces = [
                entry.get('stack_trace', '') 
                for entry in error_entries 
                if entry.get('stack_trace')
            ]
            
            if timestamps:
                error_summaries.append(ErrorSummary(
                    error_type=error_type,
                    count=len(error_entries),
                    first_seen=min(timestamps),
                    last_seen=max(timestamps),
                    affected_commands=commands,
                    stack_traces=stack_traces[:3],  # Keep only first 3
                    suggested_fixes=self._suggest_fixes(error_type, error_entries)
                ))
        
        return sorted(error_summaries, key=lambda e: e.count, reverse=True)
    
    def _analyze_performance(self, entries: List[Dict[str, Any]]) -> List[PerformanceMetrics]:
        """Analyze performance metrics from logs."""
        performance_data = defaultdict(list)
        
        for entry in entries:
            if entry.get('level') == 'PERFORMANCE' and 'duration' in entry.get('context', {}):
                operation = entry.get('message', '').split(' ')[1] if ' ' in entry.get('message', '') else 'unknown'
                duration = entry['context']['duration']
                performance_data[operation].append(duration)
        
        # Calculate metrics
        metrics = []
        for operation, durations in performance_data.items():
            if durations:
                failure_count = sum(1 for d in durations if d > self.performance_thresholds['very_slow'])
                
                metrics.append(PerformanceMetrics(
                    operation=operation,
                    total_executions=len(durations),
                    avg_duration=statistics.mean(durations),
                    min_duration=min(durations),
                    max_duration=max(durations),
                    p95_duration=statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else durations[0],
                    failure_rate=failure_count / len(durations)
                ))
        
        return sorted(metrics, key=lambda m: m.avg_duration, reverse=True)
    
    def _classify_error(self, message: str) -> str:
        """Classify error based on message content."""
        message_lower = message.lower()
        
        for error_type, pattern in self.error_patterns.items():
            if re.search(pattern, message_lower):
                return error_type
        
        # Extract exception type if available
        exception_match = re.search(r'(\w+Error|\w+Exception)', message)
        if exception_match:
            return exception_match.group(1)
        
        return 'unknown_error'
    
    def _determine_severity(self, entries: List[Dict[str, Any]]) -> str:
        """Determine severity based on log levels and frequency."""
        levels = [entry.get('level', 'INFO') for entry in entries]
        level_counts = Counter(levels)
        
        if level_counts.get('CRITICAL', 0) > 0:
            return 'CRITICAL'
        elif level_counts.get('ERROR', 0) > 0:
            return 'ERROR'
        elif level_counts.get('WARNING', 0) > len(entries) * 0.5:
            return 'WARNING'
        else:
            return 'INFO'
    
    def _suggest_fixes(self, error_type: str, entries: List[Dict[str, Any]]) -> List[str]:
        """Generate suggested fixes for common errors."""
        suggestions = {
            'connection_error': [
                "Check network connectivity",
                "Verify service endpoints are accessible",
                "Check firewall settings",
                "Increase timeout values"
            ],
            'permission_error': [
                "Check file/directory permissions",
                "Run with appropriate user privileges",
                "Verify API key permissions",
                "Check authentication configuration"
            ],
            'file_error': [
                "Verify file paths are correct",
                "Check if files exist",
                "Ensure proper file permissions",
                "Check disk space availability"
            ],
            'validation_error': [
                "Validate input parameters",
                "Check data format requirements",
                "Verify configuration syntax",
                "Review API documentation"
            ],
            'framework_error': [
                "Update framework dependencies",
                "Check framework compatibility",
                "Review framework configuration",
                "Consult framework documentation"
            ],
            'memory_error': [
                "Increase available memory",
                "Optimize data processing",
                "Use streaming for large datasets",
                "Check for memory leaks"
            ],
            'import_error': [
                "Install missing dependencies",
                "Check Python path configuration",
                "Verify package versions",
                "Update requirements.txt"
            ]
        }
        
        return suggestions.get(error_type, ["Review error details and consult documentation"])
    
    def _generate_summary(self, entries: List[Dict[str, Any]], patterns: List[LogPattern], 
                         errors: List[ErrorSummary], performance: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Generate analysis summary."""
        if not entries:
            return {"message": "No log entries found"}
        
        timestamps = [
            datetime.fromisoformat(entry['timestamp']) 
            for entry in entries 
            if 'timestamp' in entry
        ]
        
        level_counts = Counter(entry.get('level', 'INFO') for entry in entries)
        
        return {
            'total_entries': len(entries),
            'date_range': {
                'start': min(timestamps).isoformat() if timestamps else None,
                'end': max(timestamps).isoformat() if timestamps else None
            },
            'level_distribution': dict(level_counts),
            'total_patterns': len(patterns),
            'total_errors': len(errors),
            'performance_operations': len(performance),
            'health_score': self._calculate_health_score(level_counts, errors, performance)
        }
    
    def _calculate_health_score(self, level_counts: Counter, errors: List[ErrorSummary], 
                               performance: List[PerformanceMetrics]) -> float:
        """Calculate overall system health score (0-100)."""
        total_entries = sum(level_counts.values())
        if total_entries == 0:
            return 100.0
        
        # Base score
        score = 100.0
        
        # Deduct for errors
        error_ratio = (level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)) / total_entries
        score -= error_ratio * 50
        
        # Deduct for warnings
        warning_ratio = level_counts.get('WARNING', 0) / total_entries
        score -= warning_ratio * 20
        
        # Deduct for performance issues
        slow_operations = sum(1 for p in performance if p.avg_duration > self.performance_thresholds['slow'])
        if performance:
            performance_penalty = (slow_operations / len(performance)) * 30
            score -= performance_penalty
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, patterns: List[LogPattern], errors: List[ErrorSummary], 
                                performance: List[PerformanceMetrics]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Error-based recommendations
        if errors:
            top_error = errors[0]
            recommendations.extend([
                f"Address '{top_error.error_type}' errors (occurred {top_error.count} times)",
                *top_error.suggested_fixes[:2]
            ])
        
        # Performance recommendations
        slow_ops = [p for p in performance if p.avg_duration > self.performance_thresholds['acceptable']]
        if slow_ops:
            recommendations.append(f"Optimize performance for {len(slow_ops)} slow operations")
        
        # Pattern-based recommendations
        critical_patterns = [p for p in patterns if p.severity in ['ERROR', 'CRITICAL']]
        if critical_patterns:
            recommendations.append(f"Investigate {len(critical_patterns)} critical patterns")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def generate_report(self, analysis: Dict[str, Any]) -> None:
        """Generate and display a comprehensive analysis report."""
        console = Console()
        
        # Summary panel
        summary = analysis['summary']
        summary_table = Table(title="Log Analysis Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Total Entries", str(summary['total_entries']))
        summary_table.add_row("Health Score", f"{summary['health_score']:.1f}/100")
        summary_table.add_row("Error Count", str(summary.get('total_errors', 0)))
        summary_table.add_row("Pattern Count", str(summary.get('total_patterns', 0)))
        
        console.print(Panel(summary_table, title="[bold blue]Analysis Summary[/bold blue]"))
        
        # Error analysis
        if analysis['errors']:
            error_table = Table(title="Top Errors")
            error_table.add_column("Error Type", style="red")
            error_table.add_column("Count", style="yellow")
            error_table.add_column("Last Seen", style="dim")
            
            for error in analysis['errors'][:5]:
                error_table.add_row(
                    error.error_type,
                    str(error.count),
                    error.last_seen.strftime("%Y-%m-%d %H:%M")
                )
            
            console.print(Panel(error_table, title="[bold red]Error Analysis[/bold red]"))
        
        # Performance analysis
        if analysis['performance']:
            perf_table = Table(title="Performance Metrics")
            perf_table.add_column("Operation", style="cyan")
            perf_table.add_column("Avg Duration", style="yellow")
            perf_table.add_column("P95 Duration", style="yellow")
            perf_table.add_column("Executions", style="dim")
            
            for perf in analysis['performance'][:5]:
                perf_table.add_row(
                    perf.operation,
                    f"{perf.avg_duration:.3f}s",
                    f"{perf.p95_duration:.3f}s",
                    str(perf.total_executions)
                )
            
            console.print(Panel(perf_table, title="[bold green]Performance Analysis[/bold green]"))
        
        # Recommendations
        if analysis['recommendations']:
            rec_text = "\n".join(f"• {rec}" for rec in analysis['recommendations'])
            console.print(Panel(rec_text, title="[bold yellow]Recommendations[/bold yellow]"))


def analyze_cli_logs(log_dir: Optional[Path] = None, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    log_level: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to analyze CLI logs."""
    if log_dir is None:
        log_dir = Path.home() / ".escai" / "logs"
    
    analyzer = LogAnalyzer(log_dir)
    return analyzer.analyze_logs(start_date, end_date, log_level)