"""
Log processing utilities for the ESCAI framework instrumentation layer.

This module provides utilities for normalizing and processing log data
from different agent frameworks into standardized event structures.
"""

import re
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback

from .events import AgentEvent, EventType, EventSeverity


class LogProcessingError(Exception):
    """Raised when log processing fails."""
    pass


class LogParsingError(LogProcessingError):
    """Raised when log parsing fails."""
    pass


@dataclass
class LogEntry:
    """Represents a raw log entry from an agent framework."""
    timestamp: datetime
    level: str
    message: str
    source: str = ""
    framework: str = ""
    component: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the log entry data."""
        if not isinstance(self.timestamp, datetime):
            return False
        if not isinstance(self.level, str):
            return False
        if not isinstance(self.message, str):
            return False
        if not isinstance(self.source, str):
            return False
        if not isinstance(self.framework, str):
            return False
        if not isinstance(self.component, str):
            return False
        if not isinstance(self.raw_data, dict):
            return False
        return True


@dataclass
class ProcessingRule:
    """Represents a rule for processing log entries into events."""
    name: str
    pattern: str  # Regex pattern to match log messages
    event_type: EventType
    severity_mapping: Dict[str, EventSeverity] = field(default_factory=dict)
    field_extractors: Dict[str, str] = field(default_factory=dict)  # field_name -> regex_group
    condition: Optional[Callable[[LogEntry], bool]] = None
    priority: int = 0  # Higher priority rules are processed first
    
    def matches(self, log_entry: LogEntry) -> bool:
        """Check if this rule matches a log entry."""
        # Check pattern match
        if not re.search(self.pattern, log_entry.message, re.IGNORECASE):
            return False
        
        # Check additional condition if provided
        if self.condition and not self.condition(log_entry):
            return False
        
        return True
    
    def extract_fields(self, log_entry: LogEntry) -> Dict[str, Any]:
        """Extract fields from log entry using configured extractors."""
        fields = {}
        
        match = re.search(self.pattern, log_entry.message, re.IGNORECASE)
        if not match:
            return fields
        
        for field_name, group_name in self.field_extractors.items():
            try:
                if group_name.isdigit():
                    # Numeric group reference (1-based)
                    group_index = int(group_name)
                    fields[field_name] = match.group(group_index)
                else:
                    # Named group reference
                    fields[field_name] = match.group(group_name)
            except (IndexError, KeyError):
                continue
        
        return fields


class LogProcessor:
    """
    Processes raw log entries from agent frameworks and converts them
    to standardized AgentEvent objects.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the log processor.
        
        Args:
            max_workers: Maximum number of worker threads for processing
        """
        self.max_workers = max_workers
        self._processing_rules: List[ProcessingRule] = []
        self._rules_lock = threading.RLock()
        
        # Thread pool for CPU-intensive processing
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix="escai-log-processor"
        )
        
        # Performance metrics
        self._metrics = {
            "logs_processed": 0,
            "events_generated": 0,
            "processing_errors": 0,
            "unmatched_logs": 0,
            "average_processing_time_ms": 0.0
        }
        self._metrics_lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize default processing rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default processing rules for common log patterns."""
        
        # Agent lifecycle events
        self.add_rule(ProcessingRule(
            name="agent_start",
            pattern=r"(?:agent|system|process)\s+(?:start|begin|init)",
            event_type=EventType.AGENT_START,
            severity_mapping={"INFO": EventSeverity.INFO, "DEBUG": EventSeverity.DEBUG},
            priority=10
        ))
        
        self.add_rule(ProcessingRule(
            name="agent_stop",
            pattern=r"(?:agent|system|process)\s+(?:stop|end|finish|complete)",
            event_type=EventType.AGENT_STOP,
            severity_mapping={"INFO": EventSeverity.INFO, "DEBUG": EventSeverity.DEBUG},
            priority=10
        ))
        
        # Task execution events
        self.add_rule(ProcessingRule(
            name="task_start",
            pattern=r"(?:task|job|execution)\s+(?:start|begin)",
            event_type=EventType.TASK_START,
            severity_mapping={"INFO": EventSeverity.INFO, "DEBUG": EventSeverity.DEBUG},
            priority=8
        ))
        
        self.add_rule(ProcessingRule(
            name="task_complete",
            pattern=r"(?:task|job|execution)\s+(?:complete|finish|success)",
            event_type=EventType.TASK_COMPLETE,
            severity_mapping={"INFO": EventSeverity.INFO, "DEBUG": EventSeverity.DEBUG},
            priority=8
        ))
        
        self.add_rule(ProcessingRule(
            name="task_fail",
            pattern=r"(?:task|job|execution)\s+(?:fail|error|exception)",
            event_type=EventType.TASK_FAIL,
            severity_mapping={"ERROR": EventSeverity.ERROR, "WARNING": EventSeverity.WARNING},
            priority=9
        ))
        
        # Decision making events
        self.add_rule(ProcessingRule(
            name="decision_start",
            pattern=r"(?:decision|choice|planning)\s+(?:start|begin)",
            event_type=EventType.DECISION_START,
            severity_mapping={"INFO": EventSeverity.INFO, "DEBUG": EventSeverity.DEBUG},
            priority=7
        ))
        
        # Tool and action events
        self.add_rule(ProcessingRule(
            name="tool_call",
            pattern=r"(?:tool|function|api)\s+(?:call|invoke|execute)",
            event_type=EventType.TOOL_CALL,
            severity_mapping={"INFO": EventSeverity.INFO, "DEBUG": EventSeverity.DEBUG},
            field_extractors={"tool_name": "tool_name", "operation": "operation"},
            priority=6
        ))
        
        # Error events
        self.add_rule(ProcessingRule(
            name="error",
            pattern=r"(?:error|exception|failure|critical)",
            event_type=EventType.AGENT_ERROR,
            severity_mapping={"ERROR": EventSeverity.ERROR, "CRITICAL": EventSeverity.CRITICAL},
            priority=15
        ))
        
        # Performance metrics
        self.add_rule(ProcessingRule(
            name="performance_metric",
            pattern=r"(?:performance|metric|timing|duration|memory|cpu)",
            event_type=EventType.PERFORMANCE_METRIC,
            severity_mapping={"INFO": EventSeverity.INFO, "DEBUG": EventSeverity.DEBUG},
            priority=5
        ))
    
    def add_rule(self, rule: ProcessingRule) -> None:
        """
        Add a processing rule.
        
        Args:
            rule: Processing rule to add
        """
        with self._rules_lock:
            self._processing_rules.append(rule)
            # Sort by priority (higher priority first)
            self._processing_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a processing rule by name.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        with self._rules_lock:
            for i, rule in enumerate(self._processing_rules):
                if rule.name == rule_name:
                    del self._processing_rules[i]
                    return True
        return False
    
    def get_rules(self) -> List[ProcessingRule]:
        """
        Get all processing rules.
        
        Returns:
            List of processing rules sorted by priority
        """
        with self._rules_lock:
            return self._processing_rules.copy()
    
    async def process_log_entry(self, log_entry: LogEntry, 
                               agent_id: str = "", session_id: str = "") -> Optional[AgentEvent]:
        """
        Process a single log entry into an AgentEvent.
        
        Args:
            log_entry: Log entry to process
            agent_id: Agent identifier
            session_id: Session identifier
            
        Returns:
            AgentEvent if processing successful, None otherwise
        """
        if not log_entry.validate():
            raise LogProcessingError("Invalid log entry")
        
        start_time = datetime.utcnow()
        
        try:
            # Find matching rule
            matching_rule = None
            with self._rules_lock:
                for rule in self._processing_rules:
                    if rule.matches(log_entry):
                        matching_rule = rule
                        break
            
            if not matching_rule:
                with self._metrics_lock:
                    self._metrics["unmatched_logs"] += 1
                return None
            
            # Extract fields using the rule
            extracted_fields = matching_rule.extract_fields(log_entry)
            
            # Determine severity
            severity = matching_rule.severity_mapping.get(
                log_entry.level.upper(), 
                EventSeverity.INFO
            )
            
            # Create event
            event = AgentEvent(
                event_type=matching_rule.event_type,
                timestamp=log_entry.timestamp,
                agent_id=agent_id,
                session_id=session_id,
                severity=severity,
                message=log_entry.message,
                framework=log_entry.framework,
                component=log_entry.component,
                data=extracted_fields,
                metadata={
                    "original_level": log_entry.level,
                    "source": log_entry.source,
                    "raw_data": log_entry.raw_data,
                    "processing_rule": matching_rule.name
                }
            )
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            with self._metrics_lock:
                self._metrics["logs_processed"] += 1
                self._metrics["events_generated"] += 1
                
                # Update rolling average
                current_avg = self._metrics["average_processing_time_ms"]
                count = self._metrics["logs_processed"]
                self._metrics["average_processing_time_ms"] = (
                    (current_avg * (count - 1) + processing_time) / count
                )
            
            return event
            
        except Exception as e:
            with self._metrics_lock:
                self._metrics["processing_errors"] += 1
            
            self.logger.error(f"Error processing log entry: {str(e)}")
            raise LogProcessingError(f"Failed to process log entry: {str(e)}")
    
    async def process_log_batch(self, log_entries: List[LogEntry], 
                               agent_id: str = "", session_id: str = "") -> List[AgentEvent]:
        """
        Process a batch of log entries into AgentEvents.
        
        Args:
            log_entries: List of log entries to process
            agent_id: Agent identifier
            session_id: Session identifier
            
        Returns:
            List of generated AgentEvents
        """
        if not log_entries:
            return []
        
        events = []
        
        # Process entries in parallel using thread pool
        loop = asyncio.get_event_loop()
        tasks = []
        
        for log_entry in log_entries:
            task = loop.run_in_executor(
                self._thread_pool,
                self._process_log_entry_sync,
                log_entry, agent_id, session_id
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing error: {str(result)}")
                with self._metrics_lock:
                    self._metrics["processing_errors"] += 1
            elif result is not None:
                events.append(result)
        
        return events
    
    def _process_log_entry_sync(self, log_entry: LogEntry, 
                               agent_id: str = "", session_id: str = "") -> Optional[AgentEvent]:
        """
        Synchronous version of process_log_entry for thread pool execution.
        """
        try:
            # This is a simplified synchronous version
            # In a real implementation, you might want to use asyncio.run()
            # or have separate sync/async processing paths
            
            if not log_entry.validate():
                raise LogProcessingError("Invalid log entry")
            
            # Find matching rule
            matching_rule = None
            with self._rules_lock:
                for rule in self._processing_rules:
                    if rule.matches(log_entry):
                        matching_rule = rule
                        break
            
            if not matching_rule:
                with self._metrics_lock:
                    self._metrics["unmatched_logs"] += 1
                return None
            
            # Extract fields using the rule
            extracted_fields = matching_rule.extract_fields(log_entry)
            
            # Determine severity
            severity = matching_rule.severity_mapping.get(
                log_entry.level.upper(), 
                EventSeverity.INFO
            )
            
            # Create event
            event = AgentEvent(
                event_type=matching_rule.event_type,
                timestamp=log_entry.timestamp,
                agent_id=agent_id,
                session_id=session_id,
                severity=severity,
                message=log_entry.message,
                framework=log_entry.framework,
                component=log_entry.component,
                data=extracted_fields,
                metadata={
                    "original_level": log_entry.level,
                    "source": log_entry.source,
                    "raw_data": log_entry.raw_data,
                    "processing_rule": matching_rule.name
                }
            )
            
            return event
            
        except Exception as e:
            self.logger.error(f"Sync processing error: {str(e)}")
            return None
    
    def parse_log_line(self, log_line: str, framework: str = "", 
                      source: str = "") -> Optional[LogEntry]:
        """
        Parse a raw log line into a LogEntry.
        
        Args:
            log_line: Raw log line string
            framework: Framework name
            source: Log source identifier
            
        Returns:
            LogEntry if parsing successful, None otherwise
        """
        try:
            # Try to parse as JSON first
            if log_line.strip().startswith('{'):
                return self._parse_json_log(log_line, framework, source)
            
            # Try common log formats
            return self._parse_text_log(log_line, framework, source)
            
        except Exception as e:
            self.logger.debug(f"Failed to parse log line: {str(e)}")
            return None
    
    def _parse_json_log(self, log_line: str, framework: str, source: str) -> Optional[LogEntry]:
        """Parse JSON-formatted log line."""
        try:
            data = json.loads(log_line)
            
            # Extract timestamp
            timestamp = datetime.utcnow()
            if 'timestamp' in data:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            elif 'time' in data:
                timestamp = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
            
            # Extract level
            level = data.get('level', data.get('severity', 'INFO')).upper()
            
            # Extract message
            message = data.get('message', data.get('msg', str(data)))
            
            # Extract component
            component = data.get('component', data.get('logger', ''))
            
            return LogEntry(
                timestamp=timestamp,
                level=level,
                message=message,
                source=source,
                framework=framework,
                component=component,
                raw_data=data
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise LogParsingError(f"Failed to parse JSON log: {str(e)}")
    
    def _parse_text_log(self, log_line: str, framework: str, source: str) -> Optional[LogEntry]:
        """Parse text-formatted log line using common patterns."""
        
        # Common log patterns
        patterns = [
            # ISO timestamp with level: 2023-01-01T12:00:00Z [INFO] Message
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s*\[(?P<level>\w+)\]\s*(?P<message>.*)',
            
            # Standard timestamp with level: 2023-01-01 12:00:00 INFO Message
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<level>\w+)\s+(?P<message>.*)',
            
            # Level first: INFO 2023-01-01 12:00:00 Message
            r'(?P<level>\w+)\s+(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<message>.*)',
            
            # Simple level and message: [INFO] Message
            r'\[(?P<level>\w+)\]\s*(?P<message>.*)',
            
            # Just message (assume INFO level) - only if not empty
            r'(?P<message>.+)'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, log_line.strip())
            if match:
                groups = match.groupdict()
                
                # Parse timestamp
                timestamp = datetime.utcnow()
                if 'timestamp' in groups and groups['timestamp']:
                    try:
                        # Try different timestamp formats
                        ts_str = groups['timestamp']
                        if 'T' in ts_str:
                            timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pass  # Use current time if parsing fails
                
                # Extract level
                level = groups.get('level', 'INFO').upper()
                
                # Extract message
                message = groups.get('message', log_line).strip()
                
                return LogEntry(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    source=source,
                    framework=framework,
                    component='',
                    raw_data={'original_line': log_line}
                )
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get processing metrics.
        
        Returns:
            Dictionary of processing metrics
        """
        with self._metrics_lock:
            return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset processing metrics."""
        with self._metrics_lock:
            self._metrics = {
                "logs_processed": 0,
                "events_generated": 0,
                "processing_errors": 0,
                "unmatched_logs": 0,
                "average_processing_time_ms": 0.0
            }
    
    def shutdown(self) -> None:
        """Shutdown the log processor and clean up resources."""
        self._thread_pool.shutdown(wait=True)
        self.logger.info("Log processor shutdown complete")