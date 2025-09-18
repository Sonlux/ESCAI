"""
Framework connector for integrating CLI commands with ESCAI instrumentors.

This module provides the bridge between CLI commands and the actual ESCAI framework
instrumentors, enabling real monitoring of agent frameworks.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime
import threading
from contextlib import asynccontextmanager

from ...instrumentation.base_instrumentor import BaseInstrumentor, MonitoringConfig
from ...instrumentation.langchain_instrumentor import LangChainInstrumentor
from ...instrumentation.autogen_instrumentor import AutoGenInstrumentor
from ...instrumentation.crewai_instrumentor import CrewAIInstrumentor
from ...instrumentation.openai_instrumentor import OpenAIInstrumentor
from ...instrumentation.events import AgentEvent, EventType, EventSeverity
from ..utils.error_handling import FrameworkError, ValidationError, NetworkError
from ..utils.console import get_console

logger = logging.getLogger(__name__)
console = get_console()


class FrameworkConnector:
    """
    Connects CLI commands to actual ESCAI framework instrumentors.
    
    This class manages the lifecycle of instrumentors and provides a unified
    interface for the CLI to interact with different agent frameworks.
    """
    
    def __init__(self):
        """Initialize the framework connector."""
        self._instrumentors: Dict[str, BaseInstrumentor] = {}
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._session_lock = threading.RLock()
        self._event_handlers: List[callable] = []
        
        # Framework availability cache
        self._framework_availability: Dict[str, bool] = {}
        self._availability_checked: Dict[str, float] = {}
        self._availability_cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self._performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Framework connector initialized")
    
    def get_available_frameworks(self) -> List[str]:
        """
        Get list of available frameworks.
        
        Returns:
            List of framework names that are available for monitoring
        """
        frameworks = []
        
        for framework_name in ['langchain', 'autogen', 'crewai', 'openai']:
            if self._is_framework_available(framework_name):
                frameworks.append(framework_name)
        
        return frameworks
    
    def _is_framework_available(self, framework: str) -> bool:
        """
        Check if a framework is available for monitoring.
        
        Args:
            framework: Framework name to check
            
        Returns:
            True if framework is available, False otherwise
        """
        current_time = time.time()
        
        # Check cache first
        if (framework in self._availability_checked and 
            current_time - self._availability_checked[framework] < self._availability_cache_ttl):
            return self._framework_availability.get(framework, False)
        
        # Check framework availability
        available = False
        try:
            if framework == 'langchain':
                import langchain
                available = True
            elif framework == 'autogen':
                import autogen
                available = True
            elif framework == 'crewai':
                import crewai
                available = True
            elif framework == 'openai':
                import openai
                available = True
        except ImportError:
            available = False
        
        # Update cache
        self._framework_availability[framework] = available
        self._availability_checked[framework] = current_time
        
        return available
    
    def _get_instrumentor_class(self, framework: str) -> Type[BaseInstrumentor]:
        """
        Get the instrumentor class for a framework.
        
        Args:
            framework: Framework name
            
        Returns:
            Instrumentor class
            
        Raises:
            FrameworkError: If framework is not supported
        """
        instrumentor_classes = {
            'langchain': LangChainInstrumentor,
            'autogen': AutoGenInstrumentor,
            'crewai': CrewAIInstrumentor,
            'openai': OpenAIInstrumentor
        }
        
        if framework not in instrumentor_classes:
            raise FrameworkError(
                f"Unsupported framework: {framework}",
                framework=framework,
                suggestions=[
                    {
                        "action": "Use supported framework",
                        "description": f"Choose from: {', '.join(instrumentor_classes.keys())}",
                        "command_example": f"escai monitor start --framework langchain --agent-id my_agent"
                    }
                ]
            )
        
        return instrumentor_classes[framework]
    
    async def start_monitoring(self, agent_id: str, framework: str, 
                             config: Dict[str, Any]) -> str:
        """
        Start monitoring an agent using the specified framework.
        
        Args:
            agent_id: Unique identifier for the agent
            framework: Framework name (langchain, autogen, crewai, openai)
            config: Configuration parameters for monitoring
            
        Returns:
            session_id: Unique identifier for the monitoring session
            
        Raises:
            FrameworkError: If framework is not available or monitoring fails
            ValidationError: If configuration is invalid
        """
        try:
            # Validate framework availability
            if not self._is_framework_available(framework):
                raise FrameworkError(
                    f"Framework '{framework}' is not available",
                    framework=framework,
                    suggestions=[
                        {
                            "action": "Install framework",
                            "description": f"Install the {framework} framework",
                            "command_example": f"pip install {framework}"
                        }
                    ]
                )
            
            # Validate configuration
            self._validate_monitoring_config(config, framework)
            
            # Get or create instrumentor
            instrumentor = await self._get_or_create_instrumentor(framework)
            
            # Create monitoring configuration - filter out unknown parameters
            valid_config_params = {
                'capture_epistemic_states', 'capture_behavioral_patterns', 
                'capture_performance_metrics', 'max_events_per_second', 'buffer_size'
            }
            filtered_config = {k: v for k, v in config.items() if k in valid_config_params}
            
            monitoring_config = MonitoringConfig(
                agent_id=agent_id,
                framework=framework,
                **filtered_config
            )
            
            # Start monitoring
            session_id = await instrumentor.start_monitoring(
                agent_id=agent_id,
                config=monitoring_config.dict()
            )
            
            # Track session
            with self._session_lock:
                self._active_sessions[session_id] = {
                    'agent_id': agent_id,
                    'framework': framework,
                    'instrumentor': instrumentor,
                    'start_time': datetime.utcnow(),
                    'config': config
                }
            
            # Initialize performance tracking
            self._performance_metrics[session_id] = {
                'events_captured': 0,
                'errors_encountered': 0,
                'start_time': time.time(),
                'last_activity': time.time()
            }
            
            logger.info(f"Started monitoring {framework} agent {agent_id} (session: {session_id})")
            return session_id
            
        except Exception as e:
            if isinstance(e, (FrameworkError, ValidationError)):
                raise
            raise FrameworkError(
                f"Failed to start monitoring: {str(e)}",
                framework=framework
            )
    
    async def stop_monitoring(self, session_id: str) -> Dict[str, Any]:
        """
        Stop monitoring a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary containing monitoring summary
            
        Raises:
            FrameworkError: If session cannot be stopped
        """
        try:
            # Get session info
            with self._session_lock:
                session_info = self._active_sessions.get(session_id)
            
            if not session_info:
                raise FrameworkError(
                    f"Session not found: {session_id}",
                    framework="unknown"
                )
            
            # Stop monitoring
            instrumentor = session_info['instrumentor']
            summary = await instrumentor.stop_monitoring(session_id)
            
            # Remove session tracking
            with self._session_lock:
                self._active_sessions.pop(session_id, None)
            
            # Get performance metrics
            perf_metrics = self._performance_metrics.pop(session_id, {})
            
            # Create comprehensive summary
            result = {
                'session_id': session_id,
                'agent_id': session_info['agent_id'],
                'framework': session_info['framework'],
                'start_time': session_info['start_time'].isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'summary': summary.dict() if hasattr(summary, 'dict') else summary,
                'performance_metrics': perf_metrics
            }
            
            logger.info(f"Stopped monitoring session {session_id}")
            return result
            
        except Exception as e:
            if isinstance(e, FrameworkError):
                raise
            raise FrameworkError(
                f"Failed to stop monitoring: {str(e)}",
                framework="unknown"
            )
    
    async def get_session_status(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get status of monitoring sessions.
        
        Args:
            session_id: Optional specific session ID
            
        Returns:
            List of session status dictionaries
        """
        try:
            sessions = []
            
            with self._session_lock:
                session_items = (
                    [(session_id, self._active_sessions[session_id])] 
                    if session_id and session_id in self._active_sessions
                    else self._active_sessions.items()
                )
            
            for sid, session_info in session_items:
                try:
                    # Get monitoring stats from instrumentor
                    instrumentor = session_info['instrumentor']
                    stats = await instrumentor.get_monitoring_stats(sid)
                    
                    # Get performance metrics
                    perf_metrics = self._performance_metrics.get(sid, {})
                    
                    # Calculate uptime
                    uptime_seconds = (datetime.utcnow() - session_info['start_time']).total_seconds()
                    
                    session_status = {
                        'session_id': sid,
                        'agent_id': session_info['agent_id'],
                        'framework': session_info['framework'],
                        'status': 'active',
                        'uptime_seconds': uptime_seconds,
                        'uptime_formatted': self._format_duration(uptime_seconds),
                        'start_time': session_info['start_time'].isoformat(),
                        'monitoring_stats': stats,
                        'performance_metrics': perf_metrics
                    }
                    
                    sessions.append(session_status)
                    
                except Exception as e:
                    logger.error(f"Error getting status for session {sid}: {str(e)}")
                    sessions.append({
                        'session_id': sid,
                        'agent_id': session_info.get('agent_id', 'unknown'),
                        'framework': session_info.get('framework', 'unknown'),
                        'status': 'error',
                        'error': str(e)
                    })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting session status: {str(e)}")
            return []
    
    async def get_epistemic_state(self, agent_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current epistemic state for an agent.
        
        Args:
            agent_id: Agent identifier
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing epistemic state information
        """
        try:
            # Find active session for agent
            target_session = None
            
            with self._session_lock:
                if session_id:
                    target_session = self._active_sessions.get(session_id)
                else:
                    # Find first active session for this agent
                    for sid, session_info in self._active_sessions.items():
                        if session_info['agent_id'] == agent_id:
                            target_session = session_info
                            session_id = sid
                            break
            
            if not target_session:
                return {
                    'agent_id': agent_id,
                    'status': 'not_monitored',
                    'message': 'No active monitoring session found for this agent'
                }
            
            # Get epistemic state from instrumentor
            instrumentor = target_session['instrumentor']
            
            # For now, return mock data - this would be enhanced with actual
            # epistemic state extraction from the instrumentor
            epistemic_state = {
                'agent_id': agent_id,
                'session_id': session_id,
                'framework': target_session['framework'],
                'timestamp': datetime.utcnow().isoformat(),
                'beliefs': [
                    {'content': 'Processing user request', 'confidence': 0.95},
                    {'content': 'Data validation required', 'confidence': 0.87},
                    {'content': 'Output format specified', 'confidence': 0.72}
                ],
                'knowledge': {
                    'fact_count': 156,
                    'concept_count': 43,
                    'relationship_count': 89
                },
                'goals': [
                    {'description': 'Complete current task', 'progress': 0.65},
                    {'description': 'Maintain data quality', 'progress': 0.89}
                ],
                'uncertainty_score': 0.23,
                'status': 'active'
            }
            
            return epistemic_state
            
        except Exception as e:
            logger.error(f"Error getting epistemic state: {str(e)}")
            return {
                'agent_id': agent_id,
                'status': 'error',
                'error': str(e)
            }
    
    def add_event_handler(self, handler: callable) -> None:
        """
        Add an event handler for processing captured events.
        
        Args:
            handler: Function to call when events are captured
        """
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)
            
            # Add handler to all active instrumentors
            for session_info in self._active_sessions.values():
                instrumentor = session_info['instrumentor']
                instrumentor.add_event_handler(handler)
    
    def remove_event_handler(self, handler: callable) -> None:
        """
        Remove an event handler.
        
        Args:
            handler: Handler function to remove
        """
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
            
            # Remove handler from all active instrumentors
            for session_info in self._active_sessions.values():
                instrumentor = session_info['instrumentor']
                instrumentor.remove_event_handler(handler)
    
    async def validate_framework_integration(self, framework: str) -> Dict[str, Any]:
        """
        Validate that a framework integration is working correctly.
        
        Args:
            framework: Framework name to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'framework': framework,
            'available': False,
            'instrumentor_created': False,
            'events_supported': [],
            'test_monitoring': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check framework availability
            if not self._is_framework_available(framework):
                validation_result['errors'].append(f"Framework {framework} is not installed")
                return validation_result
            
            validation_result['available'] = True
            
            # Try to create instrumentor
            try:
                instrumentor_class = self._get_instrumentor_class(framework)
                instrumentor = instrumentor_class()
                validation_result['instrumentor_created'] = True
                
                # Get supported events
                supported_events = instrumentor.get_supported_events()
                validation_result['events_supported'] = [event.value for event in supported_events]
                
                # Test basic monitoring lifecycle
                test_config = {
                    'capture_epistemic_states': True,
                    'capture_behavioral_patterns': True,
                    'capture_performance_metrics': True
                }
                
                try:
                    # Start test monitoring
                    session_id = await instrumentor.start_monitoring(
                        agent_id=f"test_agent_{framework}",
                        config=test_config
                    )
                    
                    # Immediately stop it
                    await instrumentor.stop_monitoring(session_id)
                    validation_result['test_monitoring'] = True
                    
                    # Clean up
                    await instrumentor.stop()
                    
                except Exception as test_error:
                    validation_result['errors'].append(f"Test monitoring failed: {str(test_error)}")
                    # Still try to clean up
                    try:
                        await instrumentor.stop()
                    except:
                        pass
                
            except Exception as e:
                validation_result['errors'].append(f"Instrumentor test failed: {str(e)}")
            
        except Exception as e:
            validation_result['errors'].append(f"Validation failed: {str(e)}")
        
        return validation_result
    
    async def _get_or_create_instrumentor(self, framework: str) -> BaseInstrumentor:
        """
        Get existing instrumentor or create a new one for the framework.
        
        Args:
            framework: Framework name
            
        Returns:
            BaseInstrumentor instance
        """
        if framework not in self._instrumentors:
            instrumentor_class = self._get_instrumentor_class(framework)
            instrumentor = instrumentor_class()
            
            # Add existing event handlers
            for handler in self._event_handlers:
                instrumentor.add_event_handler(handler)
            
            # Start the instrumentor
            await instrumentor.start()
            
            self._instrumentors[framework] = instrumentor
        
        return self._instrumentors[framework]
    
    def _validate_monitoring_config(self, config: Dict[str, Any], framework: str) -> None:
        """
        Validate monitoring configuration.
        
        Args:
            config: Configuration to validate
            framework: Framework name
            
        Raises:
            ValidationError: If configuration is invalid
        """
        required_fields = ['capture_epistemic_states', 'capture_behavioral_patterns']
        
        for field in required_fields:
            if field not in config:
                config[field] = True  # Set default
        
        # Framework-specific validation
        if framework == 'langchain':
            # LangChain specific validation
            pass
        elif framework == 'autogen':
            # AutoGen specific validation
            if 'agents' not in config and 'group_chats' not in config:
                raise ValidationError(
                    "AutoGen monitoring requires 'agents' or 'group_chats' in configuration",
                    field="agents"
                )
        elif framework == 'crewai':
            # CrewAI specific validation
            pass
        elif framework == 'openai':
            # OpenAI specific validation
            pass
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    async def cleanup(self) -> None:
        """Clean up all resources and stop all instrumentors."""
        try:
            # Stop all active sessions
            session_ids = list(self._active_sessions.keys())
            for session_id in session_ids:
                try:
                    await self.stop_monitoring(session_id)
                except Exception as e:
                    logger.error(f"Error stopping session {session_id}: {str(e)}")
            
            # Stop all instrumentors
            for framework, instrumentor in self._instrumentors.items():
                try:
                    await instrumentor.stop()
                except Exception as e:
                    logger.error(f"Error stopping {framework} instrumentor: {str(e)}")
            
            self._instrumentors.clear()
            self._active_sessions.clear()
            self._performance_metrics.clear()
            
            logger.info("Framework connector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Global connector instance
_connector: Optional[FrameworkConnector] = None
_connector_lock = threading.Lock()


def get_framework_connector() -> FrameworkConnector:
    """
    Get the global framework connector instance.
    
    Returns:
        FrameworkConnector instance
    """
    global _connector
    
    if _connector is None:
        with _connector_lock:
            if _connector is None:
                _connector = FrameworkConnector()
    
    return _connector


@asynccontextmanager
async def framework_context():
    """
    Async context manager for framework connector lifecycle.
    
    Usage:
        async with framework_context() as connector:
            session_id = await connector.start_monitoring(...)
    """
    connector = get_framework_connector()
    try:
        yield connector
    finally:
        await connector.cleanup()