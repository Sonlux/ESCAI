"""
CLI-specific pytest configuration and fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import os

from escai_framework.cli.utils.console import get_console


@pytest.fixture
def cli_runner():
    """Provide Click CLI test runner"""
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def temp_config_dir():
    """Provide temporary configuration directory"""
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir) / '.escai'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    yield config_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config_file(temp_config_dir):
    """Provide mock configuration file"""
    config_file = temp_config_dir / 'config.json'
    
    default_config = {
        'postgresql': {
            'host': 'localhost',
            'port': 5432,
            'database': 'escai_test',
            'username': 'test_user',
            'password': 'test_pass'
        },
        'api': {
            'host': 'localhost',
            'port': 8000,
            'jwt_secret': 'test-secret-key'
        },
        'monitoring': {
            'max_overhead_percent': 10,
            'buffer_size': 1000,
            'retention_days': 30
        },
        'ui': {
            'color_scheme': 'default',
            'theme': 'default'
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    return config_file


@pytest.fixture
def mock_session_data():
    """Provide mock session data for testing"""
    return [
        {
            'session_id': 'session_001',
            'agent_id': 'agent_001',
            'framework': 'langchain',
            'status': 'active',
            'start_time': '2024-01-15T10:00:00Z',
            'end_time': None,
            'event_count': 150,
            'commands': [
                {
                    'timestamp': '2024-01-15T10:01:00Z',
                    'command': 'monitor start --agent-id agent_001 --framework langchain',
                    'result': 'success'
                },
                {
                    'timestamp': '2024-01-15T10:05:00Z',
                    'command': 'analyze patterns --agent-id agent_001',
                    'result': 'success'
                }
            ]
        },
        {
            'session_id': 'session_002',
            'agent_id': 'agent_002',
            'framework': 'autogen',
            'status': 'completed',
            'start_time': '2024-01-15T09:00:00Z',
            'end_time': '2024-01-15T09:30:00Z',
            'event_count': 75,
            'commands': [
                {
                    'timestamp': '2024-01-15T09:01:00Z',
                    'command': 'monitor start --agent-id agent_002 --framework autogen',
                    'result': 'success'
                },
                {
                    'timestamp': '2024-01-15T09:25:00Z',
                    'command': 'monitor stop --session-id session_002',
                    'result': 'success'
                }
            ]
        }
    ]


@pytest.fixture
def mock_agent_data():
    """Provide mock agent data for testing"""
    return [
        {
            'id': 'agent_001',
            'status': 'active',
            'framework': 'langchain',
            'uptime': '2h 15m',
            'event_count': 1247,
            'last_activity': '2s ago',
            'confidence': 0.92,
            'metadata': {
                'priority': 'high',
                'tags': ['production', 'critical']
            }
        },
        {
            'id': 'agent_002',
            'status': 'idle',
            'framework': 'autogen',
            'uptime': '45m',
            'event_count': 892,
            'last_activity': '1m ago',
            'confidence': 0.78,
            'metadata': {
                'priority': 'medium',
                'tags': ['development', 'testing']
            }
        },
        {
            'id': 'agent_003',
            'status': 'error',
            'framework': 'crewai',
            'uptime': '10m',
            'event_count': 23,
            'last_activity': '5m ago',
            'confidence': 0.34,
            'metadata': {
                'priority': 'low',
                'tags': ['experimental']
            }
        }
    ]


@pytest.fixture
def mock_pattern_data():
    """Provide mock behavioral pattern data for testing"""
    return [
        {
            'pattern_name': 'Sequential Processing',
            'frequency': 45,
            'success_rate': 0.89,
            'average_duration': '2.3s',
            'statistical_significance': 0.95,
            'description': 'Agent processes tasks in sequential order'
        },
        {
            'pattern_name': 'Error Recovery',
            'frequency': 12,
            'success_rate': 0.67,
            'average_duration': '5.1s',
            'statistical_significance': 0.78,
            'description': 'Agent attempts to recover from errors'
        },
        {
            'pattern_name': 'Batch Processing',
            'frequency': 8,
            'success_rate': 0.94,
            'average_duration': '15.2s',
            'statistical_significance': 0.88,
            'description': 'Agent groups similar tasks for batch processing'
        }
    ]


@pytest.fixture
def mock_causal_data():
    """Provide mock causal relationship data for testing"""
    return [
        {
            'cause_event': 'Data Validation Error',
            'effect_event': 'Retry Mechanism Triggered',
            'strength': 0.87,
            'confidence': 0.92,
            'frequency': 23,
            'description': 'Validation errors consistently trigger retry logic'
        },
        {
            'cause_event': 'Large Dataset',
            'effect_event': 'Batch Processing Mode',
            'strength': 0.94,
            'confidence': 0.96,
            'frequency': 15,
            'description': 'Large datasets automatically enable batch processing'
        },
        {
            'cause_event': 'Network Latency',
            'effect_event': 'Timeout Increase',
            'strength': 0.73,
            'confidence': 0.81,
            'frequency': 8,
            'description': 'High latency causes timeout adjustments'
        }
    ]


@pytest.fixture
def mock_prediction_data():
    """Provide mock prediction data for testing"""
    return [
        {
            'predicted_outcome': 'success',
            'confidence': 0.87,
            'risk_factors': ['High memory usage', 'Complex query'],
            'trend': 'improving',
            'time_horizon': '30m',
            'model_accuracy': 0.91
        },
        {
            'predicted_outcome': 'failure',
            'confidence': 0.72,
            'risk_factors': ['Network instability', 'Resource contention'],
            'trend': 'declining',
            'time_horizon': '15m',
            'model_accuracy': 0.85
        },
        {
            'predicted_outcome': 'unknown',
            'confidence': 0.45,
            'risk_factors': ['Insufficient data'],
            'trend': 'stable',
            'time_horizon': '1h',
            'model_accuracy': 0.67
        }
    ]


@pytest.fixture
def mock_epistemic_state():
    """Provide mock epistemic state data for testing"""
    return {
        'agent_id': 'test_agent_001',
        'timestamp': '2024-01-15T10:30:00Z',
        'beliefs': [
            {
                'content': 'User wants to analyze customer data',
                'confidence': 0.95,
                'source': 'user_input'
            },
            {
                'content': 'Data is in CSV format',
                'confidence': 0.87,
                'source': 'file_analysis'
            },
            {
                'content': 'Analysis should focus on trends',
                'confidence': 0.73,
                'source': 'context_inference'
            }
        ],
        'knowledge': {
            'fact_count': 156,
            'concept_count': 43,
            'relationship_count': 89,
            'confidence_avg': 0.82
        },
        'goals': [
            {
                'description': 'Load and validate customer data',
                'progress': 1.0,
                'priority': 'high'
            },
            {
                'description': 'Perform trend analysis',
                'progress': 0.65,
                'priority': 'high'
            },
            {
                'description': 'Generate summary report',
                'progress': 0.0,
                'priority': 'medium'
            }
        ],
        'uncertainty_score': 0.34,
        'reasoning_depth': 3,
        'context_awareness': 0.78
    }


@pytest.fixture
def mock_console():
    """Provide mock console for testing output"""
    with patch('escai_framework.cli.utils.console.get_console') as mock_get_console:
        mock_console_instance = MagicMock()
        mock_get_console.return_value = mock_console_instance
        
        # Mock console methods
        mock_console_instance.print = MagicMock()
        mock_console_instance.log = MagicMock()
        mock_console_instance.rule = MagicMock()
        mock_console_instance.status = MagicMock()
        
        # Mock capture context manager
        mock_capture = MagicMock()
        mock_capture.get.return_value = "Mock console output"
        mock_console_instance.capture.return_value.__enter__.return_value = mock_capture
        mock_console_instance.capture.return_value.__exit__.return_value = None
        
        yield mock_console_instance


@pytest.fixture
def mock_api_client():
    """Provide mock API client for testing"""
    with patch('escai_framework.cli.services.api_client.ESCAIAPIClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock async methods
        mock_client.start_monitoring = MagicMock(return_value={'session_id': 'test_session_123'})
        mock_client.stop_monitoring = MagicMock(return_value={'status': 'stopped'})
        mock_client.get_agent_status = MagicMock(return_value=[])
        mock_client.get_epistemic_state = MagicMock(return_value={})
        mock_client.get_behavioral_patterns = MagicMock(return_value=[])
        mock_client.get_causal_relationships = MagicMock(return_value=[])
        mock_client.get_predictions = MagicMock(return_value=[])
        
        yield mock_client


@pytest.fixture
def mock_framework_connector():
    """Provide mock framework connector for testing"""
    with patch('escai_framework.cli.integration.framework_connector.FrameworkConnector') as mock_connector_class:
        mock_connector = MagicMock()
        mock_connector_class.return_value = mock_connector
        
        # Mock methods
        mock_connector.is_valid_framework = MagicMock(return_value=True)
        mock_connector.start_monitoring = MagicMock(return_value={'session_id': 'test_session'})
        mock_connector.stop_monitoring = MagicMock(return_value={'status': 'stopped'})
        mock_connector.get_agent_status = MagicMock(return_value={'status': 'active'})
        mock_connector.get_all_agent_status = MagicMock(return_value=[])
        
        yield mock_connector


@pytest.fixture
def mock_session_storage():
    """Provide mock session storage for testing"""
    with patch('escai_framework.cli.session_storage.SessionStorage') as mock_storage_class:
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage
        
        # Mock methods
        mock_storage.create_session = MagicMock(return_value='test_session_id')
        mock_storage.get_session = MagicMock(return_value=None)
        mock_storage.get_all_sessions = MagicMock(return_value=[])
        mock_storage.update_session = MagicMock()
        mock_storage.delete_session = MagicMock()
        mock_storage.cleanup_old_sessions = MagicMock(return_value=0)
        
        yield mock_storage


@pytest.fixture
def mock_interactive_menu():
    """Provide mock interactive menu for testing"""
    with patch('escai_framework.cli.utils.interactive_menu.InteractiveMenu') as mock_menu_class:
        mock_menu = MagicMock()
        mock_menu_class.return_value = mock_menu
        
        # Mock methods
        mock_menu.run = MagicMock()
        mock_menu.display_main_menu = MagicMock()
        mock_menu.handle_selection = MagicMock()
        mock_menu.show_breadcrumbs = MagicMock()
        
        yield mock_menu


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables and cleanup"""
    # Set test environment variables
    original_env = os.environ.copy()
    
    os.environ['ESCAI_TEST_MODE'] = 'true'
    os.environ['ESCAI_LOG_LEVEL'] = 'ERROR'  # Reduce log noise in tests
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities for tests"""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.start_time = None
            self.start_memory = None
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            if self.start_time is None:
                raise ValueError("Performance monitoring not started")
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'duration': end_time - self.start_time,
                'memory_used': end_memory - self.start_memory,
                'peak_memory': end_memory
            }
    
    return PerformanceMonitor()


@pytest.fixture
def large_dataset_generator():
    """Provide utilities for generating large test datasets"""
    def generate_agents(count=1000):
        return [
            {
                'id': f'agent_{i:06d}',
                'status': ['active', 'idle', 'error'][i % 3],
                'framework': ['langchain', 'autogen', 'crewai', 'openai'][i % 4],
                'uptime': f'{i % 24}h {i % 60}m',
                'event_count': i * 10 + (i % 100),
                'last_activity': f'{i % 60}s ago',
                'confidence': 0.5 + (i % 50) / 100.0
            }
            for i in range(count)
        ]
    
    def generate_patterns(count=500):
        return [
            {
                'pattern_name': f'Pattern_{i:04d}',
                'frequency': i % 1000 + 1,
                'success_rate': 0.1 + (i % 90) / 100.0,
                'average_duration': f'{(i % 10) + 1}.{i % 10}s',
                'statistical_significance': 0.5 + (i % 50) / 100.0
            }
            for i in range(count)
        ]
    
    return {
        'agents': generate_agents,
        'patterns': generate_patterns
    }


# CLI-specific pytest markers
def pytest_configure(config):
    """Configure CLI-specific pytest markers"""
    config.addinivalue_line(
        "markers", "cli_unit: mark test as CLI unit test"
    )
    config.addinivalue_line(
        "markers", "cli_integration: mark test as CLI integration test"
    )
    config.addinivalue_line(
        "markers", "cli_e2e: mark test as CLI end-to-end test"
    )
    config.addinivalue_line(
        "markers", "cli_performance: mark test as CLI performance test"
    )
    config.addinivalue_line(
        "markers", "cli_ux: mark test as CLI user experience test"
    )
    config.addinivalue_line(
        "markers", "cli_documentation: mark test as CLI documentation test"
    )
    config.addinivalue_line(
        "markers", "cli_accessibility: mark test as CLI accessibility test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )


# CLI test collection hook
def pytest_collection_modifyitems(config, items):
    """Modify test collection for CLI tests"""
    for item in items:
        # Add markers based on test file location
        if "test_cli_commands" in item.nodeid:
            item.add_marker(pytest.mark.cli_unit)
        elif "test_cli_framework_integration" in item.nodeid:
            item.add_marker(pytest.mark.cli_integration)
        elif "test_cli_workflows" in item.nodeid:
            item.add_marker(pytest.mark.cli_e2e)
        elif "test_cli_performance" in item.nodeid:
            item.add_marker(pytest.mark.cli_performance)
            item.add_marker(pytest.mark.slow)
        elif "test_cli_user_experience" in item.nodeid:
            item.add_marker(pytest.mark.cli_ux)
        elif "test_cli_documentation" in item.nodeid:
            item.add_marker(pytest.mark.cli_documentation)
        elif "test_cli_accessibility" in item.nodeid:
            item.add_marker(pytest.mark.cli_accessibility)