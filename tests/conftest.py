"""
Pytest configuration and fixtures for ESCAI Framework tests.
"""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import AsyncMock, MagicMock
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import ESCAI models and components
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState
from escai_framework.models.behavioral_pattern import BehavioralPattern, ExecutionSequence
from escai_framework.models.causal_relationship import CausalRelationship
from escai_framework.models.prediction_result import PredictionResult
from escai_framework.instrumentation.events import AgentEvent, EventType


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_agent_id():
    """Sample agent ID for testing."""
    return "test_agent_001"


@pytest.fixture
def sample_session_id():
    """Sample session ID for testing."""
    return "session_12345"


@pytest.fixture
def sample_belief_state():
    """Sample belief state for testing."""
    return BeliefState(
        belief_id="belief_001",
        content="The user wants to analyze sales data",
        confidence=0.85,
        timestamp=datetime.now(),
        evidence=["User said 'analyze sales'", "Context contains sales.csv"]
    )


@pytest.fixture
def sample_knowledge_state():
    """Sample knowledge state for testing."""
    return KnowledgeState(
        facts=["Sales data is in CSV format", "Data contains 1000 records"],
        concepts=["data analysis", "sales metrics"],
        relationships={"sales": ["revenue", "profit", "customers"]},
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_goal_state():
    """Sample goal state for testing."""
    return GoalState(
        primary_goal="Analyze sales performance",
        sub_goals=["Load data", "Calculate metrics", "Generate report"],
        progress=0.3,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_epistemic_state(sample_agent_id, sample_belief_state, sample_knowledge_state, sample_goal_state):
    """Sample epistemic state for testing."""
    return EpistemicState(
        agent_id=sample_agent_id,
        timestamp=datetime.now(),
        belief_states=[sample_belief_state],
        knowledge_state=sample_knowledge_state,
        goal_state=sample_goal_state,
        confidence_level=0.8,
        uncertainty_score=0.2,
        decision_context={"task": "data_analysis", "priority": "high"}
    )


@pytest.fixture
def sample_execution_sequence():
    """Sample execution sequence for testing."""
    return ExecutionSequence(
        sequence_id="seq_001",
        agent_id="test_agent_001",
        steps=[
            {"action": "load_data", "timestamp": datetime.now(), "success": True},
            {"action": "validate_data", "timestamp": datetime.now(), "success": True},
            {"action": "analyze_data", "timestamp": datetime.now(), "success": False}
        ],
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        success=False,
        error_message="Analysis failed due to missing column"
    )


@pytest.fixture
def sample_behavioral_pattern(sample_execution_sequence):
    """Sample behavioral pattern for testing."""
    return BehavioralPattern(
        pattern_id="pattern_001",
        pattern_name="Data Analysis Workflow",
        execution_sequences=[sample_execution_sequence],
        frequency=15,
        success_rate=0.73,
        average_duration=300.5,
        common_triggers=["data_analysis_request", "csv_file_upload"],
        failure_modes=["missing_column", "invalid_format"],
        statistical_significance=0.95
    )


@pytest.fixture
def sample_causal_relationship():
    """Sample causal relationship for testing."""
    return CausalRelationship(
        cause_event="data_validation_failure",
        effect_event="analysis_task_failure",
        strength=0.82,
        confidence=0.91,
        delay_ms=150,
        evidence=["Temporal correlation", "Statistical significance"],
        statistical_significance=0.99,
        causal_mechanism="Invalid data prevents analysis execution"
    )


@pytest.fixture
def sample_prediction_result():
    """Sample prediction result for testing."""
    return PredictionResult(
        prediction_id="pred_001",
        agent_id="test_agent_001",
        predicted_outcome="success",
        confidence=0.87,
        probability_distribution={"success": 0.87, "failure": 0.13},
        risk_factors=["data_quality", "time_constraint"],
        recommended_actions=["validate_data_first", "increase_timeout"],
        timestamp=datetime.now(),
        model_version="v1.2.3"
    )


@pytest.fixture
def sample_agent_events():
    """Sample agent events for testing."""
    base_time = datetime.now()
    return [
        AgentEvent(
            event_id="event_001",
            agent_id="test_agent_001",
            event_type=EventType.TASK_START,
            timestamp=base_time,
            data={"task": "data_analysis", "priority": "high"}
        ),
        AgentEvent(
            event_id="event_002",
            agent_id="test_agent_001",
            event_type=EventType.DECISION_MADE,
            timestamp=base_time + timedelta(seconds=30),
            data={"decision": "load_csv_data", "confidence": 0.9}
        ),
        AgentEvent(
            event_id="event_003",
            agent_id="test_agent_001",
            event_type=EventType.ERROR_OCCURRED,
            timestamp=base_time + timedelta(minutes=2),
            data={"error": "FileNotFoundError", "message": "sales.csv not found"}
        )
    ]


@pytest.fixture
def sample_time_series_data():
    """Sample time series data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    return pd.DataFrame({
        'timestamp': dates,
        'success_rate': np.random.normal(0.8, 0.1, len(dates)),
        'response_time': np.random.exponential(200, len(dates)),
        'error_count': np.random.poisson(2, len(dates))
    })


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.fetch = AsyncMock()
    mock_db.fetchrow = AsyncMock()
    return mock_db


@pytest.fixture
def mock_redis():
    """Mock Redis connection for testing."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock()
    mock_redis.set = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.exists = AsyncMock()
    return mock_redis


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB connection for testing."""
    mock_mongo = MagicMock()
    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock()
    mock_collection.find_one = AsyncMock()
    mock_collection.find = MagicMock()
    mock_collection.update_one = AsyncMock()
    mock_collection.delete_one = AsyncMock()
    mock_mongo.__getitem__ = MagicMock(return_value=mock_collection)
    return mock_mongo


@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_monitoring_overhead": 0.1,  # 10%
        "max_response_time_ms": 500,
        "min_throughput_events_per_sec": 1000,
        "max_concurrent_agents": 100,
        "test_duration_seconds": 60
    }


@pytest.fixture
def load_test_config():
    """Configuration for load tests."""
    return {
        "concurrent_users": [1, 5, 10, 25, 50, 100],
        "ramp_up_time": 30,  # seconds
        "test_duration": 300,  # seconds
        "endpoints_to_test": [
            "/api/v1/monitor/start",
            "/api/v1/epistemic/test_agent/current",
            "/api/v1/patterns/test_agent/analyze"
        ]
    }


@pytest.fixture
def accuracy_test_config():
    """Configuration for accuracy tests."""
    return {
        "min_prediction_accuracy": 0.85,
        "min_pattern_detection_accuracy": 0.90,
        "min_causal_inference_confidence": 0.80,
        "test_data_size": 1000,
        "cross_validation_folds": 5
    }


# Test data generators
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_agent_logs(count: int, agent_id: str = "test_agent") -> List[Dict[str, Any]]:
        """Generate synthetic agent logs."""
        logs = []
        base_time = datetime.now() - timedelta(hours=count)
        
        for i in range(count):
            log_time = base_time + timedelta(minutes=i)
            logs.append({
                "timestamp": log_time,
                "agent_id": agent_id,
                "level": np.random.choice(["INFO", "DEBUG", "WARNING", "ERROR"], p=[0.6, 0.3, 0.08, 0.02]),
                "message": f"Agent action {i}: {np.random.choice(['processing', 'analyzing', 'deciding', 'executing'])}",
                "context": {
                    "task_id": f"task_{i // 10}",
                    "confidence": np.random.uniform(0.5, 1.0),
                    "success": np.random.choice([True, False], p=[0.8, 0.2])
                }
            })
        
        return logs
    
    @staticmethod
    def generate_behavioral_sequences(count: int) -> List[ExecutionSequence]:
        """Generate synthetic behavioral sequences."""
        sequences = []
        actions = ["initialize", "load_data", "validate", "process", "analyze", "report", "cleanup"]
        
        for i in range(count):
            steps = []
            start_time = datetime.now() - timedelta(hours=i)
            current_time = start_time
            
            # Generate random sequence of actions
            sequence_length = np.random.randint(3, len(actions))
            selected_actions = np.random.choice(actions, sequence_length, replace=False)
            
            success = True
            for j, action in enumerate(selected_actions):
                step_success = np.random.choice([True, False], p=[0.9, 0.1])
                if not step_success:
                    success = False
                
                steps.append({
                    "action": action,
                    "timestamp": current_time,
                    "success": step_success,
                    "duration": np.random.exponential(30)  # seconds
                })
                current_time += timedelta(seconds=np.random.exponential(60))
            
            sequences.append(ExecutionSequence(
                sequence_id=f"seq_{i:04d}",
                agent_id=f"agent_{i % 10}",
                steps=steps,
                start_time=start_time,
                end_time=current_time,
                success=success,
                error_message="Random test error" if not success else None
            ))
        
        return sequences


@pytest.fixture
def test_data_generator():
    """Test data generator fixture."""
    return TestDataGenerator()


# Performance monitoring utilities
class PerformanceMonitor:
    """Utility class for monitoring test performance."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = datetime.now()
    
    def stop(self):
        """Stop performance monitoring."""
        self.end_time = datetime.now()
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_duration(self) -> float:
        """Get test duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_average_metric(self, name: str) -> float:
        """Get average value of a metric."""
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "duration": self.get_duration(),
            "metrics": {
                name: {
                    "count": len(values),
                    "average": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
                for name, values in self.metrics.items()
            }
        }


@pytest.fixture
def performance_monitor():
    """Performance monitor fixture."""
    return PerformanceMonitor()