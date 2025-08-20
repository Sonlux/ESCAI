"""
Unit tests for BehavioralPattern data model.
"""

import pytest
from datetime import datetime, timedelta
import json

from escai_framework.models.behavioral_pattern import (
    BehavioralPattern, ExecutionSequence, ExecutionStep,
    PatternType, ExecutionStatus
)


class TestExecutionStep:
    """Test cases for ExecutionStep model."""
    
    def test_execution_step_creation(self):
        """Test creating a valid ExecutionStep."""
        timestamp = datetime.utcnow()
        step = ExecutionStep(
            step_id="step_001",
            action="process_data",
            timestamp=timestamp,
            duration_ms=1500,
            status=ExecutionStatus.SUCCESS,
            inputs={"data": "input_data"},
            outputs={"result": "processed_data"},
            error_message=None
        )
        
        assert step.step_id == "step_001"
        assert step.action == "process_data"
        assert step.timestamp == timestamp
        assert step.duration_ms == 1500
        assert step.status == ExecutionStatus.SUCCESS
        assert step.inputs == {"data": "input_data"}
        assert step.outputs == {"result": "processed_data"}
        assert step.error_message is None
    
    def test_execution_step_validation_valid(self):
        """Test validation of valid ExecutionStep."""
        step = ExecutionStep(
            step_id="valid_step",
            action="valid_action",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        assert step.validate() is True
    
    def test_execution_step_validation_invalid(self):
        """Test validation with invalid data."""
        step = ExecutionStep(
            step_id="",  # Invalid empty ID
            action="valid_action",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        assert step.validate() is False
        
        step.step_id = "valid_id"
        step.duration_ms = -100  # Invalid negative duration
        assert step.validate() is False
    
    def test_execution_step_serialization(self):
        """Test ExecutionStep serialization."""
        timestamp = datetime.utcnow()
        original = ExecutionStep(
            step_id="serialize_test",
            action="test_action",
            timestamp=timestamp,
            duration_ms=2000,
            status=ExecutionStatus.FAILURE,
            error_message="Test error"
        )
        
        data = original.to_dict()
        restored = ExecutionStep.from_dict(data)
        
        assert restored.step_id == original.step_id
        assert restored.action == original.action
        assert restored.timestamp == original.timestamp
        assert restored.duration_ms == original.duration_ms
        assert restored.status == original.status
        assert restored.error_message == original.error_message


class TestExecutionSequence:
    """Test cases for ExecutionSequence model."""
    
    def test_execution_sequence_creation(self):
        """Test creating a valid ExecutionSequence."""
        step1 = ExecutionStep(
            step_id="step1",
            action="action1",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        
        step2 = ExecutionStep(
            step_id="step2",
            action="action2",
            timestamp=datetime.utcnow(),
            duration_ms=1500,
            status=ExecutionStatus.SUCCESS
        )
        
        sequence = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="agent_001",
            task_description="Test task",
            steps=[step1, step2]
        )
        
        assert sequence.sequence_id == "seq_001"
        assert sequence.agent_id == "agent_001"
        assert sequence.task_description == "Test task"
        assert len(sequence.steps) == 2
    
    def test_execution_sequence_calculate_metrics(self):
        """Test calculation of sequence metrics."""
        step1 = ExecutionStep(
            step_id="step1",
            action="action1",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        
        step2 = ExecutionStep(
            step_id="step2",
            action="action2",
            timestamp=datetime.utcnow() + timedelta(seconds=1),
            duration_ms=2000,
            status=ExecutionStatus.FAILURE
        )
        
        sequence = ExecutionSequence(
            sequence_id="metrics_test",
            agent_id="agent_001",
            task_description="Metrics test",
            steps=[step1, step2]
        )
        
        sequence.calculate_metrics()
        
        assert sequence.total_duration_ms == 3000
        assert sequence.success_rate == 0.5  # 1 success out of 2 steps
        assert sequence.end_time is not None
    
    def test_execution_sequence_validation(self):
        """Test ExecutionSequence validation."""
        sequence = ExecutionSequence(
            sequence_id="valid_seq",
            agent_id="valid_agent",
            task_description="Valid task"
        )
        assert sequence.validate() is True
        
        sequence.sequence_id = ""
        assert sequence.validate() is False
        
        sequence.sequence_id = "valid_seq"
        sequence.success_rate = 1.5  # Invalid rate
        assert sequence.validate() is False
    
    def test_execution_sequence_serialization(self):
        """Test ExecutionSequence serialization."""
        step = ExecutionStep(
            step_id="test_step",
            action="test_action",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        
        original = ExecutionSequence(
            sequence_id="serialize_seq",
            agent_id="serialize_agent",
            task_description="Serialization test",
            steps=[step]
        )
        
        data = original.to_dict()
        restored = ExecutionSequence.from_dict(data)
        
        assert restored.sequence_id == original.sequence_id
        assert restored.agent_id == original.agent_id
        assert restored.task_description == original.task_description
        assert len(restored.steps) == 1
        assert restored.steps[0].step_id == step.step_id


class TestBehavioralPattern:
    """Test cases for BehavioralPattern model."""
    
    def test_behavioral_pattern_creation(self):
        """Test creating a valid BehavioralPattern."""
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            pattern_name="Test Pattern",
            pattern_type=PatternType.SEQUENTIAL,
            description="A test behavioral pattern",
            frequency=10,
            success_rate=0.8,
            average_duration_ms=5000.0,
            common_triggers=["trigger1", "trigger2"],
            failure_modes=["timeout", "error"],
            statistical_significance=0.95
        )
        
        assert pattern.pattern_id == "pattern_001"
        assert pattern.pattern_name == "Test Pattern"
        assert pattern.pattern_type == PatternType.SEQUENTIAL
        assert pattern.description == "A test behavioral pattern"
        assert pattern.frequency == 10
        assert pattern.success_rate == 0.8
        assert pattern.average_duration_ms == 5000.0
        assert pattern.common_triggers == ["trigger1", "trigger2"]
        assert pattern.failure_modes == ["timeout", "error"]
        assert pattern.statistical_significance == 0.95
    
    def test_behavioral_pattern_validation(self):
        """Test BehavioralPattern validation."""
        pattern = BehavioralPattern(
            pattern_id="valid_pattern",
            pattern_name="Valid Pattern",
            pattern_type=PatternType.CYCLICAL,
            description="Valid description"
        )
        assert pattern.validate() is True
        
        pattern.pattern_id = ""
        assert pattern.validate() is False
        
        pattern.pattern_id = "valid_pattern"
        pattern.success_rate = 1.5
        assert pattern.validate() is False
    
    def test_behavioral_pattern_calculate_statistics(self):
        """Test calculation of pattern statistics."""
        # Create execution sequences
        step1 = ExecutionStep(
            step_id="step1",
            action="action1",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        
        step2 = ExecutionStep(
            step_id="step2",
            action="action2",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        
        seq1 = ExecutionSequence(
            sequence_id="seq1",
            agent_id="agent1",
            task_description="Task 1",
            steps=[step1, step2]
        )
        seq1.calculate_metrics()  # This will set success_rate to 1.0
        
        seq2 = ExecutionSequence(
            sequence_id="seq2",
            agent_id="agent2",
            task_description="Task 2",
            steps=[step1]  # Only one successful step
        )
        seq2.calculate_metrics()  # This will set success_rate to 1.0
        
        pattern = BehavioralPattern(
            pattern_id="stats_test",
            pattern_name="Statistics Test",
            pattern_type=PatternType.SEQUENTIAL,
            description="Test pattern for statistics",
            execution_sequences=[seq1, seq2]
        )
        
        pattern.calculate_statistics()
        
        assert pattern.frequency == 2
        assert pattern.success_rate == 1.0  # Both sequences have >80% success rate
        assert pattern.average_duration_ms == 1500.0  # (2000 + 1000) / 2
        assert pattern.last_observed is not None
    
    def test_behavioral_pattern_add_sequence(self):
        """Test adding execution sequences to pattern."""
        pattern = BehavioralPattern(
            pattern_id="add_test",
            pattern_name="Add Test",
            pattern_type=PatternType.CONDITIONAL,
            description="Test adding sequences"
        )
        
        step = ExecutionStep(
            step_id="step1",
            action="action1",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        
        sequence = ExecutionSequence(
            sequence_id="new_seq",
            agent_id="agent1",
            task_description="New sequence",
            steps=[step]
        )
        
        initial_count = len(pattern.execution_sequences)
        pattern.add_sequence(sequence)
        
        assert len(pattern.execution_sequences) == initial_count + 1
        assert pattern.frequency == 1
    
    def test_behavioral_pattern_serialization(self):
        """Test BehavioralPattern serialization."""
        original = BehavioralPattern(
            pattern_id="serialize_pattern",
            pattern_name="Serialization Pattern",
            pattern_type=PatternType.HIERARCHICAL,
            description="Pattern for serialization testing",
            frequency=5,
            success_rate=0.9,
            statistical_significance=0.99
        )
        
        # Test to_dict
        data = original.to_dict()
        assert data["pattern_id"] == "serialize_pattern"
        assert data["pattern_type"] == "hierarchical"
        
        # Test from_dict
        restored = BehavioralPattern.from_dict(data)
        assert restored.pattern_id == original.pattern_id
        assert restored.pattern_name == original.pattern_name
        assert restored.pattern_type == original.pattern_type
        assert restored.description == original.description
        assert restored.frequency == original.frequency
        assert restored.success_rate == original.success_rate
        assert restored.statistical_significance == original.statistical_significance
    
    def test_behavioral_pattern_json_serialization(self):
        """Test BehavioralPattern JSON serialization."""
        pattern = BehavioralPattern(
            pattern_id="json_test",
            pattern_name="JSON Test",
            pattern_type=PatternType.PARALLEL,
            description="JSON serialization test"
        )
        
        json_str = pattern.to_json()
        assert isinstance(json_str, str)
        
        restored = BehavioralPattern.from_json(json_str)
        assert restored.pattern_id == pattern.pattern_id
        assert restored.pattern_name == pattern.pattern_name
        assert restored.pattern_type == pattern.pattern_type
        assert restored.description == pattern.description
    
    def test_behavioral_pattern_with_invalid_sequences(self):
        """Test BehavioralPattern validation with invalid sequences."""
        invalid_step = ExecutionStep(
            step_id="",  # Invalid empty ID
            action="action",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        
        invalid_sequence = ExecutionSequence(
            sequence_id="seq1",
            agent_id="agent1",
            task_description="Task",
            steps=[invalid_step]
        )
        
        pattern = BehavioralPattern(
            pattern_id="invalid_test",
            pattern_name="Invalid Test",
            pattern_type=PatternType.SEQUENTIAL,
            description="Test with invalid sequence",
            execution_sequences=[invalid_sequence]
        )
        
        assert pattern.validate() is False


if __name__ == "__main__":
    pytest.main([__file__])