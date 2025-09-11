"""
Unit tests for behavioral pattern models.
"""

import pytest
from datetime import datetime, timedelta
from escai_framework.models.behavioral_pattern import (
    ExecutionStep, ExecutionSequence, BehavioralPattern,
    ExecutionStatus, PatternType
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
            duration=1.5,
            status=ExecutionStatus.SUCCESS
        )
        
        assert step.step_id == "step_001"
        assert step.action == "process_data"
        assert step.timestamp == timestamp
        assert step.duration == 1.5
        assert step.status == ExecutionStatus.SUCCESS

    def test_execution_step_validation_valid(self):
        """Test validation of valid ExecutionStep."""
        step = ExecutionStep(
            step_id="step_001",
            action="validate_data",
            duration=0.5,
            success_probability=0.95,
            status=ExecutionStatus.SUCCESS
        )
        
        assert step.validate() is True

    def test_execution_step_validation_invalid(self):
        """Test validation of invalid ExecutionStep."""
        step = ExecutionStep(
            step_id="",  # Invalid empty step_id
            action="validate_data",
            duration=-1.0,  # Invalid negative duration
            success_probability=1.5,  # Invalid probability > 1
            status=ExecutionStatus.SUCCESS
        )
        
        assert step.validate() is False

    def test_execution_step_serialization(self):
        """Test ExecutionStep serialization."""
        timestamp = datetime.utcnow()
        original = ExecutionStep(
            step_id="step_001",
            action="serialize_test",
            timestamp=timestamp,
            duration=2.0,
            success_probability=0.8,
            context={"key": "value"},
            error_message="test error"
        )
        
        data = original.to_dict()
        
        assert data["step_id"] == "step_001"
        assert data["action"] == "serialize_test"
        assert data["duration"] == 2.0
        assert data["success_probability"] == 0.8
        assert data["context"] == {"key": "value"}
        assert data["error_message"] == "test error"


class TestExecutionSequence:
    """Test cases for ExecutionSequence model."""
    
    def test_execution_sequence_creation(self):
        """Test creating a valid ExecutionSequence."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=30)
        
        step1 = ExecutionStep(
            step_id="step_001",
            action="initialize",
            duration=1.0,
            status=ExecutionStatus.SUCCESS
        )
        step2 = ExecutionStep(
            step_id="step_002", 
            action="process",
            duration=2.0,
            status=ExecutionStatus.SUCCESS
        )
        
        sequence = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="agent_001",
            steps=[step1, step2],
            start_time=start_time,
            end_time=end_time,
            success=True
        )
        
        assert sequence.sequence_id == "seq_001"
        assert sequence.agent_id == "agent_001"
        assert len(sequence.steps) == 2
        assert sequence.start_time == start_time
        assert sequence.end_time == end_time
        assert sequence.success is True

    def test_execution_sequence_calculate_metrics(self):
        """Test ExecutionSequence metrics calculation."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=30)
        
        step1 = ExecutionStep(
            step_id="step_001",
            action="step1",
            duration=1.0,
            status=ExecutionStatus.SUCCESS
        )
        step2 = ExecutionStep(
            step_id="step_002",
            action="step2", 
            duration=2.0,
            status=ExecutionStatus.FAILED
        )
        
        sequence = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="agent_001",
            steps=[step1, step2],
            start_time=start_time,
            end_time=end_time,
            success=False
        )
        
        # Test that actions are populated from steps
        assert "step1" in sequence.actions
        assert "step2" in sequence.actions

    def test_execution_sequence_validation(self):
        """Test ExecutionSequence validation."""
        sequence = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="agent_001",
            steps=[],
            success=True
        )
        
        assert sequence.validate() is True

    def test_execution_sequence_serialization(self):
        """Test ExecutionSequence serialization."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=30)
        
        step = ExecutionStep(
            step_id="step_001",
            action="test_action",
            duration=1.0,
            status=ExecutionStatus.SUCCESS
        )
        
        original = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="agent_001",
            steps=[step],
            start_time=start_time,
            end_time=end_time,
            success=True,
            error_message=None
        )
        
        data = original.to_dict()
        
        assert data["sequence_id"] == "seq_001"
        assert data["agent_id"] == "agent_001"
        assert len(data["steps"]) == 1
        assert data["success"] is True


class TestBehavioralPattern:
    """Test cases for BehavioralPattern model."""
    
    def test_behavioral_pattern_creation(self):
        """Test creating a valid BehavioralPattern."""
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            pattern_name="Data Processing Pattern",
            pattern_type=PatternType.SEQUENTIAL,
            description="A pattern for processing data sequentially",
            frequency=10.0,
            success_rate=0.85,
            average_duration=120.5,
            common_triggers=["data_received", "process_request"],
            failure_modes=["timeout", "invalid_data"],
            statistical_significance=0.95
        )
        
        assert pattern.pattern_id == "pattern_001"
        assert pattern.pattern_name == "Data Processing Pattern"
        assert pattern.pattern_type == PatternType.SEQUENTIAL
        assert pattern.frequency == 10.0
        assert pattern.success_rate == 0.85
        assert pattern.average_duration == 120.5

    def test_behavioral_pattern_validation(self):
        """Test BehavioralPattern validation."""
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            pattern_type=PatternType.TEMPORAL,
            frequency=0.5,
            success_rate=0.8,
            average_duration=100.0,
            execution_sequences=[]
        )
        
        assert pattern.validate() is True

    def test_behavioral_pattern_calculate_statistics(self):
        """Test BehavioralPattern statistics calculation."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=60)
        
        step1 = ExecutionStep(
            step_id="step_001",
            action="action1",
            duration=1.0,
            status=ExecutionStatus.SUCCESS
        )
        
        sequence1 = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="agent_001",
            steps=[step1],
            start_time=start_time,
            end_time=end_time,
            success=True
        )
        
        sequence2 = ExecutionSequence(
            sequence_id="seq_002",
            agent_id="agent_001", 
            steps=[step1],
            start_time=start_time,
            end_time=end_time,
            success=False
        )
        
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            execution_sequences=[sequence1, sequence2]
        )
        
        pattern.calculate_statistics()
        
        assert pattern.frequency == 2.0
        assert pattern.success_rate == 0.5  # 1 success out of 2
        assert pattern.average_duration == 60.0  # 60 seconds

    def test_behavioral_pattern_add_sequence(self):
        """Test adding sequences to BehavioralPattern."""
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            pattern_type=PatternType.CAUSAL,
            execution_sequences=[]
        )
        
        step = ExecutionStep(
            step_id="step_001",
            action="test_action",
            duration=1.0,
            status=ExecutionStatus.SUCCESS
        )
        
        sequence = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="agent_001",
            steps=[step],
            success=True
        )
        
        pattern.execution_sequences.append(sequence)
        
        assert len(pattern.execution_sequences) == 1
        assert pattern.execution_sequences[0].sequence_id == "seq_001"

    def test_behavioral_pattern_serialization(self):
        """Test BehavioralPattern serialization."""
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            pattern_name="Test Pattern",
            pattern_type=PatternType.FEEDBACK_LOOP,
            description="Test description",
            frequency=5.0,
            success_rate=0.9,
            average_duration=150.0,
            common_triggers=["trigger1"],
            failure_modes=["mode1"],
            statistical_significance=0.99
        )
        
        data = pattern.to_dict()
        
        assert data["pattern_id"] == "pattern_001"
        assert data["pattern_name"] == "Test Pattern"
        assert data["pattern_type"] == "feedback_loop"
        assert data["frequency"] == 5.0
        assert data["success_rate"] == 0.9

    def test_behavioral_pattern_json_serialization(self):
        """Test BehavioralPattern JSON serialization."""
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            pattern_name="JSON Test Pattern",
            pattern_type=PatternType.SEQUENTIAL,
            description="JSON test description"
        )
        
        json_str = pattern.to_json()
        assert isinstance(json_str, str)
        assert "pattern_001" in json_str
        assert "JSON Test Pattern" in json_str

    def test_behavioral_pattern_with_invalid_sequences(self):
        """Test BehavioralPattern with invalid sequences."""
        invalid_step = ExecutionStep(
            step_id="",  # Invalid empty step_id
            action="invalid_action",
            duration=-1.0,  # Invalid negative duration
            status=ExecutionStatus.SUCCESS
        )
        
        invalid_sequence = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="agent_001",
            steps=[invalid_step],
            success=True
        )
        
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            execution_sequences=[invalid_sequence]
        )
        
        assert pattern.validate() is False