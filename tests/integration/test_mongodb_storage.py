"""
Integration tests for MongoDB storage functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import os

from escai_framework.storage.database import DatabaseManager
from escai_framework.storage.mongo_models import (
    RawLogDocument, ProcessedEventDocument, ExplanationDocument,
    ConfigurationDocument, AnalyticsResultDocument
)


@pytest.fixture
async def db_manager():
    """Create a test database manager."""
    manager = DatabaseManager()
    
    # Use test MongoDB database
    test_mongo_url = os.getenv('TEST_MONGO_URL', 'mongodb://localhost:27017')
    test_mongo_db = os.getenv('TEST_MONGO_DB', 'escai_test')
    
    manager.initialize(
        database_url='sqlite:///:memory:',  # In-memory SQLite for tests
        async_database_url='sqlite+aiosqlite:///:memory:',
        mongo_url=test_mongo_url,
        mongo_db_name=test_mongo_db
    )
    
    if manager.mongo_available:
        await manager.create_tables()
        yield manager
        
        # Cleanup: drop test collections
        if manager.mongo_manager:
            for repo_name in manager.mongo_manager.repository_names:
                repo = await manager.mongo_manager.get_repository(repo_name)
                await repo.collection.drop()
    else:
        pytest.skip("MongoDB not available for testing")
    
    await manager.close()


@pytest.mark.asyncio
class TestRawLogRepository:
    """Test raw log repository functionality."""
    
    async def test_insert_and_find_raw_log(self, db_manager):
        """Test inserting and finding raw logs."""
        repo = db_manager.mongo_manager.raw_logs
        
        # Create test log
        log = RawLogDocument(
            agent_id="test_agent_1",
            session_id="test_session_1",
            framework="langchain",
            log_level="INFO",
            message="Test log message",
            metadata={"key": "value"},
            timestamp=datetime.utcnow()
        )
        
        # Insert log
        log_id = await repo.insert_one(log)
        assert log_id is not None
        
        # Find by ID
        found_log = await repo.find_by_id(log_id)
        assert found_log is not None
        assert found_log.agent_id == "test_agent_1"
        assert found_log.message == "Test log message"
    
    async def test_find_logs_by_agent(self, db_manager):
        """Test finding logs by agent."""
        repo = db_manager.mongo_manager.raw_logs
        
        # Insert multiple logs
        logs = [
            RawLogDocument(
                agent_id="agent_1",
                session_id="session_1",
                framework="langchain",
                log_level="INFO",
                message=f"Log message {i}",
                timestamp=datetime.utcnow() - timedelta(minutes=i)
            )
            for i in range(5)
        ]
        
        await repo.insert_many(logs)
        
        # Find logs by agent
        found_logs = await repo.find_by_agent("agent_1", limit=10)
        assert len(found_logs) == 5
        
        # Should be sorted by timestamp descending
        timestamps = [log.timestamp for log in found_logs]
        assert timestamps == sorted(timestamps, reverse=True)
    
    async def test_search_logs(self, db_manager):
        """Test text search functionality."""
        repo = db_manager.mongo_manager.raw_logs
        
        # Insert logs with searchable content
        logs = [
            RawLogDocument(
                agent_id="agent_1",
                session_id="session_1",
                framework="langchain",
                log_level="INFO",
                message="Processing user request for data analysis",
                timestamp=datetime.utcnow()
            ),
            RawLogDocument(
                agent_id="agent_1",
                session_id="session_1",
                framework="langchain",
                log_level="ERROR",
                message="Failed to connect to database",
                timestamp=datetime.utcnow()
            )
        ]
        
        await repo.insert_many(logs)
        
        # Search for logs
        search_results = await repo.search_logs("data analysis")
        assert len(search_results) >= 1
        assert any("data analysis" in log.message for log in search_results)


@pytest.mark.asyncio
class TestProcessedEventRepository:
    """Test processed event repository functionality."""
    
    async def test_insert_and_find_event(self, db_manager):
        """Test inserting and finding processed events."""
        repo = db_manager.mongo_manager.processed_events
        
        # Create test event
        event = ProcessedEventDocument(
            agent_id="test_agent_1",
            session_id="test_session_1",
            event_type="decision_made",
            event_data={"decision": "use_tool", "confidence": 0.85},
            timestamp=datetime.utcnow()
        )
        
        # Insert event
        event_id = await repo.insert_one(event)
        assert event_id is not None
        
        # Find by ID
        found_event = await repo.find_by_id(event_id)
        assert found_event is not None
        assert found_event.event_type == "decision_made"
        assert found_event.event_data["decision"] == "use_tool"
    
    async def test_find_decision_events(self, db_manager):
        """Test finding decision events."""
        repo = db_manager.mongo_manager.processed_events
        
        # Insert mixed events
        events = [
            ProcessedEventDocument(
                agent_id="agent_1",
                session_id="session_1",
                event_type="decision_made",
                event_data={"decision": f"decision_{i}"},
                timestamp=datetime.utcnow() - timedelta(minutes=i)
            )
            for i in range(3)
        ] + [
            ProcessedEventDocument(
                agent_id="agent_1",
                session_id="session_1",
                event_type="tool_used",
                event_data={"tool": "calculator"},
                timestamp=datetime.utcnow()
            )
        ]
        
        await repo.insert_many(events)
        
        # Find only decision events
        decision_events = await repo.find_decision_events("agent_1")
        assert len(decision_events) == 3
        assert all(event.event_type == "decision_made" for event in decision_events)


@pytest.mark.asyncio
class TestExplanationRepository:
    """Test explanation repository functionality."""
    
    async def test_insert_and_find_explanation(self, db_manager):
        """Test inserting and finding explanations."""
        repo = db_manager.mongo_manager.explanations
        
        # Create test explanation
        explanation = ExplanationDocument(
            agent_id="test_agent_1",
            session_id="test_session_1",
            explanation_type="behavior_summary",
            title="Agent Behavior Summary",
            content="The agent successfully completed the task by using appropriate tools.",
            confidence_score=0.9,
            supporting_evidence=[{"type": "log", "content": "Tool usage successful"}]
        )
        
        # Insert explanation
        explanation_id = await repo.insert_one(explanation)
        assert explanation_id is not None
        
        # Find by ID
        found_explanation = await repo.find_by_id(explanation_id)
        assert found_explanation is not None
        assert found_explanation.explanation_type == "behavior_summary"
        assert found_explanation.confidence_score == 0.9
    
    async def test_find_high_confidence_explanations(self, db_manager):
        """Test finding high confidence explanations."""
        repo = db_manager.mongo_manager.explanations
        
        # Insert explanations with different confidence scores
        explanations = [
            ExplanationDocument(
                agent_id="agent_1",
                session_id="session_1",
                explanation_type="behavior_summary",
                title=f"Explanation {i}",
                content=f"Content {i}",
                confidence_score=0.5 + (i * 0.1)
            )
            for i in range(6)  # Scores: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]
        
        await repo.insert_many(explanations)
        
        # Find high confidence explanations (>= 0.8)
        high_confidence = await repo.find_high_confidence_explanations(min_confidence=0.8)
        assert len(high_confidence) == 3  # 0.8, 0.9, 1.0
        assert all(exp.confidence_score >= 0.8 for exp in high_confidence)


@pytest.mark.asyncio
class TestConfigurationRepository:
    """Test configuration repository functionality."""
    
    async def test_create_and_update_configuration(self, db_manager):
        """Test creating and updating configurations."""
        repo = db_manager.mongo_manager.configurations
        
        # Create initial configuration
        config_id = await repo.create_or_update_configuration(
            config_type="system",
            config_name="monitoring_settings",
            config_data={"max_agents": 100, "log_level": "INFO"}
        )
        assert config_id is not None
        
        # Find the configuration
        config = await repo.find_by_name("system", "monitoring_settings")
        assert config is not None
        assert config.version == 1
        assert config.config_data["max_agents"] == 100
        
        # Update the configuration
        new_config_id = await repo.create_or_update_configuration(
            config_type="system",
            config_name="monitoring_settings",
            config_data={"max_agents": 200, "log_level": "DEBUG"}
        )
        
        # Should create new version
        updated_config = await repo.find_by_name("system", "monitoring_settings")
        assert updated_config.version == 2
        assert updated_config.config_data["max_agents"] == 200
        
        # Old version should be deactivated
        old_config = await repo.find_by_id(config_id)
        assert old_config.is_active is False
    
    async def test_configuration_history(self, db_manager):
        """Test configuration version history."""
        repo = db_manager.mongo_manager.configurations
        
        # Create multiple versions
        for i in range(3):
            await repo.create_or_update_configuration(
                config_type="user_preferences",
                config_name="dashboard_layout",
                config_data={"layout": f"version_{i+1}"},
                user_id="user_123"
            )
        
        # Get history
        history = await repo.get_configuration_history(
            "user_preferences", "dashboard_layout", user_id="user_123"
        )
        
        assert len(history) == 3
        assert history[0].version == 3  # Latest first
        assert history[1].version == 2
        assert history[2].version == 1


@pytest.mark.asyncio
class TestAnalyticsResultRepository:
    """Test analytics result repository functionality."""
    
    async def test_insert_and_find_analytics_result(self, db_manager):
        """Test inserting and finding analytics results."""
        repo = db_manager.mongo_manager.analytics_results
        
        # Create test result
        result = AnalyticsResultDocument(
            analysis_type="pattern_mining",
            agent_id="agent_1",
            model_name="PrefixSpan",
            model_version="1.0",
            input_data_hash="abc123",
            results={"patterns": [{"pattern": "A->B->C", "support": 0.8}]},
            metrics={"precision": 0.85, "recall": 0.78},
            execution_time_ms=1500.0
        )
        
        # Insert result
        result_id = await repo.insert_one(result)
        assert result_id is not None
        
        # Find by ID
        found_result = await repo.find_by_id(result_id)
        assert found_result is not None
        assert found_result.analysis_type == "pattern_mining"
        assert found_result.execution_time_ms == 1500.0
    
    async def test_find_by_analysis_type(self, db_manager):
        """Test finding results by analysis type."""
        repo = db_manager.mongo_manager.analytics_results
        
        # Insert results of different types
        results = [
            AnalyticsResultDocument(
                analysis_type="pattern_mining",
                model_name="PrefixSpan",
                model_version="1.0",
                input_data_hash=f"hash_{i}",
                results={"patterns": []},
                execution_time_ms=1000.0 + i * 100
            )
            for i in range(3)
        ] + [
            AnalyticsResultDocument(
                analysis_type="anomaly_detection",
                model_name="IsolationForest",
                model_version="1.0",
                input_data_hash="hash_anomaly",
                results={"anomalies": []},
                execution_time_ms=2000.0
            )
        ]
        
        await repo.insert_many(results)
        
        # Find pattern mining results
        pattern_results = await repo.find_by_analysis_type("pattern_mining")
        assert len(pattern_results) == 3
        assert all(r.analysis_type == "pattern_mining" for r in pattern_results)
        
        # Find anomaly detection results
        anomaly_results = await repo.find_by_analysis_type("anomaly_detection")
        assert len(anomaly_results) == 1
        assert anomaly_results[0].analysis_type == "anomaly_detection"


@pytest.mark.asyncio
class TestMongoManager:
    """Test MongoDB manager functionality."""
    
    async def test_health_check(self, db_manager):
        """Test MongoDB manager health check."""
        health = await db_manager.mongo_manager.health_check()
        
        assert 'mongodb' in health
        assert health['mongodb']['status'] == 'healthy'
        assert 'repositories' in health
        
        # All repositories should be healthy
        for repo_name in db_manager.mongo_manager.repository_names:
            assert repo_name in health['repositories']
            assert health['repositories'][repo_name]['status'] == 'healthy'
    
    async def test_database_info(self, db_manager):
        """Test getting database information."""
        info = await db_manager.mongo_manager.get_database_info()
        
        assert 'database' in info
        assert 'collections' in info
        assert info['database']['name'] is not None
        
        # Should have info for all repositories
        for repo_name in db_manager.mongo_manager.repository_names:
            assert repo_name in info['collections']
    
    async def test_cleanup_old_data(self, db_manager):
        """Test cleanup functionality."""
        # Insert some old data
        raw_logs_repo = db_manager.mongo_manager.raw_logs
        old_log = RawLogDocument(
            agent_id="agent_1",
            session_id="session_1",
            framework="langchain",
            log_level="INFO",
            message="Old log",
            timestamp=datetime.utcnow() - timedelta(days=35)  # Older than default retention
        )
        await raw_logs_repo.insert_one(old_log)
        
        # Run cleanup
        cleanup_results = await db_manager.mongo_manager.cleanup_old_data(
            raw_logs_days=30
        )
        
        assert 'raw_logs' in cleanup_results
        assert cleanup_results['raw_logs'] >= 1  # Should have cleaned up at least the old log


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])