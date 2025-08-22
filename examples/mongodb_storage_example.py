"""
Example demonstrating MongoDB storage functionality in ESCAI Framework.
"""

import asyncio
from datetime import datetime, timedelta
import logging

from escai_framework.storage.database import DatabaseManager
from escai_framework.storage.mongo_models import (
    RawLogDocument, ProcessedEventDocument, ExplanationDocument,
    ConfigurationDocument, AnalyticsResultDocument
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate MongoDB storage functionality."""
    
    # Initialize database manager
    db_manager = DatabaseManager()
    db_manager.initialize(
        database_url='sqlite:///:memory:',  # In-memory SQLite for demo
        async_database_url='sqlite+aiosqlite:///:memory:',
        mongo_url='mongodb://localhost:27017',
        mongo_db_name='escai_demo'
    )
    
    if not db_manager.mongo_available:
        logger.error("MongoDB not available. Please ensure MongoDB is running.")
        return
    
    try:
        # Initialize MongoDB collections and indexes
        await db_manager.create_tables()
        
        logger.info("=== MongoDB Storage Demo ===")
        
        # Demo 1: Raw Logs
        await demo_raw_logs(db_manager)
        
        # Demo 2: Processed Events
        await demo_processed_events(db_manager)
        
        # Demo 3: Explanations
        await demo_explanations(db_manager)
        
        # Demo 4: Configurations
        await demo_configurations(db_manager)
        
        # Demo 5: Analytics Results
        await demo_analytics_results(db_manager)
        
        # Demo 6: Manager Operations
        await demo_manager_operations(db_manager)
        
    finally:
        await db_manager.close()


async def demo_raw_logs(db_manager):
    """Demonstrate raw log operations."""
    logger.info("\n--- Raw Logs Demo ---")
    
    repo = db_manager.mongo_manager.raw_logs
    
    # Insert sample logs
    logs = [
        RawLogDocument(
            agent_id="demo_agent_1",
            session_id="demo_session_1",
            framework="langchain",
            log_level="INFO",
            message="Agent started processing user request",
            metadata={"user_id": "user123", "request_type": "analysis"},
            timestamp=datetime.utcnow() - timedelta(minutes=10)
        ),
        RawLogDocument(
            agent_id="demo_agent_1",
            session_id="demo_session_1",
            framework="langchain",
            log_level="DEBUG",
            message="Loading data from database",
            metadata={"table": "user_data", "rows": 1000},
            timestamp=datetime.utcnow() - timedelta(minutes=9)
        ),
        RawLogDocument(
            agent_id="demo_agent_1",
            session_id="demo_session_1",
            framework="langchain",
            log_level="ERROR",
            message="Failed to connect to external API",
            metadata={"api": "weather_service", "error_code": 503},
            timestamp=datetime.utcnow() - timedelta(minutes=5)
        )
    ]
    
    log_ids = await repo.insert_many(logs)
    logger.info(f"Inserted {len(log_ids)} raw logs")
    
    # Find logs by agent
    agent_logs = await repo.find_by_agent("demo_agent_1", limit=10)
    logger.info(f"Found {len(agent_logs)} logs for demo_agent_1")
    
    # Find error logs
    error_logs = await repo.find_errors(agent_id="demo_agent_1", hours_back=1)
    logger.info(f"Found {len(error_logs)} error logs")
    
    # Search logs
    search_results = await repo.search_logs("database", agent_id="demo_agent_1")
    logger.info(f"Found {len(search_results)} logs containing 'database'")
    
    # Get statistics
    stats = await repo.get_log_statistics(agent_id="demo_agent_1", hours_back=1)
    logger.info(f"Log statistics: {stats['total_logs']} total logs")


async def demo_processed_events(db_manager):
    """Demonstrate processed event operations."""
    logger.info("\n--- Processed Events Demo ---")
    
    repo = db_manager.mongo_manager.processed_events
    
    # Insert sample events
    events = [
        ProcessedEventDocument(
            agent_id="demo_agent_1",
            session_id="demo_session_1",
            event_type="agent_start",
            event_data={"start_time": datetime.utcnow().isoformat()},
            timestamp=datetime.utcnow() - timedelta(minutes=10)
        ),
        ProcessedEventDocument(
            agent_id="demo_agent_1",
            session_id="demo_session_1",
            event_type="decision_made",
            event_data={
                "decision": "use_calculator_tool",
                "confidence": 0.85,
                "reasoning": "Mathematical calculation required"
            },
            timestamp=datetime.utcnow() - timedelta(minutes=8)
        ),
        ProcessedEventDocument(
            agent_id="demo_agent_1",
            session_id="demo_session_1",
            event_type="tool_used",
            event_data={
                "tool_name": "calculator",
                "input": "2 + 2",
                "output": "4",
                "execution_time_ms": 150
            },
            timestamp=datetime.utcnow() - timedelta(minutes=7)
        )
    ]
    
    event_ids = await repo.insert_many(events)
    logger.info(f"Inserted {len(event_ids)} processed events")
    
    # Find decision events
    decision_events = await repo.find_decision_events("demo_agent_1")
    logger.info(f"Found {len(decision_events)} decision events")
    
    # Find tool usage events
    tool_events = await repo.find_tool_usage_events("demo_agent_1")
    logger.info(f"Found {len(tool_events)} tool usage events")
    
    # Get event statistics
    stats = await repo.get_event_statistics(agent_id="demo_agent_1", hours_back=1)
    logger.info(f"Event statistics: {stats['total_events']} total events")


async def demo_explanations(db_manager):
    """Demonstrate explanation operations."""
    logger.info("\n--- Explanations Demo ---")
    
    repo = db_manager.mongo_manager.explanations
    
    # Insert sample explanations
    explanations = [
        ExplanationDocument(
            agent_id="demo_agent_1",
            session_id="demo_session_1",
            explanation_type="behavior_summary",
            title="Agent Task Completion Summary",
            content="The agent successfully completed the mathematical calculation task by using the calculator tool. The decision to use the calculator was made with high confidence (0.85) based on the requirement for precise mathematical computation.",
            confidence_score=0.92,
            supporting_evidence=[
                {"type": "decision_event", "confidence": 0.85},
                {"type": "tool_success", "execution_time": 150}
            ]
        ),
        ExplanationDocument(
            agent_id="demo_agent_1",
            session_id="demo_session_1",
            explanation_type="decision_pathway",
            title="Calculator Tool Selection",
            content="The agent chose to use the calculator tool because: 1) The task required mathematical computation, 2) The calculator tool was available and appropriate, 3) Previous similar tasks showed high success rates with this tool.",
            confidence_score=0.88,
            supporting_evidence=[
                {"type": "task_analysis", "math_required": True},
                {"type": "tool_availability", "calculator": True}
            ]
        )
    ]
    
    explanation_ids = await repo.insert_many(explanations)
    logger.info(f"Inserted {len(explanation_ids)} explanations")
    
    # Find high confidence explanations
    high_confidence = await repo.find_high_confidence_explanations(min_confidence=0.9)
    logger.info(f"Found {len(high_confidence)} high confidence explanations")
    
    # Search explanations
    search_results = await repo.search_explanations("calculator tool")
    logger.info(f"Found {len(search_results)} explanations about calculator tool")
    
    # Get explanation statistics
    stats = await repo.get_explanation_statistics(agent_id="demo_agent_1")
    logger.info(f"Explanation statistics: {stats['total_explanations']} total explanations")


async def demo_configurations(db_manager):
    """Demonstrate configuration operations."""
    logger.info("\n--- Configurations Demo ---")
    
    repo = db_manager.mongo_manager.configurations
    
    # Create system configuration
    system_config_id = await repo.create_or_update_configuration(
        config_type="system",
        config_name="monitoring_settings",
        config_data={
            "max_concurrent_agents": 100,
            "log_retention_days": 30,
            "enable_real_time_alerts": True,
            "alert_thresholds": {
                "error_rate": 0.05,
                "response_time_ms": 5000
            }
        }
    )
    logger.info(f"Created system configuration: {system_config_id}")
    
    # Create user configuration
    user_config_id = await repo.create_or_update_configuration(
        config_type="user_preferences",
        config_name="dashboard_layout",
        config_data={
            "theme": "dark",
            "widgets": ["agent_status", "recent_logs", "performance_metrics"],
            "refresh_interval_seconds": 30
        },
        user_id="demo_user_123"
    )
    logger.info(f"Created user configuration: {user_config_id}")
    
    # Update system configuration (creates new version)
    updated_config_id = await repo.create_or_update_configuration(
        config_type="system",
        config_name="monitoring_settings",
        config_data={
            "max_concurrent_agents": 150,  # Updated value
            "log_retention_days": 30,
            "enable_real_time_alerts": True,
            "alert_thresholds": {
                "error_rate": 0.03,  # Updated value
                "response_time_ms": 5000
            }
        }
    )
    logger.info(f"Updated system configuration: {updated_config_id}")
    
    # Get configuration history
    history = await repo.get_configuration_history("system", "monitoring_settings")
    logger.info(f"Configuration has {len(history)} versions")
    
    # Get current configuration
    current_config = await repo.find_by_name("system", "monitoring_settings")
    logger.info(f"Current configuration version: {current_config.version}")


async def demo_analytics_results(db_manager):
    """Demonstrate analytics result operations."""
    logger.info("\n--- Analytics Results Demo ---")
    
    repo = db_manager.mongo_manager.analytics_results
    
    # Insert sample analytics results
    results = [
        AnalyticsResultDocument(
            analysis_type="pattern_mining",
            agent_id="demo_agent_1",
            model_name="PrefixSpan",
            model_version="1.0",
            input_data_hash="abc123def456",
            results={
                "patterns": [
                    {"pattern": "start -> decision -> tool_use", "support": 0.85},
                    {"pattern": "decision -> tool_use -> success", "support": 0.78}
                ],
                "total_patterns": 2
            },
            metrics={
                "precision": 0.92,
                "recall": 0.87,
                "f1_score": 0.89
            },
            execution_time_ms=2500.0
        ),
        AnalyticsResultDocument(
            analysis_type="anomaly_detection",
            agent_id="demo_agent_1",
            model_name="IsolationForest",
            model_version="0.24.2",
            input_data_hash="xyz789abc123",
            results={
                "anomalies": [
                    {"timestamp": "2024-01-15T10:30:00Z", "score": -0.15},
                    {"timestamp": "2024-01-15T11:45:00Z", "score": -0.22}
                ],
                "total_anomalies": 2
            },
            metrics={
                "contamination": 0.1,
                "accuracy": 0.94
            },
            execution_time_ms=1800.0
        )
    ]
    
    result_ids = await repo.insert_many(results)
    logger.info(f"Inserted {len(result_ids)} analytics results")
    
    # Find pattern mining results
    pattern_results = await repo.find_pattern_mining_results(agent_id="demo_agent_1")
    logger.info(f"Found {len(pattern_results)} pattern mining results")
    
    # Find anomaly detection results
    anomaly_results = await repo.find_anomaly_detection_results(agent_id="demo_agent_1")
    logger.info(f"Found {len(anomaly_results)} anomaly detection results")
    
    # Get model performance metrics
    model_metrics = await repo.get_model_performance_metrics("PrefixSpan")
    logger.info(f"Model performance metrics: {len(model_metrics)} versions analyzed")


async def demo_manager_operations(db_manager):
    """Demonstrate MongoDB manager operations."""
    logger.info("\n--- Manager Operations Demo ---")
    
    manager = db_manager.mongo_manager
    
    # Health check
    health = await manager.health_check()
    logger.info(f"MongoDB health: {health['mongodb']['status']}")
    logger.info(f"Repository health: {len([r for r in health['repositories'].values() if r['status'] == 'healthy'])} healthy")
    
    # Database info
    db_info = await manager.get_database_info()
    logger.info(f"Database: {db_info['database']['name']}")
    logger.info(f"Collections: {len(db_info['collections'])}")
    
    # Cleanup old data (demo with very short retention)
    cleanup_results = await manager.cleanup_old_data(
        raw_logs_days=0,  # Clean all for demo
        processed_events_days=0,
        explanations_days=0,
        analytics_results_days=0
    )
    logger.info(f"Cleanup results: {cleanup_results}")


if __name__ == "__main__":
    asyncio.run(main())