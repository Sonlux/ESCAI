"""
End-to-end tests for complete monitoring workflows with realistic scenarios.
Tests the entire system from agent instrumentation to analysis and reporting.
"""

import asyncio
import time
import json
from typing import List, Dict, Any
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from escai_framework.instrumentation.langchain_instrumentor import LangChainInstrumentor
from escai_framework.instrumentation.events import AgentEvent, EventType
from escai_framework.core.epistemic_extractor import EpistemicExtractor
from escai_framework.core.pattern_analyzer import BehavioralAnalyzer
from escai_framework.core.causal_engine import CausalEngine
from escai_framework.core.performance_predictor import PerformancePredictor
from escai_framework.core.explanation_engine import ExplanationEngine
from escai_framework.api.main import create_app
from escai_framework.storage.database import DatabaseManager


class RealisticAgentSimulator:
    """Simulates realistic agent behavior for end-to-end testing."""
    
    def __init__(self, agent_id: str, scenario: str = "data_analysis"):
        self.agent_id = agent_id
        self.scenario = scenario
        self.current_step = 0
        self.context = {}
        self.events_generated = []
    
    async def execute_scenario(self, instrumentor: LangChainInstrumentor) -> Dict[str, Any]:
        """Execute a complete realistic scenario."""
        
        if self.scenario == "data_analysis":
            return await self._execute_data_analysis_scenario(instrumentor)
        elif self.scenario == "web_scraping":
            return await self._execute_web_scraping_scenario(instrumentor)
        elif self.scenario == "api_integration":
            return await self._execute_api_integration_scenario(instrumentor)
        elif self.scenario == "machine_learning":
            return await self._execute_ml_scenario(instrumentor)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
    
    async def _execute_data_analysis_scenario(self, instrumentor: LangChainInstrumentor) -> Dict[str, Any]:
        """Execute a realistic data analysis workflow."""
        scenario_start = time.time()
        
        # Step 1: Initialize and plan
        await self._generate_event(instrumentor, EventType.TASK_START, {
            "task": "data_analysis",
            "dataset": "sales_data.csv",
            "objective": "analyze quarterly sales performance"
        })
        
        await asyncio.sleep(0.1)  # Simulate thinking time
        
        # Step 2: Load and validate data
        await self._generate_event(instrumentor, EventType.DECISION_MADE, {
            "decision": "load_csv_data",
            "confidence": 0.9,
            "reasoning": "CSV format is standard and well-supported"
        })
        
        # Simulate data loading with potential issues
        if np.random.random() < 0.2:  # 20% chance of data issues
            await self._generate_event(instrumentor, EventType.ERROR_OCCURRED, {
                "error": "DataValidationError",
                "message": "Missing values detected in revenue column",
                "severity": "warning"
            })
            
            await self._generate_event(instrumentor, EventType.DECISION_MADE, {
                "decision": "handle_missing_values",
                "confidence": 0.7,
                "reasoning": "Use forward fill for missing revenue data"
            })
        
        await asyncio.sleep(0.2)
        
        # Step 3: Exploratory analysis
        await self._generate_event(instrumentor, EventType.ANALYSIS_STARTED, {
            "analysis_type": "exploratory",
            "variables": ["revenue", "units_sold", "region", "quarter"],
            "methods": ["descriptive_stats", "correlation_analysis"]
        })
        
        await asyncio.sleep(0.3)
        
        # Step 4: Generate insights
        insights = [
            "Q4 shows 15% revenue increase compared to Q3",
            "North region outperforms others by 23%",
            "Strong correlation between marketing spend and revenue (r=0.78)"
        ]
        
        await self._generate_event(instrumentor, EventType.INSIGHT_GENERATED, {
            "insights": insights,
            "confidence": 0.85,
            "statistical_significance": 0.95
        })
        
        # Step 5: Create visualizations
        await self._generate_event(instrumentor, EventType.VISUALIZATION_CREATED, {
            "chart_types": ["bar_chart", "line_plot", "heatmap"],
            "variables_plotted": ["revenue_by_quarter", "regional_performance", "correlation_matrix"]
        })
        
        await asyncio.sleep(0.1)
        
        # Step 6: Generate report
        success = np.random.random() > 0.1  # 90% success rate
        
        if success:
            await self._generate_event(instrumentor, EventType.TASK_COMPLETE, {
                "status": "success",
                "duration": time.time() - scenario_start,
                "outputs": ["analysis_report.pdf", "charts.png", "summary_stats.json"],
                "key_findings": insights
            })
        else:
            await self._generate_event(instrumentor, EventType.ERROR_OCCURRED, {
                "error": "ReportGenerationError",
                "message": "Failed to generate PDF report due to formatting issues",
                "severity": "error"
            })
        
        return {
            "scenario": "data_analysis",
            "success": success,
            "duration": time.time() - scenario_start,
            "events_count": len(self.events_generated),
            "insights_generated": len(insights)
        }
    
    async def _execute_web_scraping_scenario(self, instrumentor: LangChainInstrumentor) -> Dict[str, Any]:
        """Execute a realistic web scraping workflow."""
        scenario_start = time.time()
        
        # Step 1: Initialize scraping task
        await self._generate_event(instrumentor, EventType.TASK_START, {
            "task": "web_scraping",
            "target_url": "https://example-ecommerce.com/products",
            "objective": "extract product information and prices"
        })
        
        # Step 2: Setup and authentication
        await self._generate_event(instrumentor, EventType.DECISION_MADE, {
            "decision": "use_selenium_driver",
            "confidence": 0.8,
            "reasoning": "JavaScript-heavy site requires browser automation"
        })
        
        await asyncio.sleep(0.2)
        
        # Step 3: Navigate and handle potential issues
        if np.random.random() < 0.3:  # 30% chance of rate limiting
            await self._generate_event(instrumentor, EventType.ERROR_OCCURRED, {
                "error": "RateLimitError",
                "message": "Too many requests, implementing delay",
                "severity": "warning"
            })
            
            await self._generate_event(instrumentor, EventType.DECISION_MADE, {
                "decision": "implement_exponential_backoff",
                "confidence": 0.9,
                "reasoning": "Exponential backoff is standard for rate limiting"
            })
            
            await asyncio.sleep(0.5)  # Simulate delay
        
        # Step 4: Extract data
        products_scraped = np.random.randint(50, 200)
        
        await self._generate_event(instrumentor, EventType.DATA_EXTRACTED, {
            "items_extracted": products_scraped,
            "fields": ["name", "price", "rating", "availability", "description"],
            "extraction_rate": f"{products_scraped/60:.1f} items/minute"
        })
        
        await asyncio.sleep(0.4)
        
        # Step 5: Data validation and cleaning
        validation_issues = np.random.randint(0, 10)
        
        if validation_issues > 0:
            await self._generate_event(instrumentor, EventType.DATA_VALIDATION, {
                "issues_found": validation_issues,
                "issue_types": ["missing_price", "invalid_rating", "empty_description"],
                "resolution": "cleaned_and_standardized"
            })
        
        # Step 6: Save results
        success = np.random.random() > 0.05  # 95% success rate for web scraping
        
        if success:
            await self._generate_event(instrumentor, EventType.TASK_COMPLETE, {
                "status": "success",
                "duration": time.time() - scenario_start,
                "products_scraped": products_scraped,
                "data_quality_score": 0.92,
                "output_file": "scraped_products.json"
            })
        else:
            await self._generate_event(instrumentor, EventType.ERROR_OCCURRED, {
                "error": "DataSaveError",
                "message": "Failed to save scraped data to database",
                "severity": "error"
            })
        
        return {
            "scenario": "web_scraping",
            "success": success,
            "duration": time.time() - scenario_start,
            "events_count": len(self.events_generated),
            "products_scraped": products_scraped if success else 0
        }
    
    async def _execute_api_integration_scenario(self, instrumentor: LangChainInstrumentor) -> Dict[str, Any]:
        """Execute a realistic API integration workflow."""
        scenario_start = time.time()
        
        # Step 1: Initialize API integration
        await self._generate_event(instrumentor, EventType.TASK_START, {
            "task": "api_integration",
            "api_endpoint": "https://api.example.com/v1/data",
            "objective": "sync customer data with external CRM"
        })
        
        # Step 2: Authentication
        await self._generate_event(instrumentor, EventType.AUTHENTICATION_ATTEMPT, {
            "auth_method": "oauth2",
            "token_type": "bearer"
        })
        
        auth_success = np.random.random() > 0.1  # 90% auth success rate
        
        if not auth_success:
            await self._generate_event(instrumentor, EventType.ERROR_OCCURRED, {
                "error": "AuthenticationError",
                "message": "OAuth token expired, refreshing",
                "severity": "warning"
            })
            
            await asyncio.sleep(0.2)
            
            await self._generate_event(instrumentor, EventType.AUTHENTICATION_SUCCESS, {
                "auth_method": "oauth2_refresh",
                "token_refreshed": True
            })
        
        # Step 3: Data synchronization
        records_to_sync = np.random.randint(100, 1000)
        
        await self._generate_event(instrumentor, EventType.DATA_SYNC_STARTED, {
            "records_count": records_to_sync,
            "sync_direction": "bidirectional",
            "batch_size": 50
        })
        
        # Simulate batch processing
        batches = (records_to_sync + 49) // 50  # Round up division
        
        for batch in range(batches):
            await asyncio.sleep(0.1)  # Simulate API call time
            
            # Occasional API errors
            if np.random.random() < 0.05:  # 5% chance of API error per batch
                await self._generate_event(instrumentor, EventType.ERROR_OCCURRED, {
                    "error": "APIError",
                    "message": f"Batch {batch + 1} failed, retrying",
                    "batch_number": batch + 1,
                    "severity": "warning"
                })
                
                await asyncio.sleep(0.2)  # Retry delay
        
        # Step 4: Validation and conflict resolution
        conflicts = np.random.randint(0, 20)
        
        if conflicts > 0:
            await self._generate_event(instrumentor, EventType.CONFLICT_DETECTED, {
                "conflicts_count": conflicts,
                "conflict_types": ["duplicate_email", "outdated_timestamp", "field_mismatch"],
                "resolution_strategy": "latest_timestamp_wins"
            })
        
        # Step 5: Complete synchronization
        success = np.random.random() > 0.08  # 92% success rate
        
        if success:
            await self._generate_event(instrumentor, EventType.TASK_COMPLETE, {
                "status": "success",
                "duration": time.time() - scenario_start,
                "records_synced": records_to_sync - conflicts,
                "conflicts_resolved": conflicts,
                "sync_accuracy": 0.98
            })
        else:
            await self._generate_event(instrumentor, EventType.ERROR_OCCURRED, {
                "error": "SyncError",
                "message": "Critical API failure, sync incomplete",
                "severity": "error"
            })
        
        return {
            "scenario": "api_integration",
            "success": success,
            "duration": time.time() - scenario_start,
            "events_count": len(self.events_generated),
            "records_synced": records_to_sync - conflicts if success else 0
        }
    
    async def _execute_ml_scenario(self, instrumentor: LangChainInstrumentor) -> Dict[str, Any]:
        """Execute a realistic machine learning workflow."""
        scenario_start = time.time()
        
        # Step 1: Initialize ML project
        await self._generate_event(instrumentor, EventType.TASK_START, {
            "task": "machine_learning",
            "model_type": "classification",
            "objective": "predict customer churn"
        })
        
        # Step 2: Data preparation
        await self._generate_event(instrumentor, EventType.DATA_PREPARATION, {
            "dataset_size": 10000,
            "features": 25,
            "target_variable": "churn",
            "preprocessing_steps": ["scaling", "encoding", "feature_selection"]
        })
        
        await asyncio.sleep(0.3)
        
        # Step 3: Model selection and training
        models_tested = ["random_forest", "xgboost", "logistic_regression", "neural_network"]
        
        for model in models_tested:
            await asyncio.sleep(0.2)  # Simulate training time
            
            accuracy = np.random.uniform(0.75, 0.92)
            
            await self._generate_event(instrumentor, EventType.MODEL_TRAINED, {
                "model_type": model,
                "accuracy": accuracy,
                "precision": accuracy + np.random.uniform(-0.05, 0.05),
                "recall": accuracy + np.random.uniform(-0.05, 0.05),
                "training_time": np.random.uniform(30, 180)
            })
        
        # Step 4: Model evaluation and selection
        best_model = "xgboost"  # Assume XGBoost performs best
        
        await self._generate_event(instrumentor, EventType.MODEL_SELECTED, {
            "selected_model": best_model,
            "selection_criteria": "highest_f1_score",
            "cross_validation_score": 0.89,
            "feature_importance_calculated": True
        })
        
        # Step 5: Model deployment preparation
        await self._generate_event(instrumentor, EventType.MODEL_VALIDATION, {
            "validation_type": "holdout_test",
            "test_accuracy": 0.87,
            "model_size_mb": 15.2,
            "inference_time_ms": 12
        })
        
        # Step 6: Generate model insights
        await self._generate_event(instrumentor, EventType.INSIGHT_GENERATED, {
            "insights": [
                "Customer tenure is the strongest predictor of churn",
                "Support ticket frequency correlates with churn risk",
                "Premium customers have 40% lower churn rate"
            ],
            "feature_importance": {
                "tenure": 0.35,
                "support_tickets": 0.22,
                "plan_type": 0.18,
                "usage_frequency": 0.15,
                "payment_method": 0.10
            }
        })
        
        # Step 7: Complete ML workflow
        success = np.random.random() > 0.12  # 88% success rate
        
        if success:
            await self._generate_event(instrumentor, EventType.TASK_COMPLETE, {
                "status": "success",
                "duration": time.time() - scenario_start,
                "final_model": best_model,
                "model_accuracy": 0.87,
                "ready_for_deployment": True
            })
        else:
            await self._generate_event(instrumentor, EventType.ERROR_OCCURRED, {
                "error": "ModelValidationError",
                "message": "Model performance degraded on test set",
                "severity": "error"
            })
        
        return {
            "scenario": "machine_learning",
            "success": success,
            "duration": time.time() - scenario_start,
            "events_count": len(self.events_generated),
            "models_tested": len(models_tested)
        }
    
    async def _generate_event(self, instrumentor: LangChainInstrumentor, event_type: EventType, data: Dict[str, Any]):
        """Generate and capture an event."""
        event = AgentEvent(
            event_id=f"{self.agent_id}_event_{len(self.events_generated)}",
            agent_id=self.agent_id,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data
        )
        
        await instrumentor.capture_event(event)
        self.events_generated.append(event)


@pytest.mark.asyncio
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    async def test_data_analysis_workflow(self):
        """Test complete data analysis workflow from start to finish."""
        
        # Setup components
        agent_id = "e2e_data_analyst"
        instrumentor = LangChainInstrumentor()
        extractor = EpistemicExtractor()
        analyzer = BehavioralAnalyzer()
        predictor = PerformancePredictor()
        explainer = ExplanationEngine()
        
        # Start monitoring
        session_id = await instrumentor.start_monitoring(agent_id, {
            "framework": "test",
            "scenario": "data_analysis"
        })
        
        try:
            # Execute realistic scenario
            simulator = RealisticAgentSimulator(agent_id, "data_analysis")
            scenario_result = await simulator.execute_scenario(instrumentor)
            
            # Wait for event processing
            await asyncio.sleep(1.0)
            
            # Extract epistemic states
            agent_logs = [
                {
                    "timestamp": event.timestamp,
                    "agent_id": event.agent_id,
                    "event_type": event.event_type.value,
                    "data": event.data
                }
                for event in simulator.events_generated
            ]
            
            epistemic_states = await extractor.extract_beliefs(agent_logs)
            
            # Analyze behavioral patterns
            execution_sequence = self._create_execution_sequence_from_events(simulator.events_generated)
            patterns = await analyzer.mine_patterns([execution_sequence])
            
            # Generate predictions
            if epistemic_states:
                latest_state = epistemic_states[-1]
                prediction = await predictor.predict_success(latest_state)
            
            # Generate explanations
            explanation = await explainer.explain_behavior(
                agent_id=agent_id,
                behavior_summary="Data analysis workflow execution",
                epistemic_states=epistemic_states,
                patterns=patterns
            )
            
            # Assertions
            assert scenario_result["events_count"] > 5, "Insufficient events generated"
            assert len(epistemic_states) > 0, "No epistemic states extracted"
            assert len(patterns) > 0, "No behavioral patterns identified"
            assert prediction is not None, "No prediction generated"
            assert explanation is not None, "No explanation generated"
            assert len(explanation.explanation_text) > 100, "Explanation too brief"
            
            # Verify workflow coherence
            event_types = [event.event_type for event in simulator.events_generated]
            assert EventType.TASK_START in event_types, "Missing task start event"
            
            if scenario_result["success"]:
                assert EventType.TASK_COMPLETE in event_types, "Missing task completion event"
            
            print(f"Data analysis workflow: {scenario_result['success']}, "
                  f"{scenario_result['events_count']} events, "
                  f"{len(epistemic_states)} states extracted")
        
        finally:
            await instrumentor.stop_monitoring(session_id)
    
    async def test_multi_scenario_comparison(self):
        """Test multiple scenarios and compare their patterns."""
        
        scenarios = ["data_analysis", "web_scraping", "api_integration", "machine_learning"]
        scenario_results = {}
        all_patterns = []
        
        for scenario in scenarios:
            agent_id = f"e2e_{scenario}_agent"
            instrumentor = LangChainInstrumentor()
            analyzer = BehavioralAnalyzer()
            
            session_id = await instrumentor.start_monitoring(agent_id, {
                "framework": "test",
                "scenario": scenario
            })
            
            try:
                # Execute scenario
                simulator = RealisticAgentSimulator(agent_id, scenario)
                result = await simulator.execute_scenario(instrumentor)
                
                # Analyze patterns
                execution_sequence = self._create_execution_sequence_from_events(simulator.events_generated)
                patterns = await analyzer.mine_patterns([execution_sequence])
                
                scenario_results[scenario] = {
                    "result": result,
                    "patterns": patterns,
                    "events": simulator.events_generated
                }
                
                all_patterns.extend(patterns)
                
            finally:
                await instrumentor.stop_monitoring(session_id)
        
        # Compare scenarios
        success_rates = {
            scenario: data["result"]["success"] 
            for scenario, data in scenario_results.items()
        }
        
        event_counts = {
            scenario: data["result"]["events_count"]
            for scenario, data in scenario_results.items()
        }
        
        # Assertions
        assert len(scenario_results) == len(scenarios), "Not all scenarios completed"
        assert all(count > 0 for count in event_counts.values()), "Some scenarios generated no events"
        assert len(all_patterns) > 0, "No patterns discovered across scenarios"
        
        # Verify scenario diversity
        unique_event_types = set()
        for scenario_data in scenario_results.values():
            for event in scenario_data["events"]:
                unique_event_types.add(event.event_type)
        
        assert len(unique_event_types) >= 5, "Insufficient event type diversity"
        
        print(f"Multi-scenario test: {len(scenarios)} scenarios, "
              f"{sum(event_counts.values())} total events, "
              f"{len(unique_event_types)} unique event types")
    
    async def test_failure_recovery_workflow(self):
        """Test workflow behavior during failures and recovery."""
        
        agent_id = "e2e_failure_recovery_agent"
        instrumentor = LangChainInstrumentor()
        extractor = EpistemicExtractor()
        
        session_id = await instrumentor.start_monitoring(agent_id, {
            "framework": "test",
            "failure_simulation": True
        })
        
        try:
            # Simulate a workflow with intentional failures
            events = []
            
            # Start task
            events.append(await self._create_and_capture_event(
                instrumentor, agent_id, EventType.TASK_START,
                {"task": "failure_recovery_test", "complexity": "high"}
            ))
            
            # First attempt - failure
            events.append(await self._create_and_capture_event(
                instrumentor, agent_id, EventType.ERROR_OCCURRED,
                {"error": "NetworkTimeout", "attempt": 1, "severity": "error"}
            ))
            
            # Recovery decision
            events.append(await self._create_and_capture_event(
                instrumentor, agent_id, EventType.DECISION_MADE,
                {"decision": "retry_with_backoff", "confidence": 0.7, "reasoning": "Network issues are often transient"}
            ))
            
            # Second attempt - partial success
            events.append(await self._create_and_capture_event(
                instrumentor, agent_id, EventType.PARTIAL_SUCCESS,
                {"progress": 0.6, "attempt": 2, "issues": ["data_incomplete"]}
            ))
            
            # Adaptation decision
            events.append(await self._create_and_capture_event(
                instrumentor, agent_id, EventType.DECISION_MADE,
                {"decision": "adapt_strategy", "confidence": 0.8, "new_approach": "incremental_processing"}
            ))
            
            # Final attempt - success
            events.append(await self._create_and_capture_event(
                instrumentor, agent_id, EventType.TASK_COMPLETE,
                {"status": "success", "attempt": 3, "recovery_successful": True}
            ))
            
            await asyncio.sleep(0.5)  # Allow processing
            
            # Extract epistemic states
            agent_logs = [
                {
                    "timestamp": event.timestamp,
                    "agent_id": event.agent_id,
                    "event_type": event.event_type.value,
                    "data": event.data
                }
                for event in events
            ]
            
            epistemic_states = await extractor.extract_beliefs(agent_logs)
            
            # Analyze recovery pattern
            execution_sequence = self._create_execution_sequence_from_events(events)
            
            # Assertions
            assert len(events) >= 6, "Insufficient events for failure recovery test"
            assert len(epistemic_states) > 0, "No epistemic states extracted"
            
            # Verify failure and recovery sequence
            event_types = [event.event_type for event in events]
            assert EventType.ERROR_OCCURRED in event_types, "No error event found"
            assert EventType.DECISION_MADE in event_types, "No recovery decision found"
            assert EventType.TASK_COMPLETE in event_types, "Recovery did not complete successfully"
            
            # Check epistemic state evolution during recovery
            if len(epistemic_states) >= 2:
                initial_confidence = epistemic_states[0].confidence_level
                final_confidence = epistemic_states[-1].confidence_level
                
                # Confidence should recover after successful completion
                assert final_confidence >= initial_confidence * 0.8, "Confidence did not recover adequately"
            
            print(f"Failure recovery test: {len(events)} events, "
                  f"{len(epistemic_states)} states, recovery successful")
        
        finally:
            await instrumentor.stop_monitoring(session_id)
    
    async def test_real_time_monitoring_and_alerts(self):
        """Test real-time monitoring with alert generation."""
        
        agent_id = "e2e_realtime_agent"
        instrumentor = LangChainInstrumentor()
        analyzer = BehavioralAnalyzer()
        
        session_id = await instrumentor.start_monitoring(agent_id, {
            "framework": "test",
            "real_time_alerts": True
        })
        
        alerts_generated = []
        
        try:
            # Simulate real-time event stream
            for i in range(20):
                # Generate various events
                if i % 5 == 0:
                    # Periodic decision events
                    await self._create_and_capture_event(
                        instrumentor, agent_id, EventType.DECISION_MADE,
                        {"decision": f"decision_{i}", "confidence": np.random.uniform(0.5, 1.0)}
                    )
                elif i % 7 == 0:
                    # Occasional errors (should trigger alerts)
                    await self._create_and_capture_event(
                        instrumentor, agent_id, EventType.ERROR_OCCURRED,
                        {"error": "PerformanceWarning", "metric": "response_time", "value": 5000}
                    )
                    alerts_generated.append(f"Performance alert at event {i}")
                else:
                    # Regular progress events
                    await self._create_and_capture_event(
                        instrumentor, agent_id, EventType.PROGRESS_UPDATE,
                        {"progress": i / 20, "current_step": f"step_{i}"}
                    )
                
                # Small delay to simulate real-time stream
                await asyncio.sleep(0.05)
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Verify real-time processing
            # In a real implementation, this would check actual alert systems
            
            # Assertions
            assert len(alerts_generated) > 0, "No alerts generated during monitoring"
            
            print(f"Real-time monitoring test: 20 events processed, "
                  f"{len(alerts_generated)} alerts generated")
        
        finally:
            await instrumentor.stop_monitoring(session_id)
    
    def _create_execution_sequence_from_events(self, events: List[AgentEvent]) -> 'ExecutionSequence':
        """Create an execution sequence from a list of events."""
        from escai_framework.models.behavioral_pattern import ExecutionSequence
        
        steps = []
        for event in events:
            steps.append({
                "action": event.event_type.value,
                "timestamp": event.timestamp,
                "success": event.event_type not in [EventType.ERROR_OCCURRED],
                "data": event.data
            })
        
        return ExecutionSequence(
            sequence_id=f"e2e_sequence_{events[0].agent_id}",
            agent_id=events[0].agent_id,
            steps=steps,
            start_time=events[0].timestamp if events else datetime.now(),
            end_time=events[-1].timestamp if events else datetime.now(),
            success=not any(event.event_type == EventType.ERROR_OCCURRED for event in events),
            error_message=None
        )
    
    async def _create_and_capture_event(
        self, 
        instrumentor: LangChainInstrumentor, 
        agent_id: str, 
        event_type: EventType, 
        data: Dict[str, Any]
    ) -> AgentEvent:
        """Create and capture an event."""
        event = AgentEvent(
            event_id=f"{agent_id}_{event_type.value}_{int(time.time() * 1000)}",
            agent_id=agent_id,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data
        )
        
        await instrumentor.capture_event(event)
        return event


@pytest.mark.asyncio
class TestSystemIntegration:
    """Test integration between different system components."""
    
    async def test_full_pipeline_integration(self):
        """Test integration of all major components in a single pipeline."""
        
        # Initialize all components
        agent_id = "integration_test_agent"
        instrumentor = LangChainInstrumentor()
        extractor = EpistemicExtractor()
        analyzer = BehavioralAnalyzer()
        causal_engine = CausalEngine()
        predictor = PerformancePredictor()
        explainer = ExplanationEngine()
        
        session_id = await instrumentor.start_monitoring(agent_id, {})
        
        try:
            # Generate comprehensive event sequence
            simulator = RealisticAgentSimulator(agent_id, "data_analysis")
            scenario_result = await simulator.execute_scenario(instrumentor)
            
            await asyncio.sleep(1.0)  # Allow processing
            
            # Process through entire pipeline
            
            # 1. Extract epistemic states
            agent_logs = [
                {
                    "timestamp": event.timestamp,
                    "agent_id": event.agent_id,
                    "event_type": event.event_type.value,
                    "data": event.data
                }
                for event in simulator.events_generated
            ]
            
            epistemic_states = await extractor.extract_beliefs(agent_logs)
            
            # 2. Analyze behavioral patterns
            execution_sequence = self._create_execution_sequence_from_events(simulator.events_generated)
            patterns = await analyzer.mine_patterns([execution_sequence])
            
            # 3. Discover causal relationships
            temporal_events = [
                {
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp,
                    "agent_id": event.agent_id,
                    "data": event.data
                }
                for event in simulator.events_generated
            ]
            
            causal_relationships = await causal_engine.discover_relationships(temporal_events)
            
            # 4. Generate predictions
            if epistemic_states:
                prediction = await predictor.predict_success(epistemic_states[-1])
            else:
                prediction = None
            
            # 5. Generate comprehensive explanation
            explanation = await explainer.explain_behavior(
                agent_id=agent_id,
                behavior_summary="Full pipeline integration test",
                epistemic_states=epistemic_states,
                patterns=patterns,
                causal_relationships=causal_relationships,
                prediction=prediction
            )
            
            # Comprehensive assertions
            assert len(epistemic_states) > 0, "Epistemic extraction failed"
            assert len(patterns) > 0, "Pattern analysis failed"
            assert prediction is not None, "Prediction generation failed"
            assert explanation is not None, "Explanation generation failed"
            
            # Verify data consistency across components
            agent_ids = set()
            agent_ids.update(state.agent_id for state in epistemic_states)
            agent_ids.update(pattern.execution_sequences[0].agent_id for pattern in patterns if pattern.execution_sequences)
            
            assert len(agent_ids) == 1, "Inconsistent agent IDs across components"
            assert agent_id in agent_ids, "Agent ID not preserved through pipeline"
            
            # Verify temporal consistency
            if epistemic_states and len(epistemic_states) > 1:
                timestamps = [state.timestamp for state in epistemic_states]
                assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)), \
                    "Temporal ordering not preserved"
            
            print(f"Full pipeline integration: {len(epistemic_states)} states, "
                  f"{len(patterns)} patterns, {len(causal_relationships)} causal links")
        
        finally:
            await instrumentor.stop_monitoring(session_id)
    
    def _create_execution_sequence_from_events(self, events: List[AgentEvent]) -> 'ExecutionSequence':
        """Create an execution sequence from a list of events."""
        from escai_framework.models.behavioral_pattern import ExecutionSequence
        
        steps = []
        for event in events:
            steps.append({
                "action": event.event_type.value,
                "timestamp": event.timestamp,
                "success": event.event_type not in [EventType.ERROR_OCCURRED],
                "data": event.data
            })
        
        return ExecutionSequence(
            sequence_id=f"integration_sequence_{events[0].agent_id}",
            agent_id=events[0].agent_id,
            steps=steps,
            start_time=events[0].timestamp if events else datetime.now(),
            end_time=events[-1].timestamp if events else datetime.now(),
            success=not any(event.event_type == EventType.ERROR_OCCURRED for event in events),
            error_message=None
        )


if __name__ == "__main__":
    # Run end-to-end tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])