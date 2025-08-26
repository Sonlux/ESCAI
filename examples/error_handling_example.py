"""
Example demonstrating comprehensive error handling and resilience in the ESCAI framework.

This example shows how to use the various error handling components together
to create a robust and resilient monitoring system.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from escai_framework.utils import (
    # Error tracking
    get_error_tracker, get_logger, monitor_errors,
    
    # Retry mechanisms
    retry_async, RetryConfig, BackoffStrategy,
    
    # Circuit breakers
    get_circuit_breaker, CircuitBreakerConfig,
    get_monitoring_circuit_breaker,
    
    # Fallback mechanisms
    execute_with_fallback,
    
    # Load shedding
    get_degradation_manager, Priority,
    
    # Exceptions
    ProcessingError, NetworkError, StorageError
)


# Configure logging
logging.basicConfig(level=logging.INFO)


class ResilientMonitoringSystem:
    """
    Example of a resilient monitoring system using all error handling components.
    """
    
    def __init__(self):
        self.logger = get_logger("resilient_monitoring")
        self.error_tracker = get_error_tracker()
        self.degradation_manager = get_degradation_manager()
        
        # Set up circuit breakers
        self.db_breaker = get_circuit_breaker("database", CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0
        ))
        
        self.api_breaker = get_circuit_breaker("external_api", CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0
        ))
        
        self.monitoring_breaker = get_monitoring_circuit_breaker()
        
        # Register features with priorities
        self._register_features()
        
        # Set up error alerts
        self._setup_error_alerts()
    
    def _register_features(self):
        """Register system features with their priorities."""
        self.degradation_manager.register_feature("core_monitoring", Priority.CRITICAL)
        self.degradation_manager.register_feature("pattern_analysis", Priority.HIGH)
        self.degradation_manager.register_feature("causal_analysis", Priority.MEDIUM)
        self.degradation_manager.register_feature("advanced_analytics", Priority.LOW)
        self.degradation_manager.register_feature("reporting", Priority.OPTIONAL)
    
    def _setup_error_alerts(self):
        """Set up error alerting."""
        def alert_handler(alert_data):
            self.logger.critical(
                f"ALERT: {alert_data['type']}",
                context=alert_data['details']
            )
        
        self.error_tracker.add_alert_callback(alert_handler)
    
    async def start(self):
        """Start the resilient monitoring system."""
        await self.degradation_manager.start()
        self.logger.info("Resilient monitoring system started")
    
    async def stop(self):
        """Stop the resilient monitoring system."""
        await self.degradation_manager.stop()
        self.logger.info("Resilient monitoring system stopped")
    
    @monitor_errors(component="database")
    @retry_async(
        max_attempts=3,
        base_delay=1.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER
    )
    async def store_monitoring_data(self, data: Dict[str, Any]) -> bool:
        """Store monitoring data with retry and circuit breaker protection."""
        
        async def database_operation():
            # Simulate database operation
            if not hasattr(self, '_db_failure_count'):
                self._db_failure_count = 0
            
            # Simulate intermittent failures
            import random
            if random.random() < 0.3:
                self._db_failure_count += 1
                if self._db_failure_count > 2:
                    raise StorageError("Database connection lost", "DB_CONNECTION_ERROR")
                else:
                    raise StorageError("Temporary database error", "DB_TEMP_ERROR")
            
            # Reset failure count on success
            self._db_failure_count = 0
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            return True
        
        return await self.db_breaker.call_async(database_operation)
    
    @monitor_errors(component="external_api")
    async def fetch_external_data(self, endpoint: str) -> Dict[str, Any]:
        """Fetch data from external API with resilience."""
        
        @retry_async(max_attempts=2, base_delay=0.5)
        async def api_call():
            # Simulate API call
            import random
            if random.random() < 0.4:
                raise NetworkError("API temporarily unavailable", "API_UNAVAILABLE")
            
            await asyncio.sleep(0.2)
            return {"data": f"response_from_{endpoint}", "timestamp": time.time()}
        
        return await self.api_breaker.call_async(api_call)
    
    async def process_epistemic_data(self, agent_logs: list) -> Dict[str, Any]:
        """Process epistemic data with fallback mechanisms."""
        
        async def primary_processing(logs):
            # Simulate NLP model processing
            import random
            if random.random() < 0.5:
                raise ProcessingError("NLP model failed to load")
            
            # Simulate processing
            await asyncio.sleep(0.3)
            return {
                "beliefs": ["extracted belief 1", "extracted belief 2"],
                "goals": ["goal 1", "goal 2"],
                "confidence": 0.85,
                "method": "advanced_nlp"
            }
        
        result = await execute_with_fallback(
            primary_processing,
            agent_logs,
            cache_key=f"epistemic_{hash(str(agent_logs))}"
        )
        
        if result.success:
            self.logger.info(
                f"Epistemic processing completed using {result.strategy_used.value}",
                context={"confidence": result.confidence}
            )
            return result.result
        else:
            self.logger.error(
                "Epistemic processing failed completely",
                context={"error": result.error_message}
            )
            raise ProcessingError("All epistemic processing methods failed")
    
    async def analyze_patterns(self, behavioral_data: list) -> Dict[str, Any]:
        """Analyze behavioral patterns with graceful degradation."""
        
        async def advanced_pattern_analysis():
            # Simulate resource-intensive analysis
            import random
            if random.random() < 0.6:
                raise ProcessingError("Pattern analysis overloaded")
            
            await asyncio.sleep(0.5)
            return {
                "patterns": ["pattern_1", "pattern_2", "pattern_3"],
                "anomalies": ["anomaly_1"],
                "confidence": 0.9,
                "method": "advanced"
            }
        
        async def simple_pattern_analysis():
            # Simple fallback analysis
            await asyncio.sleep(0.1)
            return {
                "patterns": ["basic_pattern"],
                "anomalies": [],
                "confidence": 0.6,
                "method": "fallback"
            }
        
        return await self.degradation_manager.execute_feature(
            "pattern_analysis",
            advanced_pattern_analysis,
            simple_pattern_analysis
        )
    
    async def generate_report(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monitoring report with load shedding."""
        
        async def detailed_report():
            # Simulate report generation
            await asyncio.sleep(0.4)
            return {
                "report_type": "detailed",
                "summary": "Comprehensive analysis report",
                "data": analysis_data,
                "generated_at": time.time()
            }
        
        async def summary_report():
            # Quick summary report
            await asyncio.sleep(0.1)
            return {
                "report_type": "summary",
                "summary": "Basic status report",
                "status": "operational",
                "generated_at": time.time()
            }
        
        return await self.degradation_manager.execute_feature(
            "reporting",
            detailed_report,
            summary_report
        )
    
    async def monitor_agent(self, agent_id: str, agent_logs: list) -> Dict[str, Any]:
        """Complete agent monitoring workflow with full resilience."""
        
        self.logger.info(f"Starting monitoring for agent {agent_id}")
        
        results = {}
        
        try:
            # 1. Store raw monitoring data (critical priority)
            await self.degradation_manager.execute_feature(
                "core_monitoring",
                lambda: self.store_monitoring_data({
                    "agent_id": agent_id,
                    "logs": agent_logs,
                    "timestamp": time.time()
                })
            )
            results["data_stored"] = True
            
            # 2. Process epistemic data (high priority)
            epistemic_data = await self.process_epistemic_data(agent_logs)
            results["epistemic_analysis"] = epistemic_data
            
            # 3. Analyze behavioral patterns (medium priority)
            try:
                pattern_data = await self.analyze_patterns(agent_logs)
                results["pattern_analysis"] = pattern_data
            except Exception as e:
                self.logger.warning(f"Pattern analysis failed: {e}")
                results["pattern_analysis"] = {"error": str(e)}
            
            # 4. Fetch external context (low priority)
            try:
                external_data = await self.fetch_external_data(f"agent/{agent_id}/context")
                results["external_context"] = external_data
            except Exception as e:
                self.logger.warning(f"External data fetch failed: {e}")
                results["external_context"] = {"error": str(e)}
            
            # 5. Generate report (optional priority)
            try:
                report = await self.generate_report(results)
                results["report"] = report
            except Exception as e:
                self.logger.warning(f"Report generation failed: {e}")
                results["report"] = {"error": str(e)}
            
            self.logger.info(f"Monitoring completed for agent {agent_id}")
            return results
            
        except Exception as e:
            self.logger.error(
                f"Critical monitoring failure for agent {agent_id}",
                exception=e,
                function_name="monitor_agent"
            )
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "error_metrics": self.error_tracker.get_metrics(),
            "degradation_status": self.degradation_manager.get_status(),
            "circuit_breakers": {
                "database": self.db_breaker.get_status(),
                "external_api": self.api_breaker.get_status(),
                "monitoring": self.monitoring_breaker.get_status()
            },
            "error_patterns": self.error_tracker.get_error_patterns()
        }


async def main():
    """Demonstrate the resilient monitoring system."""
    
    print("🚀 Starting ESCAI Resilient Monitoring System Demo")
    print("=" * 60)
    
    # Create and start the system
    system = ResilientMonitoringSystem()
    await system.start()
    
    try:
        # Simulate monitoring multiple agents
        agents = ["agent_001", "agent_002", "agent_003"]
        
        for i, agent_id in enumerate(agents):
            print(f"\n📊 Monitoring {agent_id}...")
            
            # Simulate agent logs
            agent_logs = [
                f"Agent {agent_id} started task execution",
                f"I believe the current approach will work",
                f"My goal is to complete the task efficiently",
                f"Processing step {i+1} completed",
                f"Confidence level: high"
            ]
            
            try:
                results = await system.monitor_agent(agent_id, agent_logs)
                
                print(f"✅ Monitoring successful for {agent_id}")
                print(f"   - Data stored: {results.get('data_stored', False)}")
                print(f"   - Epistemic analysis: {'✅' if 'epistemic_analysis' in results else '❌'}")
                print(f"   - Pattern analysis: {'✅' if 'pattern_analysis' in results and 'error' not in results['pattern_analysis'] else '⚠️'}")
                print(f"   - External context: {'✅' if 'external_context' in results and 'error' not in results['external_context'] else '⚠️'}")
                print(f"   - Report generated: {'✅' if 'report' in results and 'error' not in results['report'] else '⚠️'}")
                
            except Exception as e:
                print(f"❌ Monitoring failed for {agent_id}: {e}")
            
            # Small delay between agents
            await asyncio.sleep(0.5)
        
        # Show system status
        print(f"\n📈 System Status Report")
        print("-" * 30)
        
        status = system.get_system_status()
        
        print(f"Total Errors: {status['error_metrics']['total_errors']}")
        print(f"Error Rate (last hour): {status['error_metrics']['error_rate_last_hour']}")
        print(f"Degraded Features: {len(status['degradation_status']['degraded_features'])}")
        
        # Circuit breaker status
        for name, breaker_status in status['circuit_breakers'].items():
            state = breaker_status['state']
            emoji = "🟢" if state == "closed" else "🔴" if state == "open" else "🟡"
            print(f"{name.title()} Circuit: {emoji} {state}")
        
        # Error patterns
        if status['error_patterns']:
            print(f"\n⚠️  Detected Error Patterns:")
            for pattern, details in status['error_patterns'].items():
                print(f"   - {pattern}: {details['count']} occurrences")
        
        print(f"\n🎉 Demo completed successfully!")
        print("The system demonstrated resilience through:")
        print("  • Automatic retry mechanisms")
        print("  • Circuit breaker protection")
        print("  • Graceful degradation under load")
        print("  • Fallback processing methods")
        print("  • Comprehensive error tracking")
        
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())