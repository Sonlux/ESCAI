"""
Example demonstrating ESCAI CLI framework integration.

This example shows how to use the CLI to monitor different agent frameworks
and validate that the integration is working correctly.
"""

import asyncio
import time
from typing import Dict, Any, List

# Import ESCAI CLI integration components
from escai_framework.cli.integration.framework_connector import (
    FrameworkConnector, get_framework_connector, framework_context
)
from escai_framework.instrumentation.events import EventType, EventSeverity


async def demonstrate_framework_validation():
    """Demonstrate framework validation capabilities."""
    print("🔍 Demonstrating Framework Validation")
    print("=" * 50)
    
    async with framework_context() as connector:
        # Check available frameworks
        available = connector.get_available_frameworks()
        print(f"Available frameworks: {', '.join(available) if available else 'None'}")
        
        # Validate each framework
        frameworks = ['langchain', 'autogen', 'crewai', 'openai']
        
        for framework in frameworks:
            print(f"\n📋 Validating {framework}...")
            
            result = await connector.validate_framework_integration(framework)
            
            print(f"  Available: {'✅' if result['available'] else '❌'}")
            print(f"  Instrumentor: {'✅' if result['instrumentor_created'] else '❌'}")
            print(f"  Test Monitoring: {'✅' if result['test_monitoring'] else '❌'}")
            
            if result['events_supported']:
                print(f"  Supported Events: {len(result['events_supported'])}")
            
            if result['errors']:
                print(f"  ❌ Errors: {len(result['errors'])}")
                for error in result['errors'][:2]:  # Show first 2 errors
                    print(f"    • {error}")
            
            if result['warnings']:
                print(f"  ⚠️  Warnings: {len(result['warnings'])}")


async def demonstrate_langchain_monitoring():
    """Demonstrate LangChain monitoring (if available)."""
    print("\n🔗 Demonstrating LangChain Monitoring")
    print("=" * 50)
    
    try:
        # Try to import LangChain to check availability
        import langchain
        print("LangChain is available - demonstrating monitoring")
        
        async with framework_context() as connector:
            # Check if LangChain is available through connector
            if 'langchain' not in connector.get_available_frameworks():
                print("❌ LangChain not available through connector")
                return
            
            # Start monitoring
            config = {
                'capture_epistemic_states': True,
                'capture_behavioral_patterns': True,
                'capture_performance_metrics': True,
                'monitor_memory': True,
                'monitor_context': True
            }
            
            print("🚀 Starting LangChain monitoring...")
            session_id = await connector.start_monitoring(
                agent_id="langchain_demo_agent",
                framework="langchain",
                config=config
            )
            
            print(f"✅ Monitoring started - Session ID: {session_id}")
            
            # Simulate some monitoring time
            print("⏱️  Monitoring for 3 seconds...")
            await asyncio.sleep(3)
            
            # Get session status
            sessions = await connector.get_session_status(session_id)
            if sessions:
                session = sessions[0]
                print(f"📊 Session Status:")
                print(f"  Agent ID: {session['agent_id']}")
                print(f"  Framework: {session['framework']}")
                print(f"  Uptime: {session['uptime_formatted']}")
                print(f"  Status: {session['status']}")
            
            # Get epistemic state
            epistemic = await connector.get_epistemic_state("langchain_demo_agent", session_id)
            print(f"🧠 Epistemic State: {epistemic['status']}")
            
            # Stop monitoring
            print("🛑 Stopping monitoring...")
            result = await connector.stop_monitoring(session_id)
            print(f"✅ Monitoring stopped - Duration: {result.get('summary', {}).get('total_duration_ms', 0)}ms")
            
    except ImportError:
        print("❌ LangChain not installed - skipping demonstration")
        print("   Install with: pip install langchain")


async def demonstrate_autogen_monitoring():
    """Demonstrate AutoGen monitoring (if available)."""
    print("\n🤖 Demonstrating AutoGen Monitoring")
    print("=" * 50)
    
    try:
        # Try to import AutoGen to check availability
        import autogen
        print("AutoGen is available - demonstrating monitoring")
        
        async with framework_context() as connector:
            # Check if AutoGen is available through connector
            if 'autogen' not in connector.get_available_frameworks():
                print("❌ AutoGen not available through connector")
                return
            
            # Create mock agents for demonstration
            mock_agents = [
                type('MockAgent', (), {'name': 'agent_1', 'system_message': 'Assistant agent'}),
                type('MockAgent', (), {'name': 'agent_2', 'system_message': 'User proxy agent'})
            ]
            
            # Start monitoring
            config = {
                'agents': mock_agents,
                'capture_epistemic_states': True,
                'capture_behavioral_patterns': True,
                'monitor_conversations': True,
                'monitor_decisions': True
            }
            
            print("🚀 Starting AutoGen monitoring...")
            session_id = await connector.start_monitoring(
                agent_id="autogen_demo_system",
                framework="autogen",
                config=config
            )
            
            print(f"✅ Monitoring started - Session ID: {session_id}")
            
            # Simulate monitoring
            print("⏱️  Monitoring for 2 seconds...")
            await asyncio.sleep(2)
            
            # Stop monitoring
            print("🛑 Stopping monitoring...")
            result = await connector.stop_monitoring(session_id)
            print(f"✅ Monitoring stopped")
            
    except ImportError:
        print("❌ AutoGen not installed - skipping demonstration")
        print("   Install with: pip install pyautogen")


async def demonstrate_error_handling():
    """Demonstrate error handling in framework integration."""
    print("\n⚠️  Demonstrating Error Handling")
    print("=" * 50)
    
    async with framework_context() as connector:
        # Test invalid framework
        try:
            await connector.start_monitoring(
                agent_id="test_agent",
                framework="invalid_framework",
                config={}
            )
        except Exception as e:
            print(f"✅ Caught expected error for invalid framework: {type(e).__name__}")
        
        # Test stopping non-existent session
        try:
            await connector.stop_monitoring("non_existent_session")
        except Exception as e:
            print(f"✅ Caught expected error for non-existent session: {type(e).__name__}")
        
        # Test getting epistemic state for non-monitored agent
        result = await connector.get_epistemic_state("non_monitored_agent")
        print(f"✅ Non-monitored agent status: {result['status']}")


async def demonstrate_event_handling():
    """Demonstrate event handling capabilities."""
    print("\n📡 Demonstrating Event Handling")
    print("=" * 50)
    
    events_received = []
    
    def event_handler(event):
        """Simple event handler for demonstration."""
        events_received.append({
            'type': event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
            'agent_id': event.agent_id,
            'message': event.message,
            'timestamp': event.timestamp.isoformat() if hasattr(event, 'timestamp') else 'unknown'
        })
        print(f"📨 Event received: {event.event_type} - {event.message}")
    
    async with framework_context() as connector:
        # Add event handler
        connector.add_event_handler(event_handler)
        print("✅ Event handler added")
        
        # If we have any available frameworks, start monitoring briefly
        available = connector.get_available_frameworks()
        if available:
            framework = available[0]
            print(f"🚀 Starting brief monitoring with {framework} to capture events...")
            
            try:
                session_id = await connector.start_monitoring(
                    agent_id="event_demo_agent",
                    framework=framework,
                    config={'capture_epistemic_states': True}
                )
                
                # Wait briefly for events
                await asyncio.sleep(1)
                
                # Stop monitoring
                await connector.stop_monitoring(session_id)
                
                print(f"📊 Total events captured: {len(events_received)}")
                
            except Exception as e:
                print(f"⚠️  Event monitoring failed: {e}")
        else:
            print("❌ No frameworks available for event demonstration")
        
        # Remove event handler
        connector.remove_event_handler(event_handler)
        print("✅ Event handler removed")


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n📈 Demonstrating Performance Monitoring")
    print("=" * 50)
    
    async with framework_context() as connector:
        available = connector.get_available_frameworks()
        
        if not available:
            print("❌ No frameworks available for performance demonstration")
            return
        
        framework = available[0]
        print(f"🚀 Starting performance monitoring with {framework}...")
        
        try:
            # Start monitoring with performance tracking
            config = {
                'capture_performance_metrics': True,
                'max_events_per_second': 50,
                'buffer_size': 500
            }
            
            session_id = await connector.start_monitoring(
                agent_id="performance_demo_agent",
                framework=framework,
                config=config
            )
            
            # Monitor for a few seconds
            for i in range(3):
                await asyncio.sleep(1)
                sessions = await connector.get_session_status(session_id)
                if sessions:
                    session = sessions[0]
                    stats = session.get('monitoring_stats', {})
                    perf = session.get('performance_metrics', {})
                    
                    print(f"⏱️  Second {i+1}:")
                    print(f"   Events captured: {stats.get('events_captured', 0)}")
                    print(f"   Performance overhead: {stats.get('performance_overhead', 0):.3f}")
                    print(f"   Errors: {perf.get('errors_encountered', 0)}")
            
            # Stop monitoring
            result = await connector.stop_monitoring(session_id)
            print(f"✅ Performance monitoring completed")
            
            # Show final metrics
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                print(f"📊 Final Performance Metrics:")
                for key, value in metrics.items():
                    print(f"   {key}: {value}")
        
        except Exception as e:
            print(f"⚠️  Performance monitoring failed: {e}")


async def main():
    """Run all demonstrations."""
    print("🎯 ESCAI CLI Framework Integration Demonstration")
    print("=" * 60)
    print()
    
    try:
        # Run all demonstrations
        await demonstrate_framework_validation()
        await demonstrate_langchain_monitoring()
        await demonstrate_autogen_monitoring()
        await demonstrate_error_handling()
        await demonstrate_event_handling()
        await demonstrate_performance_monitoring()
        
        print("\n🎉 All demonstrations completed!")
        print("\nTo use the CLI commands:")
        print("  escai monitor validate                    # Validate all frameworks")
        print("  escai monitor validate --framework langchain  # Validate specific framework")
        print("  escai monitor start --agent-id my_agent --framework langchain")
        print("  escai monitor status                      # View active sessions")
        print("  escai monitor epistemic --agent-id my_agent")
        print("  escai monitor stop --session-id <session_id>")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())