"""
Integration tests for CLI framework interactions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from click.testing import CliRunner
import asyncio
import json
import tempfile
from pathlib import Path

from escai_framework.cli.main import cli
from escai_framework.cli.integration.framework_connector import FrameworkConnector
from escai_framework.cli.services.api_client import ESCAIAPIClient


class TestFrameworkIntegration:
    """Test CLI integration with agent frameworks"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
        self.connector = FrameworkConnector()
    
    @pytest.mark.asyncio
    async def test_langchain_integration(self):
        """Test LangChain framework integration"""
        with patch('escai_framework.instrumentation.langchain_instrumentor.LangChainInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance
            mock_instance.start_monitoring = AsyncMock()
            mock_instance.stop_monitoring = AsyncMock()
            mock_instance.get_status = AsyncMock(return_value={'status': 'active'})
            
            # Test starting monitoring
            result = await self.connector.start_monitoring('test_agent', 'langchain', {})
            assert result is not None
            
            # Test getting status
            status = await self.connector.get_agent_status('test_agent')
            assert status['status'] == 'active'
            
            # Test stopping monitoring
            await self.connector.stop_monitoring('test_agent')
    
    @pytest.mark.asyncio
    async def test_autogen_integration(self):
        """Test AutoGen framework integration"""
        with patch('escai_framework.instrumentation.autogen_instrumentor.AutoGenInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance
            mock_instance.start_monitoring = AsyncMock()
            mock_instance.stop_monitoring = AsyncMock()
            mock_instance.get_status = AsyncMock(return_value={'status': 'active'})
            
            # Test starting monitoring
            result = await self.connector.start_monitoring('test_agent', 'autogen', {})
            assert result is not None
            
            # Test getting status
            status = await self.connector.get_agent_status('test_agent')
            assert status['status'] == 'active'
            
            # Test stopping monitoring
            await self.connector.stop_monitoring('test_agent')
    
    @pytest.mark.asyncio
    async def test_crewai_integration(self):
        """Test CrewAI framework integration"""
        with patch('escai_framework.instrumentation.crewai_instrumentor.CrewAIInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance
            mock_instance.start_monitoring = AsyncMock()
            mock_instance.stop_monitoring = AsyncMock()
            mock_instance.get_status = AsyncMock(return_value={'status': 'active'})
            
            # Test starting monitoring
            result = await self.connector.start_monitoring('test_agent', 'crewai', {})
            assert result is not None
            
            # Test getting status
            status = await self.connector.get_agent_status('test_agent')
            assert status['status'] == 'active'
            
            # Test stopping monitoring
            await self.connector.stop_monitoring('test_agent')
    
    @pytest.mark.asyncio
    async def test_openai_integration(self):
        """Test OpenAI Assistants framework integration"""
        with patch('escai_framework.instrumentation.openai_instrumentor.OpenAIInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance
            mock_instance.start_monitoring = AsyncMock()
            mock_instance.stop_monitoring = AsyncMock()
            mock_instance.get_status = AsyncMock(return_value={'status': 'active'})
            
            # Test starting monitoring
            result = await self.connector.start_monitoring('test_agent', 'openai', {})
            assert result is not None
            
            # Test getting status
            status = await self.connector.get_agent_status('test_agent')
            assert status['status'] == 'active'
            
            # Test stopping monitoring
            await self.connector.stop_monitoring('test_agent')
    
    def test_framework_validation(self):
        """Test framework validation"""
        # Test valid frameworks
        valid_frameworks = ['langchain', 'autogen', 'crewai', 'openai']
        for framework in valid_frameworks:
            assert self.connector.is_valid_framework(framework)
        
        # Test invalid framework
        assert not self.connector.is_valid_framework('invalid_framework')
    
    @pytest.mark.asyncio
    async def test_framework_error_handling(self):
        """Test framework error handling"""
        with patch('escai_framework.instrumentation.langchain_instrumentor.LangChainInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance
            mock_instance.start_monitoring = AsyncMock(side_effect=Exception("Connection failed"))
            
            # Should handle errors gracefully
            result = await self.connector.start_monitoring('test_agent', 'langchain', {})
            assert result is None
    
    def test_cli_framework_integration_commands(self):
        """Test CLI commands with framework integration"""
        # Test monitor start with different frameworks
        frameworks = ['langchain', 'autogen', 'crewai', 'openai']
        
        for framework in frameworks:
            with patch('escai_framework.cli.commands.monitor.console') as mock_console:
                result = self.runner.invoke(cli, [
                    'monitor', 'start',
                    '--agent-id', f'test_agent_{framework}',
                    '--framework', framework
                ])
                
                assert result.exit_code == 0
                mock_console.print.assert_called()
    
    def test_cli_invalid_framework_error(self):
        """Test CLI error handling for invalid framework"""
        result = self.runner.invoke(cli, [
            'monitor', 'start',
            '--agent-id', 'test_agent',
            '--framework', 'invalid_framework'
        ])
        
        assert result.exit_code != 0
    
    @pytest.mark.asyncio
    async def test_multiple_framework_monitoring(self):
        """Test monitoring multiple frameworks simultaneously"""
        frameworks = ['langchain', 'autogen']
        
        for framework in frameworks:
            with patch(f'escai_framework.instrumentation.{framework}_instrumentor.{framework.title()}Instrumentor') as mock_instrumentor:
                mock_instance = MagicMock()
                mock_instrumentor.return_value = mock_instance
                mock_instance.start_monitoring = AsyncMock()
                mock_instance.get_status = AsyncMock(return_value={'status': 'active'})
                
                result = await self.connector.start_monitoring(f'agent_{framework}', framework, {})
                assert result is not None
        
        # Test getting status for all agents
        all_status = await self.connector.get_all_agent_status()
        assert isinstance(all_status, list)


class TestAPIClientIntegration:
    """Test CLI API client integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.api_client = ESCAIAPIClient(base_url="http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_start_monitoring_api(self):
        """Test start monitoring API call"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {'session_id': 'test_session_123'}
            mock_response.raise_for_status.return_value = None
            
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await self.api_client.start_monitoring('test_agent', 'langchain', {})
            assert result == {'session_id': 'test_session_123'}
    
    @pytest.mark.asyncio
    async def test_stop_monitoring_api(self):
        """Test stop monitoring API call"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {'status': 'stopped'}
            mock_response.raise_for_status.return_value = None
            
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await self.api_client.stop_monitoring('test_session')
            assert result == {'status': 'stopped'}
    
    @pytest.mark.asyncio
    async def test_get_agent_status_api(self):
        """Test get agent status API call"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {'id': 'agent_001', 'status': 'active', 'framework': 'langchain'},
                {'id': 'agent_002', 'status': 'idle', 'framework': 'autogen'}
            ]
            mock_response.raise_for_status.return_value = None
            
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await self.api_client.get_agent_status()
            assert len(result) == 2
            assert result[0]['id'] == 'agent_001'
            assert result[1]['framework'] == 'autogen'
    
    @pytest.mark.asyncio
    async def test_get_epistemic_state_api(self):
        """Test get epistemic state API call"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'agent_id': 'test_agent',
                'beliefs': [{'content': 'Test belief', 'confidence': 0.9}],
                'knowledge': {'fact_count': 10},
                'goals': [{'description': 'Test goal', 'progress': 0.5}],
                'uncertainty_score': 0.3
            }
            mock_response.raise_for_status.return_value = None
            
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await self.api_client.get_epistemic_state('test_agent')
            assert result['agent_id'] == 'test_agent'
            assert len(result['beliefs']) == 1
            assert result['uncertainty_score'] == 0.3
    
    @pytest.mark.asyncio
    async def test_get_behavioral_patterns_api(self):
        """Test get behavioral patterns API call"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    'pattern_name': 'Sequential Processing',
                    'frequency': 45,
                    'success_rate': 0.89,
                    'average_duration': '2.3s',
                    'statistical_significance': 0.95
                }
            ]
            mock_response.raise_for_status.return_value = None
            
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await self.api_client.get_behavioral_patterns('test_agent')
            assert len(result) == 1
            assert result[0]['pattern_name'] == 'Sequential Processing'
            assert result[0]['success_rate'] == 0.89
    
    @pytest.mark.asyncio
    async def test_get_causal_relationships_api(self):
        """Test get causal relationships API call"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    'cause_event': 'Data Validation Error',
                    'effect_event': 'Retry Mechanism Triggered',
                    'strength': 0.87,
                    'confidence': 0.92
                }
            ]
            mock_response.raise_for_status.return_value = None
            
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await self.api_client.get_causal_relationships()
            assert len(result) == 1
            assert result[0]['cause_event'] == 'Data Validation Error'
            assert result[0]['strength'] == 0.87
    
    @pytest.mark.asyncio
    async def test_get_predictions_api(self):
        """Test get predictions API call"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    'predicted_outcome': 'success',
                    'confidence': 0.87,
                    'risk_factors': ['High memory usage'],
                    'trend': 'improving'
                }
            ]
            mock_response.raise_for_status.return_value = None
            
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await self.api_client.get_predictions('test_agent')
            assert len(result) == 1
            assert result[0]['predicted_outcome'] == 'success'
            assert result[0]['confidence'] == 0.87
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test API error handling"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.get = AsyncMock(side_effect=Exception("Connection error"))
            mock_client.return_value = mock_context
            
            result = await self.api_client.get_agent_status()
            assert result == []  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_api_timeout_handling(self):
        """Test API timeout handling"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_context.get = AsyncMock(side_effect=asyncio.TimeoutError("Request timeout"))
            mock_client.return_value = mock_context
            
            result = await self.api_client.get_agent_status()
            assert result == []  # Should return empty list on timeout


class TestFrameworkSpecificFeatures:
    """Test framework-specific feature integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_langchain_chain_monitoring(self):
        """Test LangChain chain execution monitoring"""
        with patch('escai_framework.cli.commands.monitor.console') as mock_console:
            result = self.runner.invoke(cli, [
                'monitor', 'start',
                '--agent-id', 'langchain_agent',
                '--framework', 'langchain',
                '--chain-type', 'sequential'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
    
    def test_autogen_conversation_monitoring(self):
        """Test AutoGen multi-agent conversation monitoring"""
        with patch('escai_framework.cli.commands.monitor.console') as mock_console:
            result = self.runner.invoke(cli, [
                'monitor', 'start',
                '--agent-id', 'autogen_group',
                '--framework', 'autogen',
                '--conversation-id', 'conv_123'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
    
    def test_crewai_task_monitoring(self):
        """Test CrewAI task delegation monitoring"""
        with patch('escai_framework.cli.commands.monitor.console') as mock_console:
            result = self.runner.invoke(cli, [
                'monitor', 'start',
                '--agent-id', 'crewai_crew',
                '--framework', 'crewai',
                '--crew-id', 'crew_456'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
    
    def test_openai_assistant_monitoring(self):
        """Test OpenAI Assistant tool usage monitoring"""
        with patch('escai_framework.cli.commands.monitor.console') as mock_console:
            result = self.runner.invoke(cli, [
                'monitor', 'start',
                '--agent-id', 'openai_assistant',
                '--framework', 'openai',
                '--assistant-id', 'asst_789'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
    
    def test_framework_specific_analysis(self):
        """Test framework-specific analysis commands"""
        frameworks = ['langchain', 'autogen', 'crewai', 'openai']
        
        for framework in frameworks:
            with patch('escai_framework.cli.commands.analyze.console') as mock_console:
                result = self.runner.invoke(cli, [
                    'analyze', 'patterns',
                    '--framework', framework,
                    '--agent-id', f'{framework}_agent'
                ])
                
                assert result.exit_code == 0
                mock_console.print.assert_called()


class TestRealTimeIntegration:
    """Test real-time monitoring integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_live_monitoring_integration(self):
        """Test live monitoring with framework integration"""
        with patch('escai_framework.cli.commands.monitor.console') as mock_console:
            with patch('escai_framework.cli.commands.monitor.Live') as mock_live:
                mock_live_instance = MagicMock()
                mock_live.return_value.__enter__.return_value = mock_live_instance
                mock_live.return_value.__exit__.return_value = None
                
                result = self.runner.invoke(cli, [
                    'monitor', 'live',
                    '--agent-id', 'test_agent',
                    '--framework', 'langchain'
                ])
                
                assert result.exit_code == 0
    
    def test_dashboard_integration(self):
        """Test dashboard integration with multiple frameworks"""
        with patch('escai_framework.cli.commands.monitor.console') as mock_console:
            result = self.runner.invoke(cli, [
                'monitor', 'dashboard',
                '--refresh-rate', '5'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
    
    def test_epistemic_state_monitoring(self):
        """Test real-time epistemic state monitoring"""
        with patch('escai_framework.cli.commands.monitor.console') as mock_console:
            result = self.runner.invoke(cli, [
                'monitor', 'epistemic',
                '--agent-id', 'test_agent',
                '--live'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()


class TestFrameworkErrorHandling:
    """Test framework-specific error handling"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
        self.connector = FrameworkConnector()
    
    @pytest.mark.asyncio
    async def test_framework_connection_failure(self):
        """Test handling of framework connection failures"""
        with patch('escai_framework.instrumentation.langchain_instrumentor.LangChainInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance
            mock_instance.start_monitoring = AsyncMock(side_effect=ConnectionError("Framework not available"))
            
            result = await self.connector.start_monitoring('test_agent', 'langchain', {})
            assert result is None
    
    @pytest.mark.asyncio
    async def test_framework_timeout_handling(self):
        """Test handling of framework timeouts"""
        with patch('escai_framework.instrumentation.autogen_instrumentor.AutoGenInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance
            mock_instance.start_monitoring = AsyncMock(side_effect=asyncio.TimeoutError("Operation timed out"))
            
            result = await self.connector.start_monitoring('test_agent', 'autogen', {})
            assert result is None
    
    def test_cli_framework_error_messages(self):
        """Test CLI error messages for framework issues"""
        with patch('escai_framework.cli.integration.framework_connector.FrameworkConnector.start_monitoring') as mock_start:
            mock_start.return_value = None  # Simulate failure
            
            result = self.runner.invoke(cli, [
                'monitor', 'start',
                '--agent-id', 'test_agent',
                '--framework', 'langchain'
            ])
            
            # Should handle error gracefully
            assert result.exit_code == 0
    
    def test_framework_compatibility_check(self):
        """Test framework compatibility checking"""
        # Test that connector can check framework availability
        assert hasattr(self.connector, 'is_valid_framework')
        
        # Test with known frameworks
        assert self.connector.is_valid_framework('langchain')
        assert self.connector.is_valid_framework('autogen')
        assert self.connector.is_valid_framework('crewai')
        assert self.connector.is_valid_framework('openai')
        
        # Test with invalid framework
        assert not self.connector.is_valid_framework('nonexistent')


if __name__ == '__main__':
    pytest.main([__file__])