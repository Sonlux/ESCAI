"""
Performance tests for CLI large dataset handling
"""

import pytest
import time
import psutil
import threading
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from escai_framework.cli.main import cli
from escai_framework.cli.utils.data_filters import DataFilter
from escai_framework.cli.utils.ascii_viz import ASCIIBarChart, ChartConfig
from escai_framework.cli.utils.formatters import format_agent_status_table, format_behavioral_patterns


class TestCLIPerformance:
    """Test CLI performance with large datasets"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
        self.process = psutil.Process()
    
    def generate_large_agent_dataset(self, size=10000):
        """Generate large dataset for testing"""
        return [
            {
                'id': f'agent_{i:06d}',
                'status': 'active' if i % 3 == 0 else 'idle',
                'framework': ['langchain', 'autogen', 'crewai', 'openai'][i % 4],
                'uptime': f'{i % 24}h {i % 60}m',
                'event_count': i * 10 + (i % 100),
                'last_activity': f'{i % 60}s ago',
                'confidence': 0.5 + (i % 50) / 100.0,
                'metadata': {
                    'priority': ['high', 'medium', 'low'][i % 3],
                    'tags': [f'tag_{j}' for j in range(i % 5)]
                }
            }
            for i in range(size)
        ]
    
    def generate_large_pattern_dataset(self, size=5000):
        """Generate large behavioral pattern dataset"""
        return [
            {
                'pattern_name': f'Pattern_{i:04d}',
                'frequency': i % 1000 + 1,
                'success_rate': 0.1 + (i % 90) / 100.0,
                'average_duration': f'{(i % 10) + 1}.{i % 10}s',
                'statistical_significance': 0.5 + (i % 50) / 100.0
            }
            for i in range(size)
        ]
    
    def measure_memory_usage(self):
        """Measure current memory usage"""
        return self.process.memory_info().rss / 1024 / 1024  # MB
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def test_large_agent_status_table_performance(self):
        """Test performance of agent status table with large dataset"""
        dataset_sizes = [1000, 5000, 10000]
        
        for size in dataset_sizes:
            agents = self.generate_large_agent_dataset(size)
            
            # Measure memory before
            memory_before = self.measure_memory_usage()
            
            # Measure execution time
            table, execution_time = self.measure_execution_time(
                format_agent_status_table, agents
            )
            
            # Measure memory after
            memory_after = self.measure_memory_usage()
            memory_used = memory_after - memory_before
            
            # Performance assertions
            assert execution_time < 5.0, f"Table generation took {execution_time:.2f}s for {size} agents"
            assert memory_used < 100, f"Memory usage {memory_used:.2f}MB too high for {size} agents"
            assert table is not None
            assert len(table.rows) == size
            
            print(f"Size: {size}, Time: {execution_time:.3f}s, Memory: {memory_used:.2f}MB")
    
    def test_large_pattern_analysis_performance(self):
        """Test performance of pattern analysis with large dataset"""
        dataset_sizes = [1000, 3000, 5000]
        
        for size in dataset_sizes:
            patterns = self.generate_large_pattern_dataset(size)
            
            # Measure memory before
            memory_before = self.measure_memory_usage()
            
            # Measure execution time
            table, execution_time = self.measure_execution_time(
                format_behavioral_patterns, patterns
            )
            
            # Measure memory after
            memory_after = self.measure_memory_usage()
            memory_used = memory_after - memory_before
            
            # Performance assertions
            assert execution_time < 3.0, f"Pattern analysis took {execution_time:.2f}s for {size} patterns"
            assert memory_used < 50, f"Memory usage {memory_used:.2f}MB too high for {size} patterns"
            assert table is not None
            assert len(table.rows) == size
            
            print(f"Patterns - Size: {size}, Time: {execution_time:.3f}s, Memory: {memory_used:.2f}MB")
    
    def test_data_filtering_performance(self):
        """Test performance of data filtering with large datasets"""
        filter_engine = DataFilter()
        dataset_sizes = [5000, 10000, 20000]
        
        for size in dataset_sizes:
            data = self.generate_large_agent_dataset(size)
            
            # Test different filter operations
            filter_operations = [
                ('status equals active', lambda d: d['status'] == 'active'),
                ('confidence > 0.8', lambda d: d['confidence'] > 0.8),
                ('framework in langchain,autogen', lambda d: d['framework'] in ['langchain', 'autogen']),
                ('complex filter', lambda d: d['status'] == 'active' and d['confidence'] > 0.7 and d['framework'] == 'langchain')
            ]
            
            for filter_name, filter_func in filter_operations:
                # Measure execution time
                filtered_data, execution_time = self.measure_execution_time(
                    lambda: [item for item in data if filter_func(item)]
                )
                
                # Performance assertions
                assert execution_time < 2.0, f"{filter_name} took {execution_time:.2f}s for {size} items"
                assert isinstance(filtered_data, list)
                
                print(f"Filter '{filter_name}' - Size: {size}, Time: {execution_time:.3f}s, Results: {len(filtered_data)}")
    
    def test_search_performance(self):
        """Test performance of search functionality"""
        filter_engine = DataFilter()
        dataset_sizes = [5000, 10000, 15000]
        
        for size in dataset_sizes:
            data = self.generate_large_agent_dataset(size)
            
            # Test different search queries
            search_queries = [
                'agent_001',
                'active',
                'langchain',
                'high'
            ]
            
            for query in search_queries:
                # Measure execution time
                results, execution_time = self.measure_execution_time(
                    filter_engine.quick_search, data, query
                )
                
                # Performance assertions
                assert execution_time < 1.0, f"Search '{query}' took {execution_time:.2f}s for {size} items"
                assert isinstance(results, list)
                
                print(f"Search '{query}' - Size: {size}, Time: {execution_time:.3f}s, Results: {len(results)}")
    
    def test_chart_generation_performance(self):
        """Test performance of ASCII chart generation"""
        data_sizes = [100, 500, 1000, 2000]
        
        for size in data_sizes:
            # Generate test data
            data = [i * 0.1 + (i % 10) * 0.01 for i in range(size)]
            labels = [f'Item_{i}' for i in range(size)]
            
            config = ChartConfig(
                width=80,
                height=20,
                title=f"Performance Test Chart ({size} items)",
                color_scheme="default"
            )
            
            chart = ASCIIBarChart(config)
            
            # Measure execution time
            result, execution_time = self.measure_execution_time(
                chart.create, data, labels
            )
            
            # Performance assertions
            assert execution_time < 2.0, f"Chart generation took {execution_time:.2f}s for {size} data points"
            assert isinstance(result, str)
            assert len(result) > 0
            
            print(f"Chart - Size: {size}, Time: {execution_time:.3f}s")
    
    def test_concurrent_cli_operations(self):
        """Test performance of concurrent CLI operations"""
        def run_cli_command(command_args):
            """Run a CLI command and measure performance"""
            start_time = time.time()
            
            with patch('escai_framework.cli.commands.monitor.console'):
                with patch('escai_framework.cli.commands.analyze.console'):
                    result = self.runner.invoke(cli, command_args)
            
            end_time = time.time()
            return {
                'command': ' '.join(command_args),
                'exit_code': result.exit_code,
                'execution_time': end_time - start_time
            }
        
        # Define concurrent operations
        commands = [
            ['monitor', 'status'],
            ['analyze', 'patterns', '--agent-id', 'test_agent_1'],
            ['analyze', 'causal', '--min-strength', '0.7'],
            ['analyze', 'predictions', '--agent-id', 'test_agent_2'],
            ['analyze', 'health'],
            ['session', 'list'],
            ['config', 'check'],
            ['analyze', 'visualize', '--chart-type', 'bar']
        ]
        
        # Run commands concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_cli_command, cmd) for cmd in commands]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 10.0, f"Concurrent operations took {total_time:.2f}s"
        
        for result in results:
            assert result['exit_code'] == 0, f"Command '{result['command']}' failed"
            assert result['execution_time'] < 5.0, f"Command '{result['command']}' took {result['execution_time']:.2f}s"
        
        print(f"Concurrent operations - Total time: {total_time:.3f}s")
        for result in results:
            print(f"  {result['command']}: {result['execution_time']:.3f}s")
    
    def test_memory_usage_with_large_datasets(self):
        """Test memory usage patterns with large datasets"""
        initial_memory = self.measure_memory_usage()
        
        # Process increasingly large datasets
        dataset_sizes = [1000, 5000, 10000, 20000]
        memory_measurements = []
        
        for size in dataset_sizes:
            # Generate large dataset
            data = self.generate_large_agent_dataset(size)
            
            # Process the data (simulate CLI operations)
            filter_engine = DataFilter()
            
            # Perform various operations
            filtered_data = [item for item in data if item['status'] == 'active']
            search_results = filter_engine.quick_search(data, 'agent')
            table = format_agent_status_table(data[:1000])  # Limit table size for performance
            
            # Measure memory
            current_memory = self.measure_memory_usage()
            memory_used = current_memory - initial_memory
            memory_measurements.append((size, memory_used))
            
            # Clean up
            del data, filtered_data, search_results, table
            
            print(f"Dataset size: {size}, Memory used: {memory_used:.2f}MB")
        
        # Check memory growth is reasonable
        for i in range(1, len(memory_measurements)):
            prev_size, prev_memory = memory_measurements[i-1]
            curr_size, curr_memory = memory_measurements[i]
            
            # Memory should not grow exponentially
            size_ratio = curr_size / prev_size
            memory_ratio = curr_memory / prev_memory if prev_memory > 0 else 1
            
            assert memory_ratio < size_ratio * 2, f"Memory growth too high: {memory_ratio:.2f}x for {size_ratio:.2f}x data"
    
    def test_cli_startup_performance(self):
        """Test CLI startup performance"""
        startup_times = []
        
        # Measure startup time multiple times
        for i in range(5):
            start_time = time.time()
            
            # Simulate CLI startup
            result = self.runner.invoke(cli, ['--help'])
            
            end_time = time.time()
            startup_time = end_time - start_time
            startup_times.append(startup_time)
            
            assert result.exit_code == 0
        
        # Calculate average startup time
        avg_startup_time = sum(startup_times) / len(startup_times)
        max_startup_time = max(startup_times)
        
        # Performance assertions
        assert avg_startup_time < 2.0, f"Average startup time {avg_startup_time:.2f}s too slow"
        assert max_startup_time < 3.0, f"Max startup time {max_startup_time:.2f}s too slow"
        
        print(f"Startup performance - Avg: {avg_startup_time:.3f}s, Max: {max_startup_time:.3f}s")
    
    def test_export_performance_large_datasets(self):
        """Test export performance with large datasets"""
        dataset_sizes = [1000, 5000, 10000]
        
        for size in dataset_sizes:
            data = self.generate_large_agent_dataset(size)
            
            # Test JSON export performance
            start_time = time.time()
            json_data = json.dumps(data)
            json_time = time.time() - start_time
            
            # Test CSV-like export performance (simulate)
            start_time = time.time()
            csv_lines = []
            if data:
                # Header
                headers = list(data[0].keys())
                csv_lines.append(','.join(headers))
                
                # Data rows
                for item in data:
                    row = [str(item.get(header, '')) for header in headers]
                    csv_lines.append(','.join(row))
            
            csv_data = '\n'.join(csv_lines)
            csv_time = time.time() - start_time
            
            # Performance assertions
            assert json_time < 2.0, f"JSON export took {json_time:.2f}s for {size} items"
            assert csv_time < 3.0, f"CSV export took {csv_time:.2f}s for {size} items"
            
            print(f"Export - Size: {size}, JSON: {json_time:.3f}s, CSV: {csv_time:.3f}s")
    
    def test_pagination_performance(self):
        """Test pagination performance with large datasets"""
        data = self.generate_large_agent_dataset(10000)
        page_sizes = [10, 50, 100, 500]
        
        for page_size in page_sizes:
            total_pages = len(data) // page_size + (1 if len(data) % page_size else 0)
            
            # Test pagination performance
            start_time = time.time()
            
            for page in range(min(10, total_pages)):  # Test first 10 pages
                start_idx = page * page_size
                end_idx = min(start_idx + page_size, len(data))
                page_data = data[start_idx:end_idx]
                
                # Simulate processing page data
                table = format_agent_status_table(page_data)
                assert table is not None
            
            pagination_time = time.time() - start_time
            
            # Performance assertions
            assert pagination_time < 1.0, f"Pagination took {pagination_time:.2f}s for page size {page_size}"
            
            print(f"Pagination - Page size: {page_size}, Time: {pagination_time:.3f}s")


class TestCLIScalability:
    """Test CLI scalability under various conditions"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_multiple_agent_monitoring_scalability(self):
        """Test scalability when monitoring multiple agents"""
        agent_counts = [10, 50, 100]
        
        for count in agent_counts:
            start_time = time.time()
            
            # Simulate starting monitoring for multiple agents
            for i in range(count):
                with patch('escai_framework.cli.commands.monitor.console'):
                    result = self.runner.invoke(cli, [
                        'monitor', 'start',
                        '--agent-id', f'scale_test_agent_{i:03d}',
                        '--framework', 'langchain'
                    ])
                    
                    assert result.exit_code == 0
            
            total_time = time.time() - start_time
            avg_time_per_agent = total_time / count
            
            # Scalability assertions
            assert avg_time_per_agent < 0.5, f"Average time per agent {avg_time_per_agent:.3f}s too slow"
            assert total_time < count * 0.2, f"Total time {total_time:.2f}s not scaling well"
            
            print(f"Agent count: {count}, Total time: {total_time:.3f}s, Avg per agent: {avg_time_per_agent:.3f}s")
    
    def test_analysis_command_scalability(self):
        """Test scalability of analysis commands"""
        analysis_commands = [
            ['analyze', 'patterns'],
            ['analyze', 'causal'],
            ['analyze', 'predictions'],
            ['analyze', 'health'],
            ['analyze', 'visualize', '--chart-type', 'bar']
        ]
        
        execution_counts = [1, 5, 10, 20]
        
        for count in execution_counts:
            start_time = time.time()
            
            # Execute analysis commands multiple times
            for i in range(count):
                for cmd in analysis_commands:
                    with patch('escai_framework.cli.commands.analyze.console'):
                        result = self.runner.invoke(cli, cmd)
                        assert result.exit_code == 0
            
            total_time = time.time() - start_time
            avg_time_per_execution = total_time / (count * len(analysis_commands))
            
            # Scalability assertions
            assert avg_time_per_execution < 0.1, f"Average execution time {avg_time_per_execution:.3f}s too slow"
            
            print(f"Executions: {count}, Total time: {total_time:.3f}s, Avg per cmd: {avg_time_per_execution:.3f}s")


if __name__ == '__main__':
    pytest.main([__file__])