"""
Unit tests for statistical analysis module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import asyncio
from datetime import datetime
from scipy import stats

from escai_framework.analytics.statistical_analysis import (
    StatisticalAnalyzer, StatisticalTest, HypothesisTest
)
from escai_framework.models.behavioral_pattern import BehavioralPattern, ExecutionSequence, ExecutionStep
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState


@pytest.fixture
def statistical_analyzer():
    """Create statistical analyzer instance."""
    return StatisticalAnalyzer(alpha=0.05)


@pytest.fixture
def sample_behavioral_pattern1():
    """Create first sample behavioral pattern."""
    sequences = []
    for i in range(10):
        steps = [
            ExecutionStep(
                step_id=f"step_{i}_{j}",
                step_type="action",
                action=f"action_{j}",
                duration=1.0,
                success_probability=0.8 + i * 0.01,
                context={},
                error_message=None
            )
            for j in range(3)
        ]
        
        sequences.append(ExecutionSequence(
            sequence_id=f"seq1_{i}",
            agent_id="agent1",
            steps=steps,
            total_duration=3.0,
            success_rate=0.8 + i * 0.01
        ))
    
    return BehavioralPattern(
        pattern_id="pattern1",
        pattern_name="High Success Pattern",
        execution_sequences=sequences,
        frequency=10,
        success_rate=0.85,
        average_duration=3.0,
        common_triggers=["trigger1"],
        failure_modes=[],
        statistical_significance=0.95
    )


@pytest.fixture
def sample_behavioral_pattern2():
    """Create second sample behavioral pattern."""
    sequences = []
    for i in range(8):
        steps = [
            ExecutionStep(
                step_id=f"step_{i}_{j}",
                step_type="action",
                action=f"action_{j}",
                duration=1.5,
                success_probability=0.6 + i * 0.02,
                context={},
                error_message=None
            )
            for j in range(3)
        ]
        
        sequences.append(ExecutionSequence(
            sequence_id=f"seq2_{i}",
            agent_id="agent2",
            steps=steps,
            total_duration=4.5,
            success_rate=0.6 + i * 0.02
        ))
    
    return BehavioralPattern(
        pattern_id="pattern2",
        pattern_name="Lower Success Pattern",
        execution_sequences=sequences,
        frequency=8,
        success_rate=0.68,
        average_duration=4.5,
        common_triggers=["trigger2"],
        failure_modes=["timeout"],
        statistical_significance=0.85
    )


@pytest.fixture
def sample_epistemic_states():
    """Create sample epistemic states."""
    states = []
    for i in range(20):
        belief_states = [
            BeliefState(
                belief_id=f"belief_{i}",
                content=f"belief_content_{i}",
                confidence=0.7 + i * 0.01,
                source="test",
                timestamp=datetime.now()
            )
        ]
        
        knowledge_state = KnowledgeState(
            facts=[f"fact_{i}"],
            rules=[f"rule_{i}"],
            concepts=[f"concept_{i}"]
        )
        
        goal_state = GoalState(
            active_goals=[f"goal_{i}"],
            completed_goals=[],
            failed_goals=[]
        )
        
        states.append(EpistemicState(
            agent_id=f"agent_{i % 3}",
            timestamp=datetime.now(),
            belief_states=belief_states,
            knowledge_state=knowledge_state,
            goal_state=goal_state,
            confidence_level=0.6 + i * 0.02,
            uncertainty_score=0.4 - i * 0.01,
            decision_context={}
        ))
    
    return states


class TestStatisticalTest:
    """Test StatisticalTest data structure."""
    
    def test_creation(self):
        """Test statistical test creation."""
        test = StatisticalTest(
            test_name="t-test",
            statistic=2.5,
            p_value=0.02,
            effect_size=0.8,
            confidence_interval=(0.1, 0.9),
            interpretation="Significant result",
            assumptions_met=True,
            sample_size=30
        )
        
        assert test.test_name == "t-test"
        assert test.statistic == 2.5
        assert test.p_value == 0.02
        assert test.effect_size == 0.8
        assert test.confidence_interval == (0.1, 0.9)
        assert test.interpretation == "Significant result"
        assert test.assumptions_met is True
        assert test.sample_size == 30


class TestHypothesisTest:
    """Test HypothesisTest data structure."""
    
    def test_creation(self):
        """Test hypothesis test creation."""
        stat_test = StatisticalTest(
            test_name="t-test",
            statistic=2.5,
            p_value=0.02,
            effect_size=0.8,
            confidence_interval=None,
            interpretation="Significant",
            assumptions_met=True,
            sample_size=30
        )
        
        hyp_test = HypothesisTest(
            null_hypothesis="No difference",
            alternative_hypothesis="Significant difference",
            alpha=0.05,
            test_result=stat_test,
            conclusion="Reject null hypothesis",
            practical_significance=True
        )
        
        assert hyp_test.null_hypothesis == "No difference"
        assert hyp_test.alternative_hypothesis == "Significant difference"
        assert hyp_test.alpha == 0.05
        assert hyp_test.test_result == stat_test
        assert hyp_test.conclusion == "Reject null hypothesis"
        assert hyp_test.practical_significance is True


class TestStatisticalAnalyzer:
    """Test statistical analyzer functionality."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = StatisticalAnalyzer(alpha=0.01)
        
        assert analyzer.alpha == 0.01
        assert analyzer.test_history == []
    
    @pytest.mark.asyncio
    async def test_compare_behavioral_patterns(self, statistical_analyzer, 
                                             sample_behavioral_pattern1, 
                                             sample_behavioral_pattern2):
        """Test behavioral pattern comparison."""
        with patch.object(statistical_analyzer, '_check_assumptions') as mock_assumptions, \
             patch.object(statistical_analyzer, '_independent_t_test') as mock_t_test:
            
            # Mock assumptions check
            mock_assumptions.return_value = {
                'normality': True,
                'equal_variance': True,
                'sample_sizes': (10, 8)
            }
            
            # Mock t-test result
            mock_test_result = StatisticalTest(
                test_name="Independent Samples t-test",
                statistic=2.5,
                p_value=0.02,
                effect_size=0.8,
                confidence_interval=None,
                interpretation="Significant difference",
                assumptions_met=True,
                sample_size=18
            )
            mock_t_test.return_value = mock_test_result
            
            result = await statistical_analyzer.compare_behavioral_patterns(
                sample_behavioral_pattern1, sample_behavioral_pattern2
            )
            
            assert isinstance(result, HypothesisTest)
            assert result.test_result == mock_test_result
            assert len(statistical_analyzer.test_history) == 1
            
            # Verify mocks were called
            mock_assumptions.assert_called_once()
            mock_t_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_pattern_significance(self, statistical_analyzer, sample_behavioral_pattern1):
        """Test pattern significance analysis."""
        with patch.object(statistical_analyzer, '_one_sample_t_test') as mock_t_test:
            mock_test_result = StatisticalTest(
                test_name="One-sample t-test",
                statistic=3.2,
                p_value=0.003,
                effect_size=1.2,
                confidence_interval=None,
                interpretation="Highly significant",
                assumptions_met=True,
                sample_size=10
            )
            mock_t_test.return_value = mock_test_result
            
            result = await statistical_analyzer.analyze_pattern_significance(
                sample_behavioral_pattern1, baseline_success_rate=0.5
            )
            
            assert isinstance(result, HypothesisTest)
            assert "baseline" in result.null_hypothesis.lower()
            assert result.test_result.p_value < statistical_analyzer.alpha
    
    @pytest.mark.asyncio
    async def test_correlation_analysis(self, statistical_analyzer, sample_epistemic_states):
        """Test correlation analysis."""
        # Create sample outcomes
        outcomes = [0.8 + i * 0.01 for i in range(len(sample_epistemic_states))]
        
        with patch('scipy.stats.pearsonr') as mock_pearson, \
             patch('scipy.stats.spearmanr') as mock_spearman, \
             patch('scipy.stats.shapiro') as mock_shapiro:
            
            # Mock correlation results
            mock_pearson.return_value = (0.7, 0.001)
            mock_spearman.return_value = (0.65, 0.002)
            mock_shapiro.return_value = (0.95, 0.1)  # Normal distribution
            
            correlations = await statistical_analyzer.correlation_analysis(
                sample_epistemic_states, outcomes
            )
            
            assert isinstance(correlations, dict)
            assert len(correlations) > 0
            
            # Check that all features have correlation results
            expected_features = ['confidence_level', 'uncertainty_score', 'num_beliefs', 'num_goals', 'num_facts']
            for feature in expected_features:
                assert feature in correlations
                assert isinstance(correlations[feature], StatisticalTest)
    
    @pytest.mark.asyncio
    async def test_time_series_analysis(self, statistical_analyzer):
        """Test time series analysis."""
        # Create sample time series data
        timestamps = list(range(50))
        values = [10 + i * 0.1 + np.random.normal(0, 0.5) for i in range(50)]
        time_series_data = list(zip(timestamps, values))
        
        with patch.object(statistical_analyzer, '_mann_kendall_trend_test') as mock_trend, \
             patch.object(statistical_analyzer, '_augmented_dickey_fuller_test') as mock_adf, \
             patch.object(statistical_analyzer, '_autocorrelation_analysis') as mock_autocorr:
            
            mock_trend.return_value = StatisticalTest(
                test_name="Mann-Kendall Trend Test",
                statistic=2.5,
                p_value=0.01,
                effect_size=None,
                confidence_interval=None,
                interpretation="Significant trend",
                assumptions_met=True,
                sample_size=50
            )
            
            mock_adf.return_value = StatisticalTest(
                test_name="Augmented Dickey-Fuller Test",
                statistic=-3.5,
                p_value=0.008,
                effect_size=None,
                confidence_interval=None,
                interpretation="Stationary",
                assumptions_met=True,
                sample_size=50
            )
            
            mock_autocorr.return_value = {'lag_1': 0.8, 'lag_2': 0.6}
            
            result = await statistical_analyzer.time_series_analysis(time_series_data)
            
            assert 'trend_test' in result
            assert 'stationarity_test' in result
            assert 'autocorrelation' in result
            assert 'descriptive_stats' in result
    
    @pytest.mark.asyncio
    async def test_granger_causality_analysis(self, statistical_analyzer):
        """Test Granger causality analysis."""
        # Create sample time series
        cause_series = [i + np.random.normal(0, 0.1) for i in range(50)]
        effect_series = [cause_series[i-1] + np.random.normal(0, 0.1) if i > 0 else 0 for i in range(50)]
        
        with patch('statsmodels.tsa.stattools.granger_causality_test') as mock_granger:
            # Mock Granger causality test results
            mock_granger.return_value = {
                1: [{'F-statistic': 5.2, 'P-value': 0.02}],
                2: [{'F-statistic': 3.8, 'P-value': 0.05}]
            }
            
            result = await statistical_analyzer.granger_causality_analysis(
                cause_series, effect_series, max_lags=2
            )
            
            assert 'granger_tests' in result
            assert 'overall_causality' in result
            assert 'best_lag' in result
            assert result['overall_causality'] is True
    
    @pytest.mark.asyncio
    async def test_multiple_comparison_correction(self, statistical_analyzer):
        """Test multiple comparison correction."""
        p_values = [0.01, 0.03, 0.02, 0.08, 0.001]
        
        # Test Bonferroni correction
        result = await statistical_analyzer.multiple_comparison_correction(
            p_values, method='bonferroni'
        )
        
        assert 'original_p_values' in result
        assert 'corrected_p_values' in result
        assert 'significant' in result
        assert 'method' in result
        assert result['method'] == 'bonferroni'
        
        # Corrected p-values should be larger
        assert all(corr >= orig for corr, orig in 
                  zip(result['corrected_p_values'], result['original_p_values']))
    
    @pytest.mark.asyncio
    async def test_power_analysis(self, statistical_analyzer):
        """Test statistical power analysis."""
        with patch('statsmodels.stats.power.ttest_power') as mock_power, \
             patch.object(statistical_analyzer, '_calculate_required_sample_size') as mock_sample_size:
            
            mock_power.return_value = 0.85
            mock_sample_size.return_value = 25
            
            result = await statistical_analyzer.power_analysis(
                effect_size=0.5, sample_size=30, test_type='t_test'
            )
            
            assert 'power' in result
            assert 'effect_size' in result
            assert 'sample_size' in result
            assert 'required_sample_size_80_power' in result
            assert 'interpretation' in result
            assert result['power'] == 0.85
            assert result['effect_size'] == 0.5
            assert result['sample_size'] == 30
    
    @pytest.mark.asyncio
    async def test_check_assumptions(self, statistical_analyzer):
        """Test assumption checking."""
        # Normal data
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2, 25)
        
        with patch('scipy.stats.shapiro') as mock_shapiro, \
             patch('scipy.stats.levene') as mock_levene:
            
            # Mock normality tests (p > 0.05 = normal)
            mock_shapiro.return_value = (0.95, 0.1)
            
            # Mock equal variance test (p > 0.05 = equal variances)
            mock_levene.return_value = (1.2, 0.3)
            
            assumptions = await statistical_analyzer._check_assumptions(group1, group2)
            
            assert 'normality' in assumptions
            assert 'equal_variance' in assumptions
            assert 'sample_sizes' in assumptions
            assert assumptions['normality'] is True
            assert assumptions['equal_variance'] is True
            assert assumptions['sample_sizes'] == (30, 25)
    
    @pytest.mark.asyncio
    async def test_independent_t_test(self, statistical_analyzer):
        """Test independent t-test."""
        group1 = [10, 12, 11, 13, 9, 14, 10, 12]
        group2 = [8, 9, 7, 10, 8, 9, 7, 8]
        
        with patch('scipy.stats.ttest_ind') as mock_ttest:
            mock_ttest.return_value = (3.2, 0.005)
            
            result = await statistical_analyzer._independent_t_test(group1, group2)
            
            assert isinstance(result, StatisticalTest)
            assert result.test_name == "Independent Samples t-test"
            assert result.statistic == 3.2
            assert result.p_value == 0.005
            assert result.effect_size is not None
            assert result.assumptions_met is True
    
    @pytest.mark.asyncio
    async def test_mann_whitney_test(self, statistical_analyzer):
        """Test Mann-Whitney U test."""
        group1 = [10, 12, 11, 13, 9, 14, 10, 12]
        group2 = [8, 9, 7, 10, 8, 9, 7, 8]
        
        with patch('scipy.stats.mannwhitneyu') as mock_mannwhitney:
            mock_mannwhitney.return_value = (15, 0.02)
            
            result = await statistical_analyzer._mann_whitney_test(group1, group2)
            
            assert isinstance(result, StatisticalTest)
            assert result.test_name == "Mann-Whitney U test"
            assert result.statistic == 15
            assert result.p_value == 0.02
            assert result.effect_size is not None
            assert result.assumptions_met is False  # Non-parametric test
    
    def test_extract_epistemic_features(self, statistical_analyzer, sample_epistemic_states):
        """Test epistemic feature extraction."""
        features = statistical_analyzer._extract_epistemic_features(sample_epistemic_states)
        
        assert isinstance(features, dict)
        expected_features = ['confidence_level', 'uncertainty_score', 'num_beliefs', 'num_goals', 'num_facts']
        
        for feature in expected_features:
            assert feature in features
            assert len(features[feature]) == len(sample_epistemic_states)
            assert all(isinstance(val, (int, float)) for val in features[feature])
    
    def test_interpret_test_result(self, statistical_analyzer):
        """Test test result interpretation."""
        # Significant result
        significant_test = StatisticalTest(
            test_name="test",
            statistic=2.5,
            p_value=0.02,
            effect_size=0.8,
            confidence_interval=None,
            interpretation="",
            assumptions_met=True,
            sample_size=30
        )
        
        interpretation = statistical_analyzer._interpret_test_result(significant_test)
        assert "significant" in interpretation.lower()
        assert "reject" in interpretation.lower()
        
        # Non-significant result
        non_significant_test = StatisticalTest(
            test_name="test",
            statistic=1.2,
            p_value=0.08,
            effect_size=0.3,
            confidence_interval=None,
            interpretation="",
            assumptions_met=True,
            sample_size=30
        )
        
        interpretation = statistical_analyzer._interpret_test_result(non_significant_test)
        assert "non-significant" in interpretation.lower()
        assert "fail to reject" in interpretation.lower()
    
    def test_assess_practical_significance(self, statistical_analyzer):
        """Test practical significance assessment."""
        # Large effect size
        large_effect_test = StatisticalTest(
            test_name="test",
            statistic=2.5,
            p_value=0.02,
            effect_size=0.9,
            confidence_interval=None,
            interpretation="",
            assumptions_met=True,
            sample_size=30
        )
        
        assert statistical_analyzer._assess_practical_significance(large_effect_test) is True
        
        # Small effect size
        small_effect_test = StatisticalTest(
            test_name="test",
            statistic=2.5,
            p_value=0.02,
            effect_size=0.2,
            confidence_interval=None,
            interpretation="",
            assumptions_met=True,
            sample_size=30
        )
        
        assert statistical_analyzer._assess_practical_significance(small_effect_test) is False
    
    def test_interpret_correlation(self, statistical_analyzer):
        """Test correlation interpretation."""
        # Strong positive correlation
        interpretation = statistical_analyzer._interpret_correlation(0.8, 0.001)
        assert "strong" in interpretation.lower()
        assert "positive" in interpretation.lower()
        assert "significant" in interpretation.lower()
        
        # Weak negative correlation
        interpretation = statistical_analyzer._interpret_correlation(-0.2, 0.1)
        assert "weak" in interpretation.lower()
        assert "negative" in interpretation.lower()
        assert "non-significant" in interpretation.lower()
    
    def test_correlation_confidence_interval(self, statistical_analyzer):
        """Test correlation confidence interval calculation."""
        r = 0.5
        n = 50
        
        ci = statistical_analyzer._correlation_confidence_interval(r, n)
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < r < ci[1]  # r should be within the interval
        assert -1 <= ci[0] <= 1
        assert -1 <= ci[1] <= 1
    
    def test_holm_correction(self, statistical_analyzer):
        """Test Holm correction for multiple comparisons."""
        p_values = np.array([0.01, 0.03, 0.02, 0.08, 0.001])
        
        corrected = statistical_analyzer._holm_correction(p_values)
        
        assert len(corrected) == len(p_values)
        assert all(corrected >= p_values)  # Corrected should be >= original
        assert all(corrected <= 1.0)  # Should not exceed 1
    
    def test_benjamini_hochberg_correction(self, statistical_analyzer):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = np.array([0.01, 0.03, 0.02, 0.08, 0.001])
        
        corrected = statistical_analyzer._benjamini_hochberg_correction(p_values)
        
        assert len(corrected) == len(p_values)
        assert all(corrected >= p_values)  # Generally corrected should be >= original
        assert all(corrected <= 1.0)  # Should not exceed 1


@pytest.mark.integration
class TestStatisticalAnalysisIntegration:
    """Integration tests for statistical analysis."""
    
    @pytest.mark.asyncio
    async def test_complete_pattern_comparison_workflow(self, sample_behavioral_pattern1, 
                                                       sample_behavioral_pattern2):
        """Test complete pattern comparison workflow."""
        analyzer = StatisticalAnalyzer(alpha=0.05)
        
        # This should work with real statistical tests
        result = await analyzer.compare_behavioral_patterns(
            sample_behavioral_pattern1, sample_behavioral_pattern2
        )
        
        assert isinstance(result, HypothesisTest)
        assert result.test_result.p_value >= 0.0
        assert result.test_result.p_value <= 1.0
        assert len(analyzer.test_history) == 1
    
    @pytest.mark.asyncio
    async def test_correlation_analysis_workflow(self, sample_epistemic_states):
        """Test correlation analysis workflow."""
        analyzer = StatisticalAnalyzer()
        
        # Create outcomes correlated with confidence
        outcomes = [state.confidence_level + np.random.normal(0, 0.1) 
                   for state in sample_epistemic_states]
        
        correlations = await analyzer.correlation_analysis(sample_epistemic_states, outcomes)
        
        assert isinstance(correlations, dict)
        assert 'confidence_level' in correlations
        
        # Should find positive correlation with confidence_level
        conf_corr = correlations['confidence_level']
        assert conf_corr.statistic > 0  # Positive correlation expected
    
    @pytest.mark.asyncio
    async def test_multiple_testing_workflow(self):
        """Test multiple testing correction workflow."""
        analyzer = StatisticalAnalyzer()
        
        # Simulate multiple p-values from different tests
        p_values = [0.001, 0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2]
        
        # Test different correction methods
        bonferroni_result = await analyzer.multiple_comparison_correction(
            p_values, method='bonferroni'
        )
        
        holm_result = await analyzer.multiple_comparison_correction(
            p_values, method='holm'
        )
        
        fdr_result = await analyzer.multiple_comparison_correction(
            p_values, method='fdr_bh'
        )
        
        # Bonferroni should be most conservative
        assert sum(bonferroni_result['significant']) <= sum(holm_result['significant'])
        assert sum(holm_result['significant']) <= sum(fdr_result['significant'])
    
    @pytest.mark.asyncio
    async def test_statistical_power_workflow(self):
        """Test statistical power analysis workflow."""
        analyzer = StatisticalAnalyzer()
        
        # Test power for different effect sizes
        effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
        sample_size = 30
        
        power_results = []
        for effect_size in effect_sizes:
            result = await analyzer.power_analysis(
                effect_size=effect_size,
                sample_size=sample_size,
                test_type='t_test'
            )
            power_results.append(result['power'])
        
        # Power should increase with effect size
        assert power_results[0] < power_results[1] < power_results[2]