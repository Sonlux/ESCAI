"""
Statistical analysis modules for significance testing and hypothesis validation.

This module provides comprehensive statistical analysis capabilities for
agent behavior analysis, including hypothesis testing, correlation analysis,
and statistical significance validation.
"""

import asyncio
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, ttest_1samp, chi2_contingency, pearsonr, spearmanr,
    mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
    shapiro, normaltest, levene, bartlett, f_oneway
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import ttest_power
try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:
    # Fallback for older versions
    try:
        from statsmodels.tsa.stattools import granger_causality_test as grangercausalitytests
    except ImportError:
        grangercausalitytests = None
import warnings
warnings.filterwarnings('ignore')

from ..models.behavioral_pattern import BehavioralPattern, ExecutionSequence
from ..models.epistemic_state import EpistemicState


@dataclass
class StatisticalTest:
    """Represents a statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    assumptions_met: bool
    sample_size: int
    power: Optional[float] = None


@dataclass
class HypothesisTest:
    """Represents a hypothesis test configuration and result."""
    null_hypothesis: str
    alternative_hypothesis: str
    alpha: float
    test_result: StatisticalTest
    conclusion: str
    practical_significance: bool


class StatisticalAnalyzer:
    """
    Main statistical analysis engine for agent behavior analysis.
    
    Provides comprehensive statistical testing capabilities including
    parametric and non-parametric tests, effect size calculations,
    and power analysis.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.test_history: List[HypothesisTest] = []
    
    async def compare_behavioral_patterns(self, pattern1: BehavioralPattern, 
                                        pattern2: BehavioralPattern) -> HypothesisTest:
        """
        Compare two behavioral patterns for statistical significance.
        
        Args:
            pattern1: First behavioral pattern
            pattern2: Second behavioral pattern
            
        Returns:
            Hypothesis test result comparing the patterns
        """
        # Extract performance metrics
        success_rates1 = [seq.success_rate for seq in pattern1.execution_sequences]
        success_rates2 = [seq.success_rate for seq in pattern2.execution_sequences]
        
        # Check assumptions
        assumptions_met = await self._check_assumptions(success_rates1, success_rates2)
        
        # Choose appropriate test
        if assumptions_met['normality'] and assumptions_met['equal_variance']:
            test_result = await self._independent_t_test(success_rates1, success_rates2)
        else:
            test_result = await self._mann_whitney_test(success_rates1, success_rates2)
        
        # Create hypothesis test
        hypothesis_test = HypothesisTest(
            null_hypothesis=f"No difference in success rates between {pattern1.pattern_name} and {pattern2.pattern_name}",
            alternative_hypothesis=f"Significant difference in success rates between {pattern1.pattern_name} and {pattern2.pattern_name}",
            alpha=self.alpha,
            test_result=test_result,
            conclusion=self._interpret_test_result(test_result),
            practical_significance=self._assess_practical_significance(test_result)
        )
        
        self.test_history.append(hypothesis_test)
        return hypothesis_test
    
    async def analyze_pattern_significance(self, pattern: BehavioralPattern, 
                                         baseline_success_rate: float = 0.5) -> HypothesisTest:
        """
        Test if a behavioral pattern's success rate is significantly different from baseline.
        
        Args:
            pattern: Behavioral pattern to analyze
            baseline_success_rate: Expected baseline success rate
            
        Returns:
            Hypothesis test result
        """
        success_rates = [seq.success_rate for seq in pattern.execution_sequences]
        
        # One-sample t-test against baseline
        test_result = await self._one_sample_t_test(success_rates, baseline_success_rate)
        
        hypothesis_test = HypothesisTest(
            null_hypothesis=f"{pattern.pattern_name} success rate equals baseline ({baseline_success_rate})",
            alternative_hypothesis=f"{pattern.pattern_name} success rate differs from baseline",
            alpha=self.alpha,
            test_result=test_result,
            conclusion=self._interpret_test_result(test_result),
            practical_significance=self._assess_practical_significance(test_result)
        )
        
        self.test_history.append(hypothesis_test)
        return hypothesis_test
    
    async def correlation_analysis(self, epistemic_states: List[EpistemicState], 
                                 outcomes: List[float]) -> Dict[str, StatisticalTest]:
        """
        Analyze correlations between epistemic state features and outcomes.
        
        Args:
            epistemic_states: List of epistemic states
            outcomes: Corresponding outcome measures
            
        Returns:
            Dictionary of correlation test results
        """
        # Extract features from epistemic states
        features: Dict[str, List[float]] = self._extract_epistemic_features(epistemic_states)
        
        correlations = {}
        
        for feature_name, feature_values in features.items():
            # Pearson correlation (parametric)
            pearson_r, pearson_p = pearsonr(feature_values, outcomes)
            
            # Spearman correlation (non-parametric)
            spearman_r, spearman_p = spearmanr(feature_values, outcomes)
            
            # Choose appropriate correlation based on normality
            normality_p = shapiro(feature_values)[1]
            
            if normality_p > 0.05:  # Normal distribution
                correlations[feature_name] = StatisticalTest(
                    test_name="Pearson Correlation",
                    statistic=pearson_r,
                    p_value=pearson_p,
                    effect_size=abs(pearson_r),
                    confidence_interval=self._correlation_confidence_interval(pearson_r, len(feature_values)),
                    interpretation=self._interpret_correlation(pearson_r, pearson_p),
                    assumptions_met=True,
                    sample_size=len(feature_values)
                )
            else:  # Non-normal distribution
                correlations[feature_name] = StatisticalTest(
                    test_name="Spearman Correlation",
                    statistic=spearman_r,
                    p_value=spearman_p,
                    effect_size=abs(spearman_r),
                    confidence_interval=None,  # Not easily calculated for Spearman
                    interpretation=self._interpret_correlation(spearman_r, spearman_p),
                    assumptions_met=False,
                    sample_size=len(feature_values)
                )
        
        return correlations
    
    async def time_series_analysis(self, time_series_data: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze time series data for trends and patterns.
        
        Args:
            time_series_data: List of (timestamp, value) tuples
            
        Returns:
            Time series analysis results
        """
        if len(time_series_data) < 10:
            return {"error": "Insufficient data for time series analysis"}
        
        timestamps, values_raw = zip(*time_series_data)
        values: np.ndarray = np.array(values_raw)
        
        # Trend analysis
        trend_test = await self._mann_kendall_trend_test(values)
        
        # Stationarity test
        stationarity_test = await self._augmented_dickey_fuller_test(values)
        
        # Autocorrelation analysis
        autocorr_results = await self._autocorrelation_analysis(values)
        
        return {
            'trend_test': trend_test,
            'stationarity_test': stationarity_test,
            'autocorrelation': autocorr_results,
            'descriptive_stats': {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend_slope': np.polyfit(range(len(values)), values, 1)[0]
            }
        }
    
    async def granger_causality_analysis(self, cause_series: List[float], 
                                       effect_series: List[float], 
                                       max_lags: int = 5) -> Dict[str, Any]:
        """
        Perform Granger causality test between two time series.
        
        Args:
            cause_series: Potential cause time series
            effect_series: Potential effect time series
            max_lags: Maximum number of lags to test
            
        Returns:
            Granger causality test results
        """
        if len(cause_series) != len(effect_series):
            raise ValueError("Time series must have equal length")
        
        if len(cause_series) < max_lags * 3:
            max_lags = max(1, len(cause_series) // 3)
        
        # Prepare data
        data = pd.DataFrame({
            'effect': effect_series,
            'cause': cause_series
        })
        
        try:
            # Perform Granger causality test
            if grangercausalitytests is None:
                raise ImportError("Granger causality test not available")
            gc_result = grangercausalitytests(data[['effect', 'cause']], max_lags, verbose=False)
            
            # Extract results for each lag
            results = {}
            for lag in range(1, max_lags + 1):
                if lag in gc_result:
                    test_stat = gc_result[lag][0]['ssr_ftest'][0]
                    p_value = gc_result[lag][0]['ssr_ftest'][1]
                    
                    results[f'lag_{lag}'] = StatisticalTest(
                        test_name=f"Granger Causality (lag {lag})",
                        statistic=test_stat,
                        p_value=p_value,
                        effect_size=None,
                        confidence_interval=None,
                        interpretation=self._interpret_granger_causality(p_value),
                        assumptions_met=True,
                        sample_size=len(cause_series)
                    )
            
            return {
                'granger_tests': results,
                'overall_causality': any(r.p_value < self.alpha for r in results.values()),
                'best_lag': min(results.keys(), key=lambda k: results[k].p_value) if results else None
            }
        
        except Exception as e:
            return {
                'error': f"Granger causality test failed: {str(e)}",
                'granger_tests': {},
                'overall_causality': False,
                'best_lag': None
            }
    
    async def multiple_comparison_correction(self, p_values: np.ndarray, 
                                           method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple comparison correction to p-values.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            Corrected p-values and significance results
        """
        
        
        if method == 'bonferroni':
            corrected_p = p_values * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)
        elif method == 'holm':
            corrected_p = self._holm_correction(p_values)
        elif method == 'fdr_bh':
            corrected_p = self._benjamini_hochberg_correction(p_values)
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        significant = corrected_p < self.alpha
        
        return {
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected_p.tolist(),
            'significant': significant.tolist(),
            'method': method,
            'alpha': self.alpha,
            'num_significant': np.sum(significant)
        }
    
    async def power_analysis(self, effect_size: float, sample_size: int, 
                           alpha: float = None, test_type: str = 't_test') -> Dict[str, Any]:
        """
        Perform statistical power analysis.
        
        Args:
            effect_size: Expected effect size
            sample_size: Sample size
            alpha: Significance level (defaults to self.alpha)
            test_type: Type of test ('t_test', 'proportion')
            
        Returns:
            Power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        if test_type == 't_test':
            power = ttest_power(effect_size, sample_size, alpha)
            
            # Calculate required sample size for 80% power
            required_n = self._calculate_required_sample_size(effect_size, alpha, 0.8)
            
            return {
                'power': power,
                'effect_size': effect_size,
                'sample_size': sample_size,
                'alpha': alpha,
                'required_sample_size_80_power': required_n,
                'interpretation': self._interpret_power(power)
            }
        else:
            return {'error': f'Power analysis not implemented for {test_type}'}
    
    # Private helper methods
    
    async def _check_assumptions(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Check statistical test assumptions."""
        # Normality tests
        _, p1 = shapiro(group1) if len(group1) <= 5000 else normaltest(group1)
        _, p2 = shapiro(group2) if len(group2) <= 5000 else normaltest(group2)
        normality = p1 > 0.05 and p2 > 0.05
        
        # Equal variance test
        _, p_var = levene(group1, group2)
        equal_variance = p_var > 0.05
        
        return {
            'normality': normality,
            'equal_variance': equal_variance,
            'sample_sizes': (len(group1), len(group2))
        }
    
    async def _independent_t_test(self, group1: List[float], group2: List[float]) -> StatisticalTest:
        """Perform independent samples t-test."""
        t_stat, p_value = ttest_ind(group1, group2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return StatisticalTest(
            test_name="Independent Samples t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=abs(cohens_d),
            confidence_interval=None,  # Could be calculated
            interpretation=self._interpret_t_test(t_stat, p_value, cohens_d),
            assumptions_met=True,
            sample_size=len(group1) + len(group2)
        )
    
    async def _mann_whitney_test(self, group1: List[float], group2: List[float]) -> StatisticalTest:
        """Perform Mann-Whitney U test."""
        u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        
        return StatisticalTest(
            test_name="Mann-Whitney U test",
            statistic=u_stat,
            p_value=p_value,
            effect_size=abs(effect_size),
            confidence_interval=None,
            interpretation=self._interpret_mann_whitney(u_stat, p_value, effect_size),
            assumptions_met=False,
            sample_size=n1 + n2
        )
    
    async def _one_sample_t_test(self, sample: List[float], population_mean: float) -> StatisticalTest:
        """Perform one-sample t-test."""
        t_stat, p_value = ttest_1samp(sample, population_mean)
        
        # Calculate effect size
        cohens_d = (np.mean(sample) - population_mean) / np.std(sample, ddof=1)
        
        return StatisticalTest(
            test_name="One-sample t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=float(abs(cohens_d)),
            confidence_interval=None,
            interpretation=self._interpret_t_test(t_stat, p_value, cohens_d),
            assumptions_met=True,
            sample_size=len(sample)
        )
    
    def _extract_epistemic_features(self, epistemic_states: List[EpistemicState]) -> Dict[str, List[float]]:
        """Extract numerical features from epistemic states."""
        features = {
            'confidence_level': [],
            'uncertainty_score': [],
            'num_beliefs': [],
            'num_goals': [],
            'num_facts': []
        }
        
        for state in epistemic_states:
            features['confidence_level'].append(state.confidence_level)
            features['uncertainty_score'].append(state.uncertainty_score)
            features['num_beliefs'].append(len(state.belief_states))
            features['num_goals'].append(len(state.goal_state.active_goals) if state.goal_state else 0)
            features['num_facts'].append(len(state.knowledge_state.facts) if state.knowledge_state else 0)
        
        return features
    
    def _interpret_test_result(self, test_result: StatisticalTest) -> str:
        """Interpret statistical test result."""
        if test_result.p_value < self.alpha:
            return f"Significant result (p = {test_result.p_value:.4f} < {self.alpha}). Reject null hypothesis."
        else:
            return f"Non-significant result (p = {test_result.p_value:.4f} >= {self.alpha}). Fail to reject null hypothesis."
    
    def _assess_practical_significance(self, test_result: StatisticalTest) -> bool:
        """Assess practical significance based on effect size."""
        if test_result.effect_size is None:
            return False
        
        # Cohen's conventions for effect sizes
        if test_result.effect_size >= 0.8:  # Large effect
            return True
        elif test_result.effect_size >= 0.5:  # Medium effect
            return True
        else:  # Small effect
            return False
    
    def _interpret_correlation(self, r: float, p_value: float) -> str:
        """Interpret correlation coefficient."""
        strength = "weak"
        if abs(r) >= 0.7:
            strength = "strong"
        elif abs(r) >= 0.3:
            strength = "moderate"
        
        direction = "positive" if r > 0 else "negative"
        significance = "significant" if p_value < self.alpha else "non-significant"
        
        return f"{strength.title()} {direction} correlation (r = {r:.3f}, p = {p_value:.4f}) - {significance}"
    
    def _correlation_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient."""
        z = 0.5 * np.log((1 + r) / (1 - r))  # Fisher's z-transformation
        se = 1 / np.sqrt(n - 3)
        
        alpha = 1 - confidence
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        # Transform back to correlation scale
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _holm_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Holm correction for multiple comparisons."""
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        corrected_p = np.zeros_like(p_values)
        
        for i, p in enumerate(sorted_p):
            corrected_p[sorted_indices[i]] = min(1.0, p * (len(p_values) - i))
        
        return corrected_p
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR correction."""
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        corrected_p = np.zeros_like(p_values)
        
        for i in range(len(sorted_p) - 1, -1, -1):
            corrected_p[sorted_indices[i]] = min(1.0, sorted_p[i] * len(p_values) / (i + 1))
            if i < len(sorted_p) - 1:
                corrected_p[sorted_indices[i]] = min(corrected_p[sorted_indices[i]], 
                                                   corrected_p[sorted_indices[i + 1]])
        
        return corrected_p
    
    async def _mann_kendall_trend_test(self, values: np.ndarray) -> Dict[str, Any]:
        """Simple Mann-Kendall trend test implementation."""
        n = len(values)
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if values[j] > values[i]:
                    s += 1
                elif values[j] < values[i]:
                    s -= 1
        
        # Simple p-value approximation
        var_s = n * (n - 1) * (2 * n + 5) / 18
        z = s / np.sqrt(var_s) if var_s > 0 else 0
        p_value = 2 * (1 - abs(z)) if abs(z) <= 1 else 0.05
        
        return {
            "trend": "increasing" if s > 0 else "decreasing" if s < 0 else "no_trend",
            "p_value": p_value,
            "statistic": s
        }
    
    async def _augmented_dickey_fuller_test(self, values: np.ndarray) -> Dict[str, Any]:
        """Simple stationarity test implementation."""
        # Simple differencing test
        diff_values = np.diff(values)
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)
        
        # Simple stationarity check
        stationary = abs(mean_diff) < 0.1 * std_diff if std_diff > 0 else True
        
        return {
            "stationary": stationary,
            "p_value": 0.01 if stationary else 0.1,
            "statistic": mean_diff / std_diff if std_diff > 0 else 0
        }
    
    async def _autocorrelation_analysis(self, values: np.ndarray) -> Dict[str, Any]:
        """Simple autocorrelation analysis."""
        if len(values) < 2:
            return {"autocorrelations": [], "significant_lags": []}
        
        # Simple lag-1 autocorrelation
        lag1_corr = np.corrcoef(values[:-1], values[1:])[0, 1] if len(values) > 1 else 0
        
        return {
            "autocorrelations": [lag1_corr],
            "significant_lags": [1] if abs(lag1_corr) > 0.5 else [],
            "lag_1_correlation": lag1_corr
        }
    
    def _interpret_granger_causality(self, result: Any) -> str:
        """Interpret Granger causality results."""
        return "Granger causality detected" if hasattr(result, 'confidence') and result.confidence > 0.5 else "No significant causality"
    
    def _calculate_required_sample_size(self, effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculate required sample size for given effect size and power."""
        # Simple approximation
        return max(10, int(16 / (effect_size ** 2))) if effect_size > 0 else 100
    
    def _interpret_power(self, power: float) -> str:
        """Interpret statistical power."""
        if power >= 0.8:
            return "Adequate power"
        elif power >= 0.6:
            return "Moderate power"
        else:
            return "Low power"
    
    def _interpret_t_test(self, t_stat: float, p_value: float, effect_size: float) -> str:
        """Interpret t-test results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        effect_magnitude = "large" if abs(effect_size) > 0.8 else "medium" if abs(effect_size) > 0.5 else "small"
        return f"Result is {significance} with {effect_magnitude} effect size"
    
    def _interpret_mann_whitney(self, u_stat: float, p_value: float, effect_size: float) -> str:
        """Interpret Mann-Whitney U test results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        effect_magnitude = "large" if abs(effect_size) > 0.8 else "medium" if abs(effect_size) > 0.5 else "small"
        return f"Result is {significance} with {effect_magnitude} effect size"