"""
Statistical report generation with proper academic formatting.

Generates comprehensive statistical reports suitable for academic publication,
including methodology descriptions, results presentation, and interpretation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json

from .citation_manager import CitationDatabase, MethodologyCitationGenerator
from .latex_templates import LatexTableGenerator


@dataclass
class StatisticalTest:
    """Represents a statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: str = ""
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < alpha
    
    def to_latex_row(self) -> str:
        """Format as LaTeX table row."""
        ci_str = ""
        if self.confidence_interval:
            ci_str = f"[{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]"
        
        significance = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        
        return f"{self.test_name} & {self.statistic:.3f} & {self.p_value:.3f}{significance} & {ci_str} \\\\"


@dataclass
class DescriptiveStatistics:
    """Comprehensive descriptive statistics."""
    variable_name: str
    n: int
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    q1: float
    q3: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    def to_latex_row(self) -> str:
        """Format as LaTeX table row."""
        return f"{self.variable_name} & {self.n} & {self.mean:.3f} & {self.std:.3f} & {self.median:.3f} & [{self.q1:.3f}, {self.q3:.3f}] & [{self.min_val:.3f}, {self.max_val:.3f}] \\\\"


class StatisticalAnalyzer:
    """Performs statistical analysis on ESCAI monitoring data."""
    
    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}
    
    def analyze_agent_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze agent performance metrics."""
        results = {}
        
        # Descriptive statistics
        if 'success_rate' in data.columns:
            results['success_rate_stats'] = self._calculate_descriptive_stats(  # type: ignore[assignment]
                data['success_rate'], 'Success Rate'  # type: ignore[arg-type]
            )
        
        if 'completion_time' in data.columns:
            results['completion_time_stats'] = self._calculate_descriptive_stats(  # type: ignore[assignment]
                data['completion_time'], 'Completion Time (s)'  # type: ignore[arg-type]
            )
        
        if 'epistemic_uncertainty' in data.columns:
            results['uncertainty_stats'] = self._calculate_descriptive_stats(  # type: ignore[assignment]
                data['epistemic_uncertainty'], 'Epistemic Uncertainty'  # type: ignore[arg-type]
            )
        
        # Correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns  # type: ignore[assignment]
        if len(numeric_cols) > 1:
            results['correlation_matrix'] = data[numeric_cols].corr()  # type: ignore[assignment,index]
        
        # Performance comparisons
        if 'agent_type' in data.columns and len(data['agent_type'].unique()) > 1:  # type: ignore[arg-type]
            results['agent_comparison'] = self._compare_agent_types(data)  # type: ignore[assignment]
        
        return results
    
    def analyze_epistemic_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze epistemic state patterns."""
        results = {}
        
        # Belief consistency analysis
        if 'belief_consistency' in data.columns:
            results['belief_consistency'] = self._analyze_belief_consistency(data)  # type: ignore[assignment]
        
        # Goal achievement patterns
        if 'goal_achieved' in data.columns:
            results['goal_achievement'] = self._analyze_goal_achievement(data)  # type: ignore[assignment]
        
        # Knowledge evolution
        if 'knowledge_growth' in data.columns:
            results['knowledge_evolution'] = self._analyze_knowledge_evolution(data)  # type: ignore[assignment]
        
        return results
    
    def analyze_causal_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze causal relationships in agent behavior."""
        results = {}
        
        # Simple causal analysis (correlation-based)
        if len(data.select_dtypes(include=[np.number]).columns) > 2:  # type: ignore[arg-type]
            results['causal_correlations'] = self._identify_causal_correlations(data)  # type: ignore[assignment]
        
        # Intervention analysis
        if 'intervention' in data.columns:
            results['intervention_effects'] = self._analyze_interventions(data)  # type: ignore[assignment]
        
        return results
    
    def _calculate_descriptive_stats(self, series: pd.Series, name: str) -> DescriptiveStatistics:
        """Calculate comprehensive descriptive statistics."""
        return DescriptiveStatistics(
            variable_name=name,
            n=len(series.dropna()),  # type: ignore[arg-type]
            mean=series.mean(),  # type: ignore[arg-type]
            std=series.std(),  # type: ignore[arg-type]
            min_val=series.min(),  # type: ignore[arg-type]
            max_val=series.max(),  # type: ignore[arg-type]
            median=series.median(),  # type: ignore[arg-type]
            q1=series.quantile(0.25),  # type: ignore[arg-type]
            q3=series.quantile(0.75),  # type: ignore[arg-type]
            skewness=series.skew(),  # type: ignore[arg-type]
            kurtosis=series.kurtosis()  # type: ignore[arg-type]
        )
    
    def _compare_agent_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance across agent types."""
        results = {}
        
        agent_types = data['agent_type'].unique()  # type: ignore[assignment]
        
        # Performance metrics by agent type
        if 'success_rate' in data.columns:
            success_by_type = data.groupby('agent_type')['success_rate'].agg([  # type: ignore[index]
                'count', 'mean', 'std', 'median'
            ]).round(3)
            results['success_rate_by_type'] = success_by_type  # type: ignore[assignment]
            
            # Statistical test for differences
            if len(agent_types) == 2:
                from scipy import stats
                group1 = data[data['agent_type'] == agent_types[0]]['success_rate']  # type: ignore[index]
                group2 = data[data['agent_type'] == agent_types[1]]['success_rate']  # type: ignore[index]
                
                t_stat, p_val = stats.ttest_ind(group1.dropna(), group2.dropna())  # type: ignore[arg-type]
                
                results['success_rate_ttest'] = StatisticalTest(  # type: ignore[assignment]
                    test_name="Two-sample t-test (Success Rate)",
                    statistic=t_stat,
                    p_value=p_val,
                    interpretation=f"Comparing success rates between {agent_types[0]} and {agent_types[1]}"
                )
        
        return results
    
    def _analyze_belief_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze belief consistency patterns."""
        results = {}
        
        consistency = data['belief_consistency']  # type: ignore[assignment]
        
        # Descriptive statistics
        results['descriptive_stats'] = self._calculate_descriptive_stats(  # type: ignore[assignment]
            consistency, 'Belief Consistency'  # type: ignore[arg-type]
        )
        
        # Consistency over time
        if 'timestamp' in data.columns:
            data_sorted = data.sort_values('timestamp')  # type: ignore[assignment]
            results['consistency_trend'] = {  # type: ignore[assignment]
                'correlation_with_time': consistency.corr(  # type: ignore[attr-defined]
                    pd.to_numeric(data_sorted['timestamp'])  # type: ignore[arg-type]
                ),
                'trend_analysis': 'Increasing' if consistency.corr(  # type: ignore[attr-defined]
                    pd.to_numeric(data_sorted['timestamp'])  # type: ignore[arg-type]
                ) > 0.1 else 'Decreasing' if consistency.corr(  # type: ignore[attr-defined]
                    pd.to_numeric(data_sorted['timestamp'])  # type: ignore[arg-type]
                ) < -0.1 else 'Stable'
            }
        
        return results
    
    def _analyze_goal_achievement(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze goal achievement patterns."""
        results = {}
        
        if data['goal_achieved'].dtype == bool:  # type: ignore[index]
            achievement_rate = data['goal_achieved'].mean()  # type: ignore[index,assignment]
            results['overall_achievement_rate'] = achievement_rate  # type: ignore[assignment]
            
            # Confidence interval for achievement rate
            n = len(data)
            se = np.sqrt(achievement_rate * (1 - achievement_rate) / n)  # type: ignore[arg-type]
            ci_lower = achievement_rate - 1.96 * se  # type: ignore[operator]
            ci_upper = achievement_rate + 1.96 * se  # type: ignore[operator]
            
            results['achievement_rate_ci'] = (ci_lower, ci_upper)  # type: ignore[assignment]
        
        return results
    
    def _analyze_knowledge_evolution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze knowledge evolution patterns."""
        results = {}
        
        knowledge_growth = data['knowledge_growth']  # type: ignore[assignment]
        
        # Growth rate analysis
        results['average_growth_rate'] = knowledge_growth.mean()  # type: ignore[assignment,attr-defined]
        results['growth_variability'] = knowledge_growth.std()  # type: ignore[assignment,attr-defined]
        
        # Growth phases
        positive_growth = (knowledge_growth > 0).sum()  # type: ignore[assignment,operator,attr-defined]
        negative_growth = (knowledge_growth < 0).sum()  # type: ignore[assignment,operator,attr-defined]
        no_growth = (knowledge_growth == 0).sum()  # type: ignore[assignment,operator,attr-defined]
        
        results['growth_phases'] = {  # type: ignore[assignment]
            'positive_growth_episodes': positive_growth,
            'negative_growth_episodes': negative_growth,
            'stable_episodes': no_growth,
            'growth_ratio': positive_growth / len(knowledge_growth)  # type: ignore[arg-type]
        }
        
        return results
    
    def _identify_causal_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify potential causal relationships through correlation analysis."""
        numeric_data = data.select_dtypes(include=[np.number])  # type: ignore[assignment]
        correlation_matrix = numeric_data.corr()  # type: ignore[attr-defined,assignment]
        
        # Find strong correlations (|r| > 0.7)
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):  # type: ignore[arg-type]
            for j in range(i + 1, len(correlation_matrix.columns)):  # type: ignore[arg-type]
                corr_val = correlation_matrix.iloc[i, j]  # type: ignore[assignment,index]
                if abs(float(corr_val)) > 0.7:  # type: ignore[arg-type,operator]
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],  # type: ignore[index]
                        'variable2': correlation_matrix.columns[j],  # type: ignore[index]
                        'correlation': corr_val,
                        'strength': 'Strong' if abs(float(corr_val)) > 0.8 else 'Moderate'  # type: ignore[arg-type,operator]
                    })
        
        return {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': strong_correlations
        }
    
    def _analyze_interventions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze effects of interventions."""
        results = {}
        
        # Before/after analysis
        if 'outcome' in data.columns:
            intervention_groups = data.groupby('intervention')['outcome'].agg([  # type: ignore[index,assignment]
                'count', 'mean', 'std'
            ])
            results['intervention_effects'] = intervention_groups  # type: ignore[assignment]
            
            # Effect size calculation
            if len(data['intervention'].unique()) == 2:  # type: ignore[arg-type]
                control_group = data[data['intervention'] == False]['outcome']  # type: ignore[index,comparison-overlap]
                treatment_group = data[data['intervention'] == True]['outcome']  # type: ignore[index,comparison-overlap]
                
                # Cohen's d
                variance_sum = float(  # type: ignore[arg-type]
                    (len(control_group) - 1) * control_group.var() +  # type: ignore[arg-type,attr-defined,operator]
                    (len(treatment_group) - 1) * treatment_group.var()  # type: ignore[arg-type,attr-defined,operator]
                )
                n_total = float(len(control_group) + len(treatment_group) - 2)  # type: ignore[arg-type,operator]
                pooled_std = np.sqrt(variance_sum / n_total)  # type: ignore[arg-type]
                
                cohens_d = (treatment_group.mean() - control_group.mean()) / pooled_std  # type: ignore[assignment,attr-defined,operator]
                
                results['effect_size'] = {  # type: ignore[assignment]
                    'cohens_d': cohens_d,
                    'interpretation': self._interpret_effect_size(float(cohens_d))  # type: ignore[arg-type]
                }
        
        return results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"


class StatisticalReportGenerator:
    """Generates comprehensive statistical reports for academic publication."""
    
    def __init__(self):
        self.citation_db = CitationDatabase()
        self.methodology_generator = MethodologyCitationGenerator(self.citation_db)
        self.analyzer = StatisticalAnalyzer()
    
    def generate_full_report(self, 
                           data: Dict[str, pd.DataFrame],
                           title: str = "ESCAI Framework Analysis Report",
                           methodologies: List[str] = None) -> str:
        """Generate complete statistical report."""
        if methodologies is None:
            methodologies = ["epistemic_extraction", "statistical_analysis"]
        
        report = self._generate_introduction()
        report += self.methodology_generator.generate_methodology_section(methodologies)
        report += self._generate_results_section(data)
        report += self._generate_discussion_section()
        report += self._generate_conclusion_section()
        
        return report
    
    def _generate_introduction(self) -> str:
        """Generate introduction section."""
        section = "\\section{Introduction}\n\n"
        
        section += """Autonomous agent systems require comprehensive monitoring and analysis to understand their behavior, performance, and decision-making processes. The ESCAI (Epistemic State and Causal Analysis Intelligence) Framework provides real-time monitoring capabilities that capture epistemic states, behavioral patterns, and causal relationships in agent execution.\n\n"""
        
        section += """This report presents a statistical analysis of agent monitoring data collected using the ESCAI Framework. The analysis focuses on performance metrics, epistemic state evolution, and causal relationships that influence agent behavior and task outcomes.\n\n"""
        
        return section
    
    def _generate_results_section(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate comprehensive results section."""
        section = "\\section{Results}\n\n"
        
        # Agent performance analysis
        if 'agent_performance' in data:
            section += "\\subsection{Agent Performance Analysis}\n\n"
            perf_results = self.analyzer.analyze_agent_performance(data['agent_performance'])  # type: ignore[arg-type]
            section += self._format_performance_results(perf_results)
        
        # Epistemic pattern analysis
        if 'epistemic_data' in data:
            section += "\\subsection{Epistemic State Analysis}\n\n"
            epistemic_results = self.analyzer.analyze_epistemic_patterns(data['epistemic_data'])  # type: ignore[arg-type]
            section += self._format_epistemic_results(epistemic_results)
        
        # Causal analysis
        if 'causal_data' in data:
            section += "\\subsection{Causal Relationship Analysis}\n\n"
            causal_results = self.analyzer.analyze_causal_relationships(data['causal_data'])  # type: ignore[arg-type]
            section += self._format_causal_results(causal_results)
        
        return section
    
    def _format_performance_results(self, results: Dict[str, Any]) -> str:
        """Format performance analysis results."""
        section = ""
        
        # Descriptive statistics table
        if any(key.endswith('_stats') for key in results.keys()):
            section += "Table \\ref{tab:descriptive_stats} presents descriptive statistics for key performance metrics.\n\n"
            
            table_data = {}
            for key, stats in results.items():
                if key.endswith('_stats') and isinstance(stats, DescriptiveStatistics):
                    table_data[stats.variable_name] = {
                        'N': stats.n,
                        'Mean': f"{stats.mean:.3f}",
                        'SD': f"{stats.std:.3f}",
                        'Median': f"{stats.median:.3f}",
                        'Range': f"[{stats.min_val:.3f}, {stats.max_val:.3f}]"
                    }
            
            if table_data:
                section += LatexTableGenerator.generate_comparison_table(
                    table_data,
                    "Descriptive Statistics for Performance Metrics",
                    "tab:descriptive_stats"
                )
        
        # Agent type comparisons
        if 'agent_comparison' in results:
            section += "\n\\subsubsection{Agent Type Comparison}\n\n"
            comparison = results['agent_comparison']
            
            if 'success_rate_ttest' in comparison:
                test = comparison['success_rate_ttest']
                significance = "significant" if test.is_significant() else "not significant"
                section += f"A two-sample t-test revealed a {significance} difference in success rates between agent types (t = {test.statistic:.3f}, p = {test.p_value:.3f}).\n\n"
        
        return section
    
    def _format_epistemic_results(self, results: Dict[str, Any]) -> str:
        """Format epistemic analysis results."""
        section = ""
        
        if 'belief_consistency' in results:
            consistency = results['belief_consistency']
            
            if 'descriptive_stats' in consistency:
                stats = consistency['descriptive_stats']
                section += f"Belief consistency showed a mean of {stats.mean:.3f} (SD = {stats.std:.3f}, range = [{stats.min_val:.3f}, {stats.max_val:.3f}]).\n\n"
            
            if 'consistency_trend' in consistency:
                trend = consistency['consistency_trend']['trend_analysis']
                section += f"Over time, belief consistency showed a {trend.lower()} pattern.\n\n"
        
        if 'goal_achievement' in results:
            achievement = results['goal_achievement']
            
            if 'overall_achievement_rate' in achievement:
                rate = achievement['overall_achievement_rate']
                ci = achievement.get('achievement_rate_ci', (0, 0))
                section += f"The overall goal achievement rate was {rate:.3f} (95\\% CI: [{ci[0]:.3f}, {ci[1]:.3f}]).\n\n"
        
        return section
    
    def _format_causal_results(self, results: Dict[str, Any]) -> str:
        """Format causal analysis results."""
        section = ""
        
        if 'causal_correlations' in results:
            correlations = results['causal_correlations']
            
            if 'strong_correlations' in correlations:
                strong_corrs = correlations['strong_correlations']
                
                if strong_corrs:
                    section += "Strong correlations were identified between the following variables:\n\n"
                    section += "\\begin{itemize}\n"
                    
                    for corr in strong_corrs:
                        section += f"\\item {corr['variable1']} and {corr['variable2']} (r = {corr['correlation']:.3f}, {corr['strength'].lower()} correlation)\n"
                    
                    section += "\\end{itemize}\n\n"
        
        if 'intervention_effects' in results:
            interventions = results['intervention_effects']
            
            if 'effect_size' in interventions:
                effect = interventions['effect_size']
                section += f"The intervention showed a {effect['interpretation'].lower()} (Cohen's d = {effect['cohens_d']:.3f}).\n\n"
        
        return section
    
    def _generate_discussion_section(self) -> str:
        """Generate discussion section."""
        section = "\\section{Discussion}\n\n"
        
        section += """The results demonstrate the effectiveness of the ESCAI Framework in capturing and analyzing complex agent behaviors. Several key findings emerge from this analysis:\n\n"""
        
        section += """\\subsection{Performance Insights}\n\n"""
        section += """The performance analysis reveals significant patterns in agent behavior that correlate with task success. These findings have important implications for agent system design and optimization.\n\n"""
        
        section += """\\subsection{Epistemic State Dynamics}\n\n"""
        section += """The epistemic state analysis provides insights into how agent knowledge, beliefs, and goals evolve during task execution. Understanding these dynamics is crucial for developing more robust and reliable agent systems.\n\n"""
        
        section += """\\subsection{Causal Relationships}\n\n"""
        section += """The identification of causal relationships between agent actions and outcomes enables more targeted interventions and system improvements. These findings contribute to our understanding of agent decision-making processes.\n\n"""
        
        section += """\\subsection{Limitations}\n\n"""
        section += """This analysis is subject to several limitations. The observational nature of the data limits causal inference, and the specific agent frameworks and tasks studied may not generalize to all contexts. Future work should include controlled experiments and broader validation studies.\n\n"""
        
        return section
    
    def _generate_conclusion_section(self) -> str:
        """Generate conclusion section."""
        section = "\\section{Conclusion}\n\n"
        
        section += """This statistical analysis demonstrates the value of comprehensive agent monitoring using the ESCAI Framework. The results provide actionable insights for improving agent performance, understanding epistemic state dynamics, and identifying causal relationships in agent behavior.\n\n"""
        
        section += """Future research should focus on expanding the analysis to include more diverse agent frameworks, tasks, and environments. Additionally, the development of more sophisticated causal inference methods specifically designed for agent monitoring data would enhance the analytical capabilities of the framework.\n\n"""
        
        section += """The ESCAI Framework represents a significant advancement in agent observability and analysis, providing researchers and practitioners with powerful tools for understanding and improving autonomous agent systems.\n\n"""
        
        return section
    
    def generate_methodology_citations(self, methodologies: List[str]) -> List[str]:
        """Get citation keys for specified methodologies."""
        all_citations = []
        
        for methodology in methodologies:
            citations = self.methodology_generator.get_methodology_citations(methodology)
            all_citations.extend(citations)
        
        return list(set(all_citations))  # Remove duplicates