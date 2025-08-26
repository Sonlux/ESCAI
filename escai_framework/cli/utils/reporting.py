"""
Advanced reporting and export capabilities for ESCAI CLI.

This module provides comprehensive report generation, automated scheduling,
and export functionality for analysis results and monitoring data.
"""

import asyncio
import json
import csv
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import zipfile

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.text import Text

from ..services.api_client import ESCAIAPIClient
# Utility functions for formatting
def format_timestamp(timestamp):
    """Format timestamp for display."""
    if isinstance(timestamp, str):
        return timestamp
    return timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "N/A"

def format_duration(seconds):
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_percentage(value):
    """Format value as percentage."""
    return f"{value:.1%}" if isinstance(value, (int, float)) else str(value)


class ReportFormat(Enum):
    """Supported report output formats."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "md"
    HTML = "html"
    PDF = "pdf"
    TXT = "txt"


class ReportType(Enum):
    """Types of reports that can be generated."""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PERFORMANCE_REPORT = "performance_report"
    CUSTOM = "custom"


@dataclass
class ReportTemplate:
    """Template configuration for report generation."""
    name: str
    type: ReportType
    description: str
    sections: List[str]
    default_format: ReportFormat
    parameters: Dict[str, Any]


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    template: ReportTemplate
    output_format: ReportFormat
    output_path: Optional[Path]
    date_range: tuple[datetime, datetime]
    filters: Dict[str, Any]
    include_charts: bool = True
    include_raw_data: bool = False
    compress_output: bool = False


class ReportGenerator:
    """Advanced report generation system."""
    
    def __init__(self, api_client: ESCAIAPIClient, console: Console):
        self.api_client = api_client
        self.console = console
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, ReportTemplate]:
        """Load predefined report templates."""
        return {
            "executive_summary": ReportTemplate(
                name="Executive Summary",
                type=ReportType.EXECUTIVE_SUMMARY,
                description="High-level overview with key insights and recommendations",
                sections=["overview", "key_metrics", "insights", "recommendations"],
                default_format=ReportFormat.PDF,
                parameters={"include_charts": True, "summary_length": "brief"}
            ),
            "detailed_analysis": ReportTemplate(
                name="Detailed Analysis",
                type=ReportType.DETAILED_ANALYSIS,
                description="Comprehensive analysis with full data and visualizations",
                sections=["overview", "epistemic_analysis", "behavioral_patterns", 
                         "causal_relationships", "performance_metrics", "raw_data"],
                default_format=ReportFormat.HTML,
                parameters={"include_charts": True, "include_raw_data": True}
            ),
            "trend_analysis": ReportTemplate(
                name="Trend Analysis",
                type=ReportType.TREND_ANALYSIS,
                description="Time-series analysis with forecasting and trend identification",
                sections=["trend_overview", "statistical_analysis", "forecasting", "anomalies"],
                default_format=ReportFormat.HTML,
                parameters={"forecast_periods": 30, "confidence_interval": 0.95}
            ),
            "comparative_analysis": ReportTemplate(
                name="Comparative Analysis",
                type=ReportType.COMPARATIVE_ANALYSIS,
                description="Side-by-side comparison of agents, sessions, or time periods",
                sections=["comparison_overview", "performance_comparison", "behavioral_differences", "recommendations"],
                default_format=ReportFormat.HTML,
                parameters={"comparison_type": "agents", "statistical_tests": True}
            )
        }
    
    async def generate_report(self, config: ReportConfig) -> Path:
        """Generate a report based on the provided configuration."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating report...", total=None)
            
            # Collect data
            progress.update(task, description="Collecting data...")
            data = await self._collect_report_data(config)
            
            # Generate content
            progress.update(task, description="Generating content...")
            content = await self._generate_content(config, data)
            
            # Format output
            progress.update(task, description="Formatting output...")
            output_path = await self._format_output(config, content)
            
            progress.update(task, description="Report generated successfully!")
        
        return output_path
    
    async def _collect_report_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Collect all necessary data for the report."""
        data = {}
        
        # Get agents data
        agents_response = await self.api_client.get("/agents")
        data["agents"] = agents_response.get("agents", [])
        
        # Get monitoring sessions
        sessions_response = await self.api_client.get("/monitoring/sessions")
        data["sessions"] = sessions_response.get("sessions", [])
        
        # Get epistemic states
        states_response = await self.api_client.get("/analysis/epistemic-states")
        data["epistemic_states"] = states_response.get("states", [])
        
        # Get behavioral patterns
        patterns_response = await self.api_client.get("/analysis/patterns")
        data["patterns"] = patterns_response.get("patterns", [])
        
        # Get causal relationships
        causal_response = await self.api_client.get("/analysis/causal")
        data["causal_relationships"] = causal_response.get("relationships", [])
        
        # Get predictions
        predictions_response = await self.api_client.get("/analysis/predictions")
        data["predictions"] = predictions_response.get("predictions", [])
        
        return data
    
    async def _generate_content(self, config: ReportConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report content based on template and data."""
        content = {
            "metadata": {
                "title": config.template.name,
                "generated_at": datetime.now().isoformat(),
                "date_range": {
                    "start": config.date_range[0].isoformat(),
                    "end": config.date_range[1].isoformat()
                },
                "template": config.template.name,
                "format": config.output_format.value
            },
            "sections": {}
        }
        
        # Generate each section based on template
        for section in config.template.sections:
            content["sections"][section] = await self._generate_section(section, data, config)
        
        return content
    
    async def _generate_section(self, section: str, data: Dict[str, Any], config: ReportConfig) -> Dict[str, Any]:
        """Generate content for a specific report section."""
        if section == "overview":
            return self._generate_overview(data)
        elif section == "key_metrics":
            return self._generate_key_metrics(data)
        elif section == "insights":
            return self._generate_insights(data)
        elif section == "recommendations":
            return self._generate_recommendations(data)
        elif section == "epistemic_analysis":
            return self._generate_epistemic_analysis(data)
        elif section == "behavioral_patterns":
            return self._generate_behavioral_patterns(data)
        elif section == "causal_relationships":
            return self._generate_causal_relationships(data)
        elif section == "performance_metrics":
            return self._generate_performance_metrics(data)
        elif section == "trend_overview":
            return self._generate_trend_overview(data)
        elif section == "statistical_analysis":
            return self._generate_statistical_analysis(data)
        elif section == "forecasting":
            return self._generate_forecasting(data)
        elif section == "anomalies":
            return self._generate_anomalies(data)
        elif section == "comparison_overview":
            return self._generate_comparison_overview(data)
        elif section == "performance_comparison":
            return self._generate_performance_comparison(data)
        elif section == "behavioral_differences":
            return self._generate_behavioral_differences(data)
        elif section == "raw_data":
            return self._generate_raw_data(data)
        else:
            return {"content": f"Section '{section}' not implemented", "type": "text"}
    
    def _generate_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overview section."""
        total_agents = len(data.get("agents", []))
        total_sessions = len(data.get("sessions", []))
        total_states = len(data.get("epistemic_states", []))
        total_patterns = len(data.get("patterns", []))
        
        return {
            "type": "overview",
            "content": {
                "summary": f"Analysis of {total_agents} agents across {total_sessions} monitoring sessions",
                "metrics": {
                    "total_agents": total_agents,
                    "total_sessions": total_sessions,
                    "epistemic_states_captured": total_states,
                    "behavioral_patterns_identified": total_patterns
                }
            }
        }
    
    def _generate_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate key metrics section."""
        # Calculate key performance indicators
        sessions = data.get("sessions", [])
        successful_sessions = [s for s in sessions if s.get("status") == "completed"]
        success_rate = len(successful_sessions) / len(sessions) if sessions else 0
        
        avg_duration = sum(s.get("duration", 0) for s in sessions) / len(sessions) if sessions else 0
        
        return {
            "type": "metrics",
            "content": {
                "success_rate": success_rate,
                "average_session_duration": avg_duration,
                "total_monitoring_time": sum(s.get("duration", 0) for s in sessions),
                "active_agents": len([a for a in data.get("agents", []) if a.get("status") == "active"])
            }
        }
    
    def _generate_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights section with key findings."""
        insights = []
        
        # Analyze patterns for insights
        patterns = data.get("patterns", [])
        if patterns:
            most_common_pattern = max(patterns, key=lambda p: p.get("frequency", 0))
            insights.append(f"Most common behavioral pattern: {most_common_pattern.get('name', 'Unknown')}")
        
        # Analyze success rates
        sessions = data.get("sessions", [])
        if sessions:
            success_rate = len([s for s in sessions if s.get("status") == "completed"]) / len(sessions)
            if success_rate > 0.8:
                insights.append("High success rate indicates stable agent performance")
            elif success_rate < 0.5:
                insights.append("Low success rate suggests need for optimization")
        
        return {
            "type": "insights",
            "content": {
                "key_findings": insights,
                "analysis_confidence": "high" if len(data.get("sessions", [])) > 10 else "medium"
            }
        }
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Analyze performance and suggest improvements
        sessions = data.get("sessions", [])
        if sessions:
            failed_sessions = [s for s in sessions if s.get("status") == "failed"]
            if len(failed_sessions) > len(sessions) * 0.2:
                recommendations.append("Consider implementing additional error handling and retry mechanisms")
        
        patterns = data.get("patterns", [])
        if patterns:
            anomalous_patterns = [p for p in patterns if p.get("anomaly_score", 0) > 0.8]
            if anomalous_patterns:
                recommendations.append("Investigate anomalous behavioral patterns for potential optimization")
        
        return {
            "type": "recommendations",
            "content": {
                "action_items": recommendations,
                "priority": "high" if len(recommendations) > 3 else "medium"
            }
        }
    
    def _generate_epistemic_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed epistemic state analysis."""
        states = data.get("epistemic_states", [])
        
        if not states:
            return {"type": "analysis", "content": {"message": "No epistemic states available"}}
        
        # Analyze belief evolution
        belief_changes = []
        confidence_trends = []
        
        for state in states:
            beliefs = state.get("beliefs", {})
            confidence = state.get("confidence", 0)
            
            belief_changes.append(len(beliefs))
            confidence_trends.append(confidence)
        
        avg_beliefs = sum(belief_changes) / len(belief_changes) if belief_changes else 0
        avg_confidence = sum(confidence_trends) / len(confidence_trends) if confidence_trends else 0
        
        return {
            "type": "analysis",
            "content": {
                "total_states": len(states),
                "average_beliefs_per_state": avg_beliefs,
                "average_confidence": avg_confidence,
                "confidence_trend": "increasing" if confidence_trends[-1] > confidence_trends[0] else "decreasing"
            }
        }
    
    def _generate_behavioral_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate behavioral pattern analysis."""
        patterns = data.get("patterns", [])
        
        if not patterns:
            return {"type": "analysis", "content": {"message": "No behavioral patterns identified"}}
        
        # Analyze pattern frequency and types
        pattern_types = {}
        total_frequency = 0
        
        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            frequency = pattern.get("frequency", 0)
            
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + frequency
            total_frequency += frequency
        
        return {
            "type": "analysis",
            "content": {
                "total_patterns": len(patterns),
                "pattern_types": pattern_types,
                "most_frequent_type": max(pattern_types.keys(), key=pattern_types.get) if pattern_types else None,
                "total_occurrences": total_frequency
            }
        }
    
    def _generate_causal_relationships(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate causal relationship analysis."""
        relationships = data.get("causal_relationships", [])
        
        if not relationships:
            return {"type": "analysis", "content": {"message": "No causal relationships identified"}}
        
        # Analyze relationship strength and types
        strong_relationships = [r for r in relationships if r.get("strength", 0) > 0.7]
        relationship_types = {}
        
        for rel in relationships:
            rel_type = rel.get("type", "unknown")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "type": "analysis",
            "content": {
                "total_relationships": len(relationships),
                "strong_relationships": len(strong_relationships),
                "relationship_types": relationship_types,
                "average_strength": sum(r.get("strength", 0) for r in relationships) / len(relationships)
            }
        }
    
    def _generate_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance metrics analysis."""
        sessions = data.get("sessions", [])
        predictions = data.get("predictions", [])
        
        if not sessions:
            return {"type": "metrics", "content": {"message": "No performance data available"}}
        
        # Calculate performance metrics
        total_duration = sum(s.get("duration", 0) for s in sessions)
        avg_duration = total_duration / len(sessions)
        success_rate = len([s for s in sessions if s.get("status") == "completed"]) / len(sessions)
        
        prediction_accuracy = 0
        if predictions:
            accurate_predictions = [p for p in predictions if p.get("accuracy", 0) > 0.8]
            prediction_accuracy = len(accurate_predictions) / len(predictions)
        
        return {
            "type": "metrics",
            "content": {
                "total_sessions": len(sessions),
                "average_duration": avg_duration,
                "success_rate": success_rate,
                "prediction_accuracy": prediction_accuracy,
                "total_monitoring_time": total_duration
            }
        }
    
    def _generate_trend_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trend analysis overview."""
        sessions = data.get("sessions", [])
        
        # Sort sessions by timestamp for trend analysis
        sorted_sessions = sorted(sessions, key=lambda s: s.get("timestamp", ""))
        
        if len(sorted_sessions) < 2:
            return {"type": "trend", "content": {"message": "Insufficient data for trend analysis"}}
        
        # Calculate trends
        recent_sessions = sorted_sessions[-10:]  # Last 10 sessions
        older_sessions = sorted_sessions[-20:-10] if len(sorted_sessions) >= 20 else sorted_sessions[:-10]
        
        recent_success_rate = len([s for s in recent_sessions if s.get("status") == "completed"]) / len(recent_sessions)
        older_success_rate = len([s for s in older_sessions if s.get("status") == "completed"]) / len(older_sessions) if older_sessions else 0
        
        trend_direction = "improving" if recent_success_rate > older_success_rate else "declining"
        
        return {
            "type": "trend",
            "content": {
                "trend_direction": trend_direction,
                "recent_success_rate": recent_success_rate,
                "previous_success_rate": older_success_rate,
                "change_magnitude": abs(recent_success_rate - older_success_rate)
            }
        }
    
    def _generate_statistical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis section."""
        sessions = data.get("sessions", [])
        
        if not sessions:
            return {"type": "statistics", "content": {"message": "No data for statistical analysis"}}
        
        # Calculate basic statistics
        durations = [s.get("duration", 0) for s in sessions]
        durations.sort()
        
        n = len(durations)
        mean_duration = sum(durations) / n
        median_duration = durations[n // 2] if n % 2 == 1 else (durations[n // 2 - 1] + durations[n // 2]) / 2
        
        # Calculate standard deviation
        variance = sum((d - mean_duration) ** 2 for d in durations) / n
        std_deviation = variance ** 0.5
        
        return {
            "type": "statistics",
            "content": {
                "sample_size": n,
                "mean_duration": mean_duration,
                "median_duration": median_duration,
                "standard_deviation": std_deviation,
                "min_duration": min(durations),
                "max_duration": max(durations)
            }
        }
    
    def _generate_forecasting(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecasting analysis."""
        sessions = data.get("sessions", [])
        
        if len(sessions) < 5:
            return {"type": "forecast", "content": {"message": "Insufficient data for forecasting"}}
        
        # Simple trend-based forecasting
        sorted_sessions = sorted(sessions, key=lambda s: s.get("timestamp", ""))
        recent_success_rates = []
        
        # Calculate success rate for each time window
        window_size = max(1, len(sorted_sessions) // 5)
        for i in range(0, len(sorted_sessions), window_size):
            window = sorted_sessions[i:i + window_size]
            success_rate = len([s for s in window if s.get("status") == "completed"]) / len(window)
            recent_success_rates.append(success_rate)
        
        # Simple linear trend
        if len(recent_success_rates) >= 2:
            trend = (recent_success_rates[-1] - recent_success_rates[0]) / (len(recent_success_rates) - 1)
            forecast = recent_success_rates[-1] + trend
            forecast = max(0, min(1, forecast))  # Clamp between 0 and 1
        else:
            forecast = recent_success_rates[-1] if recent_success_rates else 0.5
        
        return {
            "type": "forecast",
            "content": {
                "forecasted_success_rate": forecast,
                "trend_slope": trend if 'trend' in locals() else 0,
                "confidence_interval": [max(0, forecast - 0.1), min(1, forecast + 0.1)],
                "forecast_horizon": "next_period"
            }
        }
    
    def _generate_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate anomaly detection analysis."""
        patterns = data.get("patterns", [])
        sessions = data.get("sessions", [])
        
        anomalies = []
        
        # Detect pattern anomalies
        if patterns:
            avg_frequency = sum(p.get("frequency", 0) for p in patterns) / len(patterns)
            for pattern in patterns:
                if pattern.get("frequency", 0) > avg_frequency * 3:  # 3x above average
                    anomalies.append({
                        "type": "high_frequency_pattern",
                        "description": f"Pattern '{pattern.get('name', 'Unknown')}' occurs unusually frequently",
                        "severity": "medium"
                    })
        
        # Detect session anomalies
        if sessions:
            durations = [s.get("duration", 0) for s in sessions]
            avg_duration = sum(durations) / len(durations)
            std_dev = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
            
            for session in sessions:
                duration = session.get("duration", 0)
                if abs(duration - avg_duration) > 2 * std_dev:  # 2 standard deviations
                    anomalies.append({
                        "type": "unusual_duration",
                        "description": f"Session {session.get('id', 'Unknown')} has unusual duration: {duration}",
                        "severity": "low"
                    })
        
        return {
            "type": "anomalies",
            "content": {
                "total_anomalies": len(anomalies),
                "anomalies": anomalies,
                "detection_method": "statistical_threshold"
            }
        }
    
    def _generate_comparison_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison overview section."""
        agents = data.get("agents", [])
        
        if len(agents) < 2:
            return {"type": "comparison", "content": {"message": "Need at least 2 agents for comparison"}}
        
        # Compare agent performance
        agent_metrics = {}
        for agent in agents:
            agent_id = agent.get("id", "unknown")
            # Get sessions for this agent
            agent_sessions = [s for s in data.get("sessions", []) if s.get("agent_id") == agent_id]
            
            if agent_sessions:
                success_rate = len([s for s in agent_sessions if s.get("status") == "completed"]) / len(agent_sessions)
                avg_duration = sum(s.get("duration", 0) for s in agent_sessions) / len(agent_sessions)
                
                agent_metrics[agent_id] = {
                    "success_rate": success_rate,
                    "average_duration": avg_duration,
                    "total_sessions": len(agent_sessions)
                }
        
        return {
            "type": "comparison",
            "content": {
                "agents_compared": len(agent_metrics),
                "metrics": agent_metrics,
                "best_performer": max(agent_metrics.keys(), key=lambda k: agent_metrics[k]["success_rate"]) if agent_metrics else None
            }
        }
    
    def _generate_performance_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed performance comparison."""
        agents = data.get("agents", [])
        sessions = data.get("sessions", [])
        
        # Group sessions by agent
        agent_performance = {}
        for session in sessions:
            agent_id = session.get("agent_id", "unknown")
            if agent_id not in agent_performance:
                agent_performance[agent_id] = []
            agent_performance[agent_id].append(session)
        
        # Calculate comparative metrics
        comparison_data = {}
        for agent_id, agent_sessions in agent_performance.items():
            success_rate = len([s for s in agent_sessions if s.get("status") == "completed"]) / len(agent_sessions)
            avg_duration = sum(s.get("duration", 0) for s in agent_sessions) / len(agent_sessions)
            total_time = sum(s.get("duration", 0) for s in agent_sessions)
            
            comparison_data[agent_id] = {
                "success_rate": success_rate,
                "average_duration": avg_duration,
                "total_monitoring_time": total_time,
                "session_count": len(agent_sessions)
            }
        
        return {
            "type": "performance_comparison",
            "content": {
                "agent_metrics": comparison_data,
                "ranking": sorted(comparison_data.keys(), key=lambda k: comparison_data[k]["success_rate"], reverse=True)
            }
        }
    
    def _generate_behavioral_differences(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate behavioral differences analysis."""
        patterns = data.get("patterns", [])
        
        # Group patterns by agent
        agent_patterns = {}
        for pattern in patterns:
            agent_id = pattern.get("agent_id", "unknown")
            if agent_id not in agent_patterns:
                agent_patterns[agent_id] = []
            agent_patterns[agent_id].append(pattern)
        
        # Analyze differences
        behavioral_analysis = {}
        for agent_id, agent_pattern_list in agent_patterns.items():
            pattern_types = {}
            for pattern in agent_pattern_list:
                pattern_type = pattern.get("type", "unknown")
                pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            
            behavioral_analysis[agent_id] = {
                "total_patterns": len(agent_pattern_list),
                "pattern_distribution": pattern_types,
                "dominant_pattern": max(pattern_types.keys(), key=pattern_types.get) if pattern_types else None
            }
        
        return {
            "type": "behavioral_differences",
            "content": {
                "agent_behaviors": behavioral_analysis,
                "unique_patterns": len(set(p.get("type", "unknown") for p in patterns))
            }
        }
    
    def _generate_raw_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate raw data section."""
        return {
            "type": "raw_data",
            "content": {
                "agents": data.get("agents", []),
                "sessions": data.get("sessions", []),
                "epistemic_states": data.get("epistemic_states", []),
                "patterns": data.get("patterns", []),
                "causal_relationships": data.get("causal_relationships", []),
                "predictions": data.get("predictions", [])
            }
        }
    
    async def _format_output(self, config: ReportConfig, content: Dict[str, Any]) -> Path:
        """Format and save the report in the specified format."""
        output_path = config.output_path or Path(f"escai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if config.output_format == ReportFormat.JSON:
            output_path = output_path.with_suffix('.json')
            with open(output_path, 'w') as f:
                json.dump(content, f, indent=2, default=str)
        
        elif config.output_format == ReportFormat.CSV:
            output_path = output_path.with_suffix('.csv')
            self._export_to_csv(content, output_path)
        
        elif config.output_format == ReportFormat.MARKDOWN:
            output_path = output_path.with_suffix('.md')
            self._export_to_markdown(content, output_path)
        
        elif config.output_format == ReportFormat.HTML:
            output_path = output_path.with_suffix('.html')
            self._export_to_html(content, output_path)
        
        elif config.output_format == ReportFormat.TXT:
            output_path = output_path.with_suffix('.txt')
            self._export_to_text(content, output_path)
        
        # Compress if requested
        if config.compress_output:
            compressed_path = output_path.with_suffix(output_path.suffix + '.zip')
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(output_path, output_path.name)
            output_path.unlink()  # Remove original
            output_path = compressed_path
        
        return output_path
    
    def _export_to_csv(self, content: Dict[str, Any], output_path: Path):
        """Export report content to CSV format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write metadata
            writer.writerow(['Report Metadata'])
            writer.writerow(['Title', content['metadata']['title']])
            writer.writerow(['Generated At', content['metadata']['generated_at']])
            writer.writerow(['Date Range', f"{content['metadata']['date_range']['start']} to {content['metadata']['date_range']['end']}"])
            writer.writerow([])
            
            # Write sections
            for section_name, section_data in content['sections'].items():
                writer.writerow([f'Section: {section_name.title()}'])
                
                if section_data['type'] == 'metrics':
                    for key, value in section_data['content'].items():
                        writer.writerow([key, value])
                elif section_data['type'] == 'analysis':
                    for key, value in section_data['content'].items():
                        if isinstance(value, dict):
                            writer.writerow([key, json.dumps(value)])
                        else:
                            writer.writerow([key, value])
                
                writer.writerow([])
    
    def _export_to_markdown(self, content: Dict[str, Any], output_path: Path):
        """Export report content to Markdown format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# {content['metadata']['title']}\n\n")
            f.write(f"**Generated:** {content['metadata']['generated_at']}  \n")
            f.write(f"**Date Range:** {content['metadata']['date_range']['start']} to {content['metadata']['date_range']['end']}  \n")
            f.write(f"**Template:** {content['metadata']['template']}  \n\n")
            
            # Write sections
            for section_name, section_data in content['sections'].items():
                f.write(f"## {section_name.replace('_', ' ').title()}\n\n")
                
                if section_data['type'] == 'overview':
                    f.write(f"{section_data['content']['summary']}\n\n")
                    f.write("### Key Statistics\n\n")
                    for key, value in section_data['content']['metrics'].items():
                        f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                
                elif section_data['type'] == 'metrics':
                    for key, value in section_data['content'].items():
                        if isinstance(value, float):
                            f.write(f"- **{key.replace('_', ' ').title()}:** {value:.2%}\n")
                        else:
                            f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                
                elif section_data['type'] == 'insights':
                    f.write("### Key Findings\n\n")
                    for insight in section_data['content']['key_findings']:
                        f.write(f"- {insight}\n")
                
                elif section_data['type'] == 'recommendations':
                    f.write("### Action Items\n\n")
                    for rec in section_data['content']['action_items']:
                        f.write(f"- {rec}\n")
                
                f.write("\n")
    
    def _export_to_html(self, content: Dict[str, Any], output_path: Path):
        """Export report content to HTML format."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{content['metadata']['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; }}
        ul {{ list-style-type: none; padding-left: 0; }}
        li {{ margin-bottom: 10px; padding: 10px; background: #f9f9f9; border-left: 4px solid #3498db; }}
        .recommendations li {{ border-left-color: #e74c3c; }}
        .insights li {{ border-left-color: #27ae60; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{content['metadata']['title']}</h1>
        <p><strong>Generated:</strong> {content['metadata']['generated_at']}</p>
        <p><strong>Date Range:</strong> {content['metadata']['date_range']['start']} to {content['metadata']['date_range']['end']}</p>
        <p><strong>Template:</strong> {content['metadata']['template']}</p>
    </div>
"""
        
        # Add sections
        for section_name, section_data in content['sections'].items():
            html_content += f'<div class="section">\n<h2>{section_name.replace("_", " ").title()}</h2>\n'
            
            if section_data['type'] == 'overview':
                html_content += f'<p>{section_data["content"]["summary"]}</p>\n'
                html_content += '<div class="metrics">\n'
                for key, value in section_data['content']['metrics'].items():
                    html_content += f'''
                    <div class="metric-card">
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{key.replace("_", " ").title()}</div>
                    </div>
                    '''
                html_content += '</div>\n'
            
            elif section_data['type'] == 'metrics':
                html_content += '<div class="metrics">\n'
                for key, value in section_data['content'].items():
                    if isinstance(value, float):
                        display_value = f"{value:.2%}"
                    else:
                        display_value = str(value)
                    html_content += f'''
                    <div class="metric-card">
                        <div class="metric-value">{display_value}</div>
                        <div class="metric-label">{key.replace("_", " ").title()}</div>
                    </div>
                    '''
                html_content += '</div>\n'
            
            elif section_data['type'] == 'insights':
                html_content += '<ul class="insights">\n'
                for insight in section_data['content']['key_findings']:
                    html_content += f'<li>{insight}</li>\n'
                html_content += '</ul>\n'
            
            elif section_data['type'] == 'recommendations':
                html_content += '<ul class="recommendations">\n'
                for rec in section_data['content']['action_items']:
                    html_content += f'<li>{rec}</li>\n'
                html_content += '</ul>\n'
            
            html_content += '</div>\n'
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_to_text(self, content: Dict[str, Any], output_path: Path):
        """Export report content to plain text format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"{content['metadata']['title']}\n")
            f.write("=" * len(content['metadata']['title']) + "\n\n")
            f.write(f"Generated: {content['metadata']['generated_at']}\n")
            f.write(f"Date Range: {content['metadata']['date_range']['start']} to {content['metadata']['date_range']['end']}\n")
            f.write(f"Template: {content['metadata']['template']}\n\n")
            
            # Write sections
            for section_name, section_data in content['sections'].items():
                section_title = section_name.replace('_', ' ').title()
                f.write(f"{section_title}\n")
                f.write("-" * len(section_title) + "\n\n")
                
                if section_data['type'] == 'overview':
                    f.write(f"{section_data['content']['summary']}\n\n")
                    f.write("Key Statistics:\n")
                    for key, value in section_data['content']['metrics'].items():
                        f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                
                elif section_data['type'] == 'metrics':
                    for key, value in section_data['content'].items():
                        if isinstance(value, float):
                            f.write(f"  {key.replace('_', ' ').title()}: {value:.2%}\n")
                        else:
                            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                
                elif section_data['type'] == 'insights':
                    f.write("Key Findings:\n")
                    for insight in section_data['content']['key_findings']:
                        f.write(f"  - {insight}\n")
                
                elif section_data['type'] == 'recommendations':
                    f.write("Action Items:\n")
                    for rec in section_data['content']['action_items']:
                        f.write(f"  - {rec}\n")
                
                f.write("\n")
    
    def list_templates(self) -> List[ReportTemplate]:
        """List all available report templates."""
        return list(self.templates.values())
    
    def get_template(self, name: str) -> Optional[ReportTemplate]:
        """Get a specific report template by name."""
        return self.templates.get(name)


class ReportScheduler:
    """Automated report scheduling system."""
    
    def __init__(self, generator: ReportGenerator, console: Console):
        self.generator = generator
        self.console = console
        self.scheduled_reports = []
    
    def schedule_report(self, config: ReportConfig, schedule: str, email_recipients: List[str] = None):
        """Schedule a report to be generated automatically."""
        scheduled_report = {
            "id": len(self.scheduled_reports) + 1,
            "config": config,
            "schedule": schedule,
            "email_recipients": email_recipients or [],
            "last_run": None,
            "next_run": self._calculate_next_run(schedule),
            "enabled": True
        }
        
        self.scheduled_reports.append(scheduled_report)
        
        self.console.print(f"[green]✓[/green] Report scheduled successfully (ID: {scheduled_report['id']})")
        return scheduled_report["id"]
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate the next run time based on schedule string."""
        now = datetime.now()
        
        if schedule == "daily":
            return now + timedelta(days=1)
        elif schedule == "weekly":
            return now + timedelta(weeks=1)
        elif schedule == "monthly":
            return now + timedelta(days=30)
        elif schedule.startswith("every_"):
            # Parse "every_X_hours" or "every_X_minutes"
            parts = schedule.split("_")
            if len(parts) == 3:
                interval = int(parts[1])
                unit = parts[2]
                
                if unit == "hours":
                    return now + timedelta(hours=interval)
                elif unit == "minutes":
                    return now + timedelta(minutes=interval)
        
        # Default to daily
        return now + timedelta(days=1)
    
    async def run_scheduled_reports(self):
        """Run any scheduled reports that are due."""
        now = datetime.now()
        
        for report in self.scheduled_reports:
            if report["enabled"] and report["next_run"] <= now:
                try:
                    self.console.print(f"[blue]Running scheduled report {report['id']}...[/blue]")
                    
                    # Generate report
                    output_path = await self.generator.generate_report(report["config"])
                    
                    # Send email if recipients specified
                    if report["email_recipients"]:
                        await self._send_report_email(output_path, report["email_recipients"])
                    
                    # Update schedule
                    report["last_run"] = now
                    report["next_run"] = self._calculate_next_run(report["schedule"])
                    
                    self.console.print(f"[green]✓[/green] Scheduled report {report['id']} completed")
                    
                except Exception as e:
                    self.console.print(f"[red]✗[/red] Scheduled report {report['id']} failed: {e}")
    
    async def _send_report_email(self, report_path: Path, recipients: List[str]):
        """Send report via email (placeholder implementation)."""
        # This would integrate with an email service
        self.console.print(f"[yellow]Email delivery not implemented. Report saved to: {report_path}[/yellow]")
    
    def list_scheduled_reports(self) -> List[Dict[str, Any]]:
        """List all scheduled reports."""
        return self.scheduled_reports
    
    def disable_scheduled_report(self, report_id: int):
        """Disable a scheduled report."""
        for report in self.scheduled_reports:
            if report["id"] == report_id:
                report["enabled"] = False
                self.console.print(f"[yellow]Scheduled report {report_id} disabled[/yellow]")
                return
        
        self.console.print(f"[red]Scheduled report {report_id} not found[/red]")
    
    def enable_scheduled_report(self, report_id: int):
        """Enable a scheduled report."""
        for report in self.scheduled_reports:
            if report["id"] == report_id:
                report["enabled"] = True
                self.console.print(f"[green]Scheduled report {report_id} enabled[/green]")
                return
        
        self.console.print(f"[red]Scheduled report {report_id} not found[/red]")


class CustomReportBuilder:
    """Interactive custom report builder."""
    
    def __init__(self, generator: ReportGenerator, console: Console):
        self.generator = generator
        self.console = console
    
    def build_custom_report(self) -> ReportConfig:
        """Interactive custom report builder."""
        self.console.print("[bold blue]Custom Report Builder[/bold blue]")
        self.console.print("Let's create a custom report tailored to your needs.\n")
        
        # Get report name
        report_name = Prompt.ask("Report name", default="Custom Analysis Report")
        
        # Select sections
        available_sections = [
            "overview", "key_metrics", "insights", "recommendations",
            "epistemic_analysis", "behavioral_patterns", "causal_relationships",
            "performance_metrics", "trend_overview", "statistical_analysis",
            "forecasting", "anomalies", "comparison_overview", "performance_comparison",
            "behavioral_differences", "raw_data"
        ]
        
        self.console.print("\nAvailable sections:")
        for i, section in enumerate(available_sections, 1):
            self.console.print(f"  {i}. {section.replace('_', ' ').title()}")
        
        selected_sections = []
        while True:
            section_input = Prompt.ask(
                "\nSelect sections (comma-separated numbers, or 'done' to finish)",
                default="1,2,3,4"
            )
            
            if section_input.lower() == 'done':
                break
            
            try:
                section_indices = [int(x.strip()) - 1 for x in section_input.split(',')]
                selected_sections = [available_sections[i] for i in section_indices if 0 <= i < len(available_sections)]
                break
            except (ValueError, IndexError):
                self.console.print("[red]Invalid selection. Please try again.[/red]")
        
        # Select output format
        formats = list(ReportFormat)
        self.console.print("\nOutput formats:")
        for i, fmt in enumerate(formats, 1):
            self.console.print(f"  {i}. {fmt.value.upper()}")
        
        format_choice = Prompt.ask("Select output format", choices=[str(i) for i in range(1, len(formats) + 1)], default="3")
        output_format = formats[int(format_choice) - 1]
        
        # Get date range
        self.console.print("\nDate range:")
        days_back = int(Prompt.ask("Days back from today", default="7"))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Additional options
        include_charts = Confirm.ask("Include charts and visualizations?", default=True)
        include_raw_data = Confirm.ask("Include raw data?", default=False)
        compress_output = Confirm.ask("Compress output file?", default=False)
        
        # Create custom template
        custom_template = ReportTemplate(
            name=report_name,
            type=ReportType.CUSTOM,
            description="Custom report created by user",
            sections=selected_sections,
            default_format=output_format,
            parameters={
                "include_charts": include_charts,
                "include_raw_data": include_raw_data
            }
        )
        
        # Create config
        config = ReportConfig(
            template=custom_template,
            output_format=output_format,
            output_path=None,
            date_range=(start_date, end_date),
            filters={},
            include_charts=include_charts,
            include_raw_data=include_raw_data,
            compress_output=compress_output
        )
        
        self.console.print(f"\n[green]✓[/green] Custom report configuration created!")
        return config


# CLI Integration Functions
def create_report_generator(api_client: ESCAIAPIClient, console: Console) -> ReportGenerator:
    """Create a report generator instance."""
    return ReportGenerator(api_client, console)


def create_report_scheduler(generator: ReportGenerator, console: Console) -> ReportScheduler:
    """Create a report scheduler instance."""
    return ReportScheduler(generator, console)


def create_custom_report_builder(generator: ReportGenerator, console: Console) -> CustomReportBuilder:
    """Create a custom report builder instance."""
    return CustomReportBuilder(generator, console)