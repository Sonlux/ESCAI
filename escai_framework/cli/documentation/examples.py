"""
Practical example generator with sample outputs.

This module generates comprehensive examples for CLI commands with
realistic sample outputs and detailed explanations.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import textwrap


@dataclass
class CommandExample:
    """Represents a practical example of command usage."""
    title: str
    description: str
    command: str
    sample_output: str
    explanation: str
    use_case: str
    prerequisites: List[str]
    follow_up_actions: List[str]


class ExampleGenerator:
    """Generates practical examples with sample outputs for CLI commands."""
    
    def __init__(self):
        self._example_templates = self._initialize_example_templates()
        self._sample_outputs = self._initialize_sample_outputs()
    
    def generate_basic_examples(self, command: str) -> List[CommandExample]:
        """
        Generate basic usage examples for a command.
        
        Args:
            command: The command name to generate examples for
            
        Returns:
            List of basic CommandExample objects
        """
        if command not in self._example_templates:
            return [CommandExample(
                title=f"Basic {command} Usage",
                description=f"Example not available for command: {command}",
                command=f"escai {command}",
                sample_output="No sample output available",
                explanation="Command documentation not found",
                use_case="Unknown",
                prerequisites=[],
                follow_up_actions=[]
            )]
        
        examples = []
        templates = self._example_templates[command]['basic']
        
        for template in templates:
            example = CommandExample(
                title=template['title'],
                description=template['description'],
                command=template['command'],
                sample_output=self._get_sample_output(command, template['output_key']),
                explanation=template['explanation'],
                use_case=template['use_case'],
                prerequisites=template['prerequisites'],
                follow_up_actions=template['follow_up_actions']
            )
            examples.append(example)
        
        return examples
    
    def generate_advanced_examples(self, command: str) -> List[CommandExample]:
        """
        Generate advanced usage examples for a command.
        
        Args:
            command: The command name to generate examples for
            
        Returns:
            List of advanced CommandExample objects
        """
        if command not in self._example_templates:
            return []
        
        examples = []
        templates = self._example_templates[command]['advanced']
        
        for template in templates:
            example = CommandExample(
                title=template['title'],
                description=template['description'],
                command=template['command'],
                sample_output=self._get_sample_output(command, template['output_key']),
                explanation=template['explanation'],
                use_case=template['use_case'],
                prerequisites=template['prerequisites'],
                follow_up_actions=template['follow_up_actions']
            )
            examples.append(example)
        
        return examples
    
    def generate_research_examples(self, command: str) -> List[CommandExample]:
        """
        Generate research-focused examples for a command.
        
        Args:
            command: The command name to generate examples for
            
        Returns:
            List of research-focused CommandExample objects
        """
        if command not in self._example_templates:
            return []
        
        examples = []
        templates = self._example_templates[command]['research']
        
        for template in templates:
            example = CommandExample(
                title=template['title'],
                description=template['description'],
                command=template['command'],
                sample_output=self._get_sample_output(command, template['output_key']),
                explanation=template['explanation'],
                use_case=template['use_case'],
                prerequisites=template['prerequisites'],
                follow_up_actions=template['follow_up_actions']
            )
            examples.append(example)
        
        return examples
    
    def generate_workflow_examples(self, workflow_type: str) -> List[CommandExample]:
        """
        Generate examples for complete research workflows.
        
        Args:
            workflow_type: Type of workflow (e.g., 'behavioral_analysis', 'performance_study')
            
        Returns:
            List of workflow CommandExample objects
        """
        workflows = {
            'behavioral_analysis': [
                CommandExample(
                    title="Complete Behavioral Analysis Workflow",
                    description="End-to-end workflow for analyzing agent behavioral patterns",
                    command=textwrap.dedent("""
                        # Step 1: Start monitoring
                        escai monitor --framework langchain --agents all --live
                        
                        # Step 2: Run your agent experiment
                        # (Your agent code runs here)
                        
                        # Step 3: Stop monitoring and analyze
                        escai session stop current
                        escai analyze --data current --type patterns --visualize
                        
                        # Step 4: Export results
                        escai export --session current --format publication
                    """).strip(),
                    sample_output=self._get_sample_output('workflow', 'behavioral_analysis'),
                    explanation="This workflow captures agent behavior, analyzes patterns, and exports publication-ready results",
                    use_case="Research paper on agent behavioral patterns",
                    prerequisites=[
                        "Agent code instrumented with ESCAI",
                        "Experimental scenario prepared",
                        "Research questions defined"
                    ],
                    follow_up_actions=[
                        "Review statistical significance of patterns",
                        "Generate additional visualizations if needed",
                        "Prepare methodology section for paper"
                    ]
                )
            ],
            'performance_study': [
                CommandExample(
                    title="Agent Performance Comparison Study",
                    description="Workflow for comparing performance across different agent configurations",
                    command=textwrap.dedent("""
                        # Step 1: Monitor baseline configuration
                        escai monitor --framework autogen --agents baseline_agent --session baseline
                        
                        # Step 2: Monitor experimental configuration  
                        escai monitor --framework autogen --agents experimental_agent --session experimental
                        
                        # Step 3: Comparative analysis
                        escai analyze --data baseline,experimental --type comparative --confidence 0.95
                        
                        # Step 4: Generate performance report
                        escai export --sessions baseline,experimental --format comparative_report
                    """).strip(),
                    sample_output=self._get_sample_output('workflow', 'performance_study'),
                    explanation="This workflow enables rigorous comparison of agent performance with statistical validation",
                    use_case="A/B testing of agent configurations for research publication",
                    prerequisites=[
                        "Multiple agent configurations ready",
                        "Controlled experimental environment",
                        "Performance metrics defined"
                    ],
                    follow_up_actions=[
                        "Validate statistical significance",
                        "Document experimental conditions",
                        "Prepare results section with confidence intervals"
                    ]
                )
            ]
        }
        
        return workflows.get(workflow_type, [])
    
    def _get_sample_output(self, command: str, output_key: str) -> str:
        """Get sample output for a command and output key."""
        if command in self._sample_outputs and output_key in self._sample_outputs[command]:
            return self._sample_outputs[command][output_key]
        return "Sample output not available"
    
    def _initialize_example_templates(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Initialize example templates for all commands."""
        return {
            "monitor": {
                "basic": [
                    {
                        "title": "Basic Agent Monitoring",
                        "description": "Start monitoring a single LangChain agent with default settings",
                        "command": "escai monitor --framework langchain --agents my_agent",
                        "output_key": "basic_monitoring",
                        "explanation": "This command starts monitoring 'my_agent' using the LangChain framework with default output format",
                        "use_case": "Initial exploration of agent behavior during development",
                        "prerequisites": ["LangChain agent instrumented with ESCAI", "Agent 'my_agent' running"],
                        "follow_up_actions": ["Review monitoring output", "Stop monitoring when complete", "Analyze collected data"]
                    },
                    {
                        "title": "Live Monitoring with Real-time Updates",
                        "description": "Monitor agents with live dashboard updates",
                        "command": "escai monitor --framework autogen --agents all --live",
                        "output_key": "live_monitoring",
                        "explanation": "Enables real-time monitoring dashboard showing agent status, conversations, and epistemic states as they happen",
                        "use_case": "Real-time observation during agent experiments",
                        "prerequisites": ["AutoGen agents running", "Terminal with Rich support"],
                        "follow_up_actions": ["Observe agent interactions", "Note interesting patterns", "Save session for later analysis"]
                    }
                ],
                "advanced": [
                    {
                        "title": "Multi-Framework Monitoring with Filtering",
                        "description": "Monitor multiple frameworks simultaneously with event filtering",
                        "command": "escai monitor --framework langchain,autogen --filter 'type:decision OR type:error' --output json",
                        "output_key": "filtered_monitoring",
                        "explanation": "Monitors both LangChain and AutoGen agents, filtering for decision and error events, outputting structured JSON",
                        "use_case": "Focused analysis of decision-making and error patterns across frameworks",
                        "prerequisites": ["Multiple frameworks instrumented", "Understanding of event types"],
                        "follow_up_actions": ["Analyze JSON output", "Compare decision patterns", "Investigate error correlations"]
                    }
                ],
                "research": [
                    {
                        "title": "Epistemic State Research Monitoring",
                        "description": "Specialized monitoring for epistemic state research",
                        "command": "escai monitor --framework crewai --epistemic --agents research_team --session epistemic_study_1",
                        "output_key": "epistemic_monitoring",
                        "explanation": "Focuses on epistemic state changes in CrewAI agents for research into agent cognition and belief formation",
                        "use_case": "Academic research into agent epistemic states and belief dynamics",
                        "prerequisites": ["CrewAI research team configured", "Research protocol defined"],
                        "follow_up_actions": ["Analyze belief evolution", "Correlate with task performance", "Prepare research findings"]
                    }
                ]
            },
            "analyze": {
                "basic": [
                    {
                        "title": "Basic Pattern Analysis",
                        "description": "Analyze behavioral patterns from monitoring data",
                        "command": "escai analyze --data session_123 --type patterns",
                        "output_key": "pattern_analysis",
                        "explanation": "Performs statistical analysis to identify recurring behavioral patterns in the specified session",
                        "use_case": "Understanding common agent behaviors and decision patterns",
                        "prerequisites": ["Completed monitoring session", "Sufficient data for analysis"],
                        "follow_up_actions": ["Review pattern significance", "Investigate interesting patterns", "Export results"]
                    }
                ],
                "advanced": [
                    {
                        "title": "Causal Analysis with Confidence Intervals",
                        "description": "Advanced causal inference analysis with statistical rigor",
                        "command": "escai analyze --data monitoring_data.json --type causal --confidence 0.99 --visualize",
                        "output_key": "causal_analysis",
                        "explanation": "Performs causal inference to identify cause-effect relationships with 99% confidence intervals and generates visualizations",
                        "use_case": "Research into causal relationships in agent behavior for publication",
                        "prerequisites": ["Large dataset with diverse events", "Understanding of causal inference"],
                        "follow_up_actions": ["Validate causal relationships", "Generate publication figures", "Document methodology"]
                    }
                ],
                "research": [
                    {
                        "title": "Longitudinal Performance Analysis",
                        "description": "Analyze agent performance evolution over time",
                        "command": "escai analyze --data sessions_1_to_10 --type longitudinal --agents learning_agent --timeframe 1w",
                        "output_key": "longitudinal_analysis",
                        "explanation": "Analyzes how agent performance changes over a week-long period across multiple sessions",
                        "use_case": "Research into agent learning and adaptation over time",
                        "prerequisites": ["Multiple sessions over time period", "Consistent agent configuration"],
                        "follow_up_actions": ["Identify learning trends", "Correlate with training data", "Model learning curves"]
                    }
                ]
            }
        }
    
    def _initialize_sample_outputs(self) -> Dict[str, Dict[str, str]]:
        """Initialize sample outputs for commands."""
        return {
            "monitor": {
                "basic_monitoring": textwrap.dedent("""
                    ╭─ ESCAI Agent Monitoring ─╮
                    │ Framework: LangChain      │
                    │ Agent: my_agent          │
                    │ Status: Active           │
                    │ Duration: 00:02:34       │
                    ╰─────────────────────────╯
                    
                    [14:23:45] Agent started task: "Analyze customer feedback"
                    [14:23:46] Epistemic state: beliefs={"customer_sentiment": "unknown"}
                    [14:23:47] Decision: Using sentiment analysis tool
                    [14:23:48] Tool result: sentiment=0.7 (positive)
                    [14:23:49] Epistemic update: beliefs={"customer_sentiment": "positive"}
                    [14:23:50] Task completed successfully
                    
                    Events captured: 15 | Decisions: 3 | Tool uses: 2
                """).strip(),
                "live_monitoring": textwrap.dedent("""
                    ╭─ Live Agent Dashboard ─╮
                    │ ● Agent_1: Processing  │
                    │ ● Agent_2: Waiting     │
                    │ ● Agent_3: Completed   │
                    ╰──────────────────────╯
                    
                    ╭─ Recent Activity ─╮
                    │ [14:25:01] Agent_1: Started reasoning about task complexity │
                    │ [14:25:02] Agent_2: Received message from Agent_1          │
                    │ [14:25:03] Agent_1: Decision - break task into subtasks    │
                    │ [14:25:04] Agent_3: Task delegation received               │
                    ╰─────────────────────────────────────────────────────────╯
                    
                    Statistics: 47 events/min | 12 decisions/min | 3 active agents
                """).strip(),
                "filtered_monitoring": textwrap.dedent("""
                    {
                      "session_id": "monitoring_session_456",
                      "events": [
                        {
                          "timestamp": "2024-01-15T14:23:45Z",
                          "type": "decision",
                          "agent": "langchain_agent_1",
                          "data": {
                            "decision": "use_search_tool",
                            "reasoning": "Need more information about topic",
                            "confidence": 0.85
                          }
                        },
                        {
                          "timestamp": "2024-01-15T14:23:47Z", 
                          "type": "error",
                          "agent": "autogen_agent_2",
                          "data": {
                            "error_type": "tool_timeout",
                            "message": "Search tool timed out after 30s",
                            "recovery_action": "retry_with_fallback"
                          }
                        }
                      ],
                      "summary": {
                        "total_events": 127,
                        "decisions": 45,
                        "errors": 8,
                        "frameworks": ["langchain", "autogen"]
                      }
                    }
                """).strip(),
                "epistemic_monitoring": textwrap.dedent("""
                    ╭─ Epistemic State Monitoring ─╮
                    │ Research Team: 3 agents      │
                    │ Session: epistemic_study_1    │
                    │ Focus: Belief Evolution       │
                    ╰─────────────────────────────╯
                    
                    Agent: researcher_1
                    ├─ Initial beliefs: {"hypothesis": "uncertain", "evidence": []}
                    ├─ [T+00:30] Evidence gathered: paper_1, paper_2
                    ├─ [T+01:15] Belief update: {"hypothesis": "likely_true", "confidence": 0.7}
                    ├─ [T+02:00] Conflicting evidence: paper_3
                    └─ [T+02:30] Belief revision: {"hypothesis": "needs_investigation", "confidence": 0.4}
                    
                    Epistemic Events: 23 | Belief Updates: 8 | Confidence Changes: 12
                """).strip()
            },
            "analyze": {
                "pattern_analysis": textwrap.dedent("""
                    ╭─ Behavioral Pattern Analysis ─╮
                    │ Session: session_123           │
                    │ Events Analyzed: 1,247         │
                    │ Patterns Found: 7              │
                    ╰──────────────────────────────╯
                    
                    Pattern 1: Sequential Tool Usage
                    ├─ Frequency: 89% of tasks
                    ├─ Sequence: search → analyze → summarize
                    ├─ Significance: p < 0.001 (highly significant)
                    └─ Description: Agent consistently follows this pattern for research tasks
                    
                    Pattern 2: Error Recovery Behavior  
                    ├─ Frequency: 12% of errors
                    ├─ Sequence: error → retry → fallback_tool
                    ├─ Significance: p < 0.05 (significant)
                    └─ Description: Systematic error recovery with fallback strategy
                    
                    Pattern 3: Collaborative Decision Making
                    ├─ Frequency: 67% of multi-agent tasks
                    ├─ Sequence: propose → discuss → consensus → execute
                    ├─ Significance: p < 0.01 (very significant)
                    └─ Description: Structured collaboration pattern in team tasks
                """).strip(),
                "causal_analysis": textwrap.dedent("""
                    ╭─ Causal Relationship Analysis ─╮
                    │ Confidence Level: 99%           │
                    │ Relationships Found: 12         │
                    │ Visualization: Generated        │
                    ╰───────────────────────────────╯
                    
                    Causal Relationship 1: Task Complexity → Tool Selection
                    ├─ Causal Effect: 0.73 (95% CI: 0.65-0.81)
                    ├─ P-value: < 0.001
                    ├─ Interpretation: Higher task complexity strongly predicts advanced tool usage
                    └─ Mechanism: Agents assess complexity before selecting appropriate tools
                    
                    Causal Relationship 2: Team Size → Communication Overhead
                    ├─ Causal Effect: 0.45 (95% CI: 0.32-0.58)
                    ├─ P-value: < 0.01
                    ├─ Interpretation: Larger teams require more coordination communication
                    └─ Mechanism: Exponential increase in coordination messages
                    
                    Visualization saved to: causal_graph_20240115.png
                """).strip(),
                "longitudinal_analysis": textwrap.dedent("""
                    ╭─ Longitudinal Performance Analysis ─╮
                    │ Agent: learning_agent                │
                    │ Time Period: 7 days                  │
                    │ Sessions: 10                         │
                    ╰────────────────────────────────────╯
                    
                    Performance Metrics Over Time:
                    
                    Day 1: Success Rate: 45% | Avg Time: 120s | Errors: 23
                    Day 2: Success Rate: 52% | Avg Time: 105s | Errors: 18
                    Day 3: Success Rate: 61% | Avg Time: 95s  | Errors: 14
                    Day 4: Success Rate: 68% | Avg Time: 87s  | Errors: 12
                    Day 5: Success Rate: 74% | Avg Time: 82s  | Errors: 9
                    Day 6: Success Rate: 79% | Avg Time: 78s  | Errors: 7
                    Day 7: Success Rate: 83% | Avg Time: 75s  | Errors: 5
                    
                    Learning Trends:
                    ├─ Success Rate: +8.4% improvement per day (R² = 0.94)
                    ├─ Task Time: -6.4s reduction per day (R² = 0.91)
                    ├─ Error Rate: -2.6 errors reduction per day (R² = 0.89)
                    └─ Learning Curve: Exponential improvement with plateau approaching
                    
                    Statistical Significance: All trends p < 0.001
                """).strip()
            },
            "workflow": {
                "behavioral_analysis": textwrap.dedent("""
                    ╭─ Complete Behavioral Analysis Results ─╮
                    │ Workflow: behavioral_analysis           │
                    │ Total Duration: 2h 34m                  │
                    │ Status: Completed Successfully          │
                    ╰───────────────────────────────────────╯
                    
                    Step 1: Monitoring Complete
                    ├─ Events Captured: 2,847
                    ├─ Agents Monitored: 3
                    ├─ Duration: 1h 45m
                    └─ Data Quality: Excellent (98.7% complete)
                    
                    Step 2: Pattern Analysis Complete
                    ├─ Patterns Identified: 15
                    ├─ Significant Patterns: 11 (p < 0.05)
                    ├─ Novel Patterns: 4
                    └─ Analysis Time: 23m
                    
                    Step 3: Export Complete
                    ├─ Format: Publication-ready LaTeX
                    ├─ Figures Generated: 8
                    ├─ Tables Generated: 5
                    └─ Files: behavioral_analysis_report.tex, figures/
                    
                    Ready for Publication: ✓
                    Reproducibility Score: 100%
                """).strip(),
                "performance_study": textwrap.dedent("""
                    ╭─ Agent Performance Comparison Study ─╮
                    │ Configurations: Baseline vs Experimental │
                    │ Statistical Power: 0.95                  │
                    │ Significance Level: α = 0.05             │
                    ╰─────────────────────────────────────────╯
                    
                    Baseline Configuration:
                    ├─ Success Rate: 72.3% (95% CI: 68.1-76.5%)
                    ├─ Avg Task Time: 94.2s (95% CI: 89.7-98.7s)
                    ├─ Error Rate: 8.4% (95% CI: 6.2-10.6%)
                    └─ Sample Size: n = 247 tasks
                    
                    Experimental Configuration:
                    ├─ Success Rate: 81.7% (95% CI: 77.8-85.6%)
                    ├─ Avg Task Time: 76.8s (95% CI: 72.9-80.7s)
                    ├─ Error Rate: 5.1% (95% CI: 3.4-6.8%)
                    └─ Sample Size: n = 251 tasks
                    
                    Statistical Comparison:
                    ├─ Success Rate Improvement: +9.4% (p < 0.001) ***
                    ├─ Time Reduction: -17.4s (p < 0.001) ***
                    ├─ Error Reduction: -3.3% (p < 0.01) **
                    └─ Effect Size: Cohen's d = 0.73 (large effect)
                    
                    Conclusion: Experimental configuration shows statistically 
                    significant improvement across all metrics.
                """).strip()
            }
        }