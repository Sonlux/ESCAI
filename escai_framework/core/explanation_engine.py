"""
Explanation Engine for the ESCAI framework.

This module provides human-readable explanations of agent behavior,
decision pathways, causal relationships, and predictions.
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from ..models.epistemic_state import EpistemicState, BeliefState, GoalState, GoalStatus
from ..models.behavioral_pattern import BehavioralPattern, ExecutionSequence, ExecutionStep, ExecutionStatus
from ..models.causal_relationship import CausalRelationship, CausalType, EvidenceType
from ..models.prediction_result import PredictionResult, RiskLevel, PredictionType


class ExplanationType(Enum):
    """Types of explanations that can be generated."""
    BEHAVIOR_SUMMARY = "behavior_summary"
    DECISION_PATHWAY = "decision_pathway"
    CAUSAL_EXPLANATION = "causal_explanation"
    PREDICTION_EXPLANATION = "prediction_explanation"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class ExplanationStyle(Enum):
    """Styles for explanation generation."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    SIMPLE = "simple"
    DETAILED = "detailed"


@dataclass
class ExplanationTemplate:
    """Template for generating explanations."""
    template_id: str
    explanation_type: ExplanationType
    style: ExplanationStyle
    template_text: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """Result of explanation generation."""
    explanation_id: str
    explanation_type: ExplanationType
    style: ExplanationStyle
    title: str
    content: str
    confidence_score: float  # 0.0 to 1.0
    coverage_score: float  # 0.0 to 1.0, how much of the data was explained
    supporting_evidence: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def validate(self) -> bool:
        """Validate the explanation result."""
        if not isinstance(self.explanation_id, str) or not self.explanation_id.strip():
            return False
        if not isinstance(self.explanation_type, ExplanationType):
            return False
        if not isinstance(self.style, ExplanationStyle):
            return False
        if not isinstance(self.title, str) or not self.title.strip():
            return False
        if not isinstance(self.content, str) or not self.content.strip():
            return False
        if not isinstance(self.confidence_score, (int, float)) or not 0.0 <= self.confidence_score <= 1.0:
            return False
        if not isinstance(self.coverage_score, (int, float)) or not 0.0 <= self.coverage_score <= 1.0:
            return False
        return True


class ExplanationEngine:
    """
    Engine for generating human-readable explanations of agent behavior.
    
    Provides natural language explanations for epistemic states, behavioral patterns,
    causal relationships, and predictions with configurable styles and detail levels.
    """
    
    def __init__(self):
        """Initialize the explanation engine with templates."""
        self.templates = self._initialize_templates()
        self.explanation_cache = {}
    
    def _initialize_templates(self) -> Dict[str, ExplanationTemplate]:
        """Initialize explanation templates."""
        templates = {}
        
        # Behavior summary templates
        templates["behavior_summary_simple"] = ExplanationTemplate(
            template_id="behavior_summary_simple",
            explanation_type=ExplanationType.BEHAVIOR_SUMMARY,
            style=ExplanationStyle.SIMPLE,
            template_text="The agent {agent_action} with {success_rate}% success rate. {key_patterns}",
            required_fields=["agent_action", "success_rate", "key_patterns"]
        )
        
        templates["behavior_summary_detailed"] = ExplanationTemplate(
            template_id="behavior_summary_detailed",
            explanation_type=ExplanationType.BEHAVIOR_SUMMARY,
            style=ExplanationStyle.DETAILED,
            template_text="Agent {agent_id} executed {total_steps} steps over {duration}. "
                         "The agent demonstrated {pattern_count} distinct behavioral patterns "
                         "with an overall success rate of {success_rate}%. "
                         "Key patterns include: {pattern_details}. "
                         "Common failure modes were: {failure_modes}.",
            required_fields=["agent_id", "total_steps", "duration", "pattern_count", 
                           "success_rate", "pattern_details", "failure_modes"]
        )
        
        # Decision pathway templates
        templates["decision_pathway_simple"] = ExplanationTemplate(
            template_id="decision_pathway_simple",
            explanation_type=ExplanationType.DECISION_PATHWAY,
            style=ExplanationStyle.SIMPLE,
            template_text="The agent decided to {decision} because {main_reason}. "
                         "This led to {outcome}.",
            required_fields=["decision", "main_reason", "outcome"]
        )
        
        templates["decision_pathway_detailed"] = ExplanationTemplate(
            template_id="decision_pathway_detailed",
            explanation_type=ExplanationType.DECISION_PATHWAY,
            style=ExplanationStyle.DETAILED,
            template_text="Decision pathway analysis for {agent_id}:\n\n"
                         "Initial State: {initial_beliefs}\n"
                         "Goal: {primary_goal}\n"
                         "Decision Process: {decision_steps}\n"
                         "Key Factors: {influencing_factors}\n"
                         "Confidence Level: {confidence}%\n"
                         "Outcome: {final_outcome}",
            required_fields=["agent_id", "initial_beliefs", "primary_goal", 
                           "decision_steps", "influencing_factors", "confidence", "final_outcome"]
        )
        
        # Causal explanation templates
        templates["causal_explanation_simple"] = ExplanationTemplate(
            template_id="causal_explanation_simple",
            explanation_type=ExplanationType.CAUSAL_EXPLANATION,
            style=ExplanationStyle.SIMPLE,
            template_text="{cause} caused {effect} with {confidence}% confidence. "
                         "This relationship was {strength} and occurred {delay} after the cause.",
            required_fields=["cause", "effect", "confidence", "strength", "delay"]
        )
        
        templates["causal_explanation_detailed"] = ExplanationTemplate(
            template_id="causal_explanation_detailed",
            explanation_type=ExplanationType.CAUSAL_EXPLANATION,
            style=ExplanationStyle.DETAILED,
            template_text="Causal Analysis:\n\n"
                         "Cause: {cause_description}\n"
                         "Effect: {effect_description}\n"
                         "Relationship Type: {causal_type}\n"
                         "Strength: {strength_description}\n"
                         "Time Delay: {delay_description}\n"
                         "Evidence: {evidence_summary}\n"
                         "Confidence: {confidence}%\n"
                         "Mechanism: {causal_mechanism}",
            required_fields=["cause_description", "effect_description", "causal_type",
                           "strength_description", "delay_description", "evidence_summary",
                           "confidence", "causal_mechanism"]
        )
        
        # Prediction explanation templates
        templates["prediction_explanation_simple"] = ExplanationTemplate(
            template_id="prediction_explanation_simple",
            explanation_type=ExplanationType.PREDICTION_EXPLANATION,
            style=ExplanationStyle.SIMPLE,
            template_text="The agent has a {prediction}% chance of {outcome}. "
                         "Risk level is {risk_level}. {main_risk_factor}",
            required_fields=["prediction", "outcome", "risk_level", "main_risk_factor"]
        )
        
        templates["prediction_explanation_detailed"] = ExplanationTemplate(
            template_id="prediction_explanation_detailed",
            explanation_type=ExplanationType.PREDICTION_EXPLANATION,
            style=ExplanationStyle.DETAILED,
            template_text="Prediction Analysis for {agent_id}:\n\n"
                         "Prediction: {prediction_value} ({prediction_type})\n"
                         "Confidence: {confidence}%\n"
                         "Risk Level: {risk_level}\n"
                         "Key Risk Factors:\n{risk_factors}\n"
                         "Recommended Actions:\n{interventions}\n"
                         "Model: {model_info}",
            required_fields=["agent_id", "prediction_value", "prediction_type",
                           "confidence", "risk_level", "risk_factors", "interventions", "model_info"]
        )
        
        return templates
    
    async def explain_behavior(
        self,
        behavioral_patterns: List[BehavioralPattern],
        execution_sequences: List[ExecutionSequence],
        style: ExplanationStyle = ExplanationStyle.SIMPLE
    ) -> ExplanationResult:
        """Generate explanation for agent behavioral patterns."""
        if not behavioral_patterns and not execution_sequences:
            return self._create_empty_explanation(ExplanationType.BEHAVIOR_SUMMARY, style)
        
        # Calculate summary statistics
        total_sequences = len(execution_sequences)
        total_steps = sum(len(seq.steps) for seq in execution_sequences)
        avg_success_rate = sum(pattern.success_rate for pattern in behavioral_patterns) / len(behavioral_patterns) if behavioral_patterns else 0
        
        # Extract key patterns
        key_patterns = []
        failure_modes = []
        
        for pattern in behavioral_patterns[:3]:  # Top 3 patterns
            key_patterns.append(f"{pattern.pattern_name} (occurs {pattern.frequency} times)")
            failure_modes.extend(pattern.failure_modes[:2])  # Top 2 failure modes per pattern
        
        # Generate explanation based on style
        if style == ExplanationStyle.SIMPLE:
            template = self.templates["behavior_summary_simple"]
            agent_action = "executed various tasks" if total_sequences > 1 else "executed a task"
            key_patterns_text = ", ".join(key_patterns[:2]) if key_patterns else "no clear patterns identified"
            
            content = template.template_text.format(
                agent_action=agent_action,
                success_rate=int(avg_success_rate * 100),
                key_patterns=key_patterns_text
            )
        else:
            template = self.templates["behavior_summary_detailed"]
            agent_id = execution_sequences[0].agent_id if execution_sequences else "unknown"
            duration = self._calculate_total_duration(execution_sequences)
            pattern_details = "; ".join(key_patterns) if key_patterns else "No significant patterns"
            failure_modes_text = ", ".join(set(failure_modes[:5])) if failure_modes else "No common failure modes"
            
            content = template.template_text.format(
                agent_id=agent_id,
                total_steps=total_steps,
                duration=duration,
                pattern_count=len(behavioral_patterns),
                success_rate=int(avg_success_rate * 100),
                pattern_details=pattern_details,
                failure_modes=failure_modes_text
            )
        
        # Calculate confidence and coverage
        confidence_score = min(0.9, len(behavioral_patterns) * 0.2 + 0.3)  # More patterns = higher confidence
        coverage_score = min(1.0, total_sequences / 10.0)  # Coverage based on data volume
        
        return ExplanationResult(
            explanation_id=f"behavior_{datetime.utcnow().timestamp()}",
            explanation_type=ExplanationType.BEHAVIOR_SUMMARY,
            style=style,
            title="Agent Behavior Analysis",
            content=content,
            confidence_score=confidence_score,
            coverage_score=coverage_score,
            supporting_evidence=[f"{len(behavioral_patterns)} patterns analyzed", 
                               f"{total_sequences} execution sequences reviewed"],
            limitations=["Analysis based on available execution data only"] if total_sequences < 5 else []
        )
    
    async def explain_decision_pathway(
        self,
        epistemic_states: List[EpistemicState],
        execution_sequence: ExecutionSequence,
        style: ExplanationStyle = ExplanationStyle.SIMPLE
    ) -> ExplanationResult:
        """Generate explanation for agent decision pathways."""
        if not epistemic_states or not execution_sequence:
            return self._create_empty_explanation(ExplanationType.DECISION_PATHWAY, style)
        
        initial_state = epistemic_states[0]
        final_state = epistemic_states[-1] if len(epistemic_states) > 1 else initial_state
        
        # Extract decision information
        primary_goal = self._extract_primary_goal(initial_state)
        decision_steps = self._extract_decision_steps(execution_sequence)
        influencing_factors = self._extract_influencing_factors(epistemic_states)
        
        if style == ExplanationStyle.SIMPLE:
            template = self.templates["decision_pathway_simple"]
            main_decision = decision_steps[0] if decision_steps else "proceed with task"
            main_reason = influencing_factors[0] if influencing_factors else "following standard procedure"
            outcome = "success" if execution_sequence.success_rate > 0.7 else "mixed results"
            
            content = template.template_text.format(
                decision=main_decision,
                main_reason=main_reason,
                outcome=outcome
            )
        else:
            template = self.templates["decision_pathway_detailed"]
            initial_beliefs = self._summarize_beliefs(initial_state.belief_states[:3])
            decision_steps_text = "; ".join(decision_steps[:5])
            influencing_factors_text = ", ".join(influencing_factors[:3])
            confidence = int(initial_state.confidence_level * 100)
            final_outcome = self._describe_outcome(execution_sequence)
            
            content = template.template_text.format(
                agent_id=initial_state.agent_id,
                initial_beliefs=initial_beliefs,
                primary_goal=primary_goal,
                decision_steps=decision_steps_text,
                influencing_factors=influencing_factors_text,
                confidence=confidence,
                final_outcome=final_outcome
            )
        
        confidence_score = min(0.9, len(epistemic_states) * 0.15 + 0.4)
        coverage_score = min(1.0, len(execution_sequence.steps) / 8.0)
        
        return ExplanationResult(
            explanation_id=f"decision_{datetime.utcnow().timestamp()}",
            explanation_type=ExplanationType.DECISION_PATHWAY,
            style=style,
            title="Decision Pathway Analysis",
            content=content,
            confidence_score=confidence_score,
            coverage_score=coverage_score,
            supporting_evidence=[f"{len(epistemic_states)} epistemic states analyzed",
                               f"{len(execution_sequence.steps)} execution steps reviewed"],
            limitations=["Analysis based on captured epistemic states only"]
        )
    
    async def explain_causal_relationship(
        self,
        causal_relationship: CausalRelationship,
        style: ExplanationStyle = ExplanationStyle.SIMPLE
    ) -> ExplanationResult:
        """Generate explanation for causal relationships."""
        if not causal_relationship:
            return self._create_empty_explanation(ExplanationType.CAUSAL_EXPLANATION, style)
        
        if style == ExplanationStyle.SIMPLE:
            template = self.templates["causal_explanation_simple"]
            cause = causal_relationship.cause_event.description
            effect = causal_relationship.effect_event.description
            confidence = int(causal_relationship.confidence * 100)
            strength = self._describe_strength(causal_relationship.strength)
            delay = self._describe_delay(causal_relationship.delay_ms)
            
            content = template.template_text.format(
                cause=cause,
                effect=effect,
                confidence=confidence,
                strength=strength,
                delay=delay
            )
        else:
            template = self.templates["causal_explanation_detailed"]
            evidence_summary = self._summarize_evidence(causal_relationship.evidence)
            mechanism = causal_relationship.causal_mechanism or "Mechanism not identified"
            
            content = template.template_text.format(
                cause_description=causal_relationship.cause_event.description,
                effect_description=causal_relationship.effect_event.description,
                causal_type=causal_relationship.causal_type.value.replace('_', ' ').title(),
                strength_description=self._describe_strength_detailed(causal_relationship.strength),
                delay_description=self._describe_delay_detailed(causal_relationship.delay_ms),
                evidence_summary=evidence_summary,
                confidence=int(causal_relationship.confidence * 100),
                causal_mechanism=mechanism
            )
        
        confidence_score = causal_relationship.confidence
        coverage_score = min(1.0, len(causal_relationship.evidence) / 3.0)
        
        return ExplanationResult(
            explanation_id=f"causal_{datetime.utcnow().timestamp()}",
            explanation_type=ExplanationType.CAUSAL_EXPLANATION,
            style=style,
            title="Causal Relationship Analysis",
            content=content,
            confidence_score=confidence_score,
            coverage_score=coverage_score,
            supporting_evidence=[f"{len(causal_relationship.evidence)} pieces of evidence",
                               f"Statistical significance: {causal_relationship.statistical_significance:.3f}"],
            limitations=["Correlation does not imply causation"] if causal_relationship.confidence < 0.7 else []
        )
    
    async def explain_prediction(
        self,
        prediction_result: PredictionResult,
        style: ExplanationStyle = ExplanationStyle.SIMPLE
    ) -> ExplanationResult:
        """Generate explanation for predictions with uncertainty bounds."""
        if not prediction_result:
            return self._create_empty_explanation(ExplanationType.PREDICTION_EXPLANATION, style)
        
        if style == ExplanationStyle.SIMPLE:
            template = self.templates["prediction_explanation_simple"]
            prediction_value = self._format_prediction_value(prediction_result)
            outcome = self._describe_prediction_outcome(prediction_result.prediction_type)
            risk_level = prediction_result.risk_level.value
            main_risk = prediction_result.risk_factors[0].name if prediction_result.risk_factors else "no significant risks identified"
            
            content = template.template_text.format(
                prediction=prediction_value,
                outcome=outcome,
                risk_level=risk_level,
                main_risk_factor=f"Main concern: {main_risk}"
            )
        else:
            template = self.templates["prediction_explanation_detailed"]
            risk_factors_text = self._format_risk_factors(prediction_result.risk_factors)
            interventions_text = self._format_interventions(prediction_result.recommended_interventions)
            model_info = f"{prediction_result.model_name} v{prediction_result.model_version}"
            
            content = template.template_text.format(
                agent_id=prediction_result.agent_id,
                prediction_value=self._format_prediction_value_detailed(prediction_result),
                prediction_type=prediction_result.prediction_type.value.replace('_', ' ').title(),
                confidence=int(prediction_result.confidence_score * 100),
                risk_level=prediction_result.risk_level.value.title(),
                risk_factors=risk_factors_text,
                interventions=interventions_text,
                model_info=model_info
            )
        
        confidence_score = prediction_result.confidence_score
        coverage_score = min(1.0, len(prediction_result.features_used) / 10.0)
        
        limitations = []
        if prediction_result.confidence_score < 0.6:
            limitations.append("Low prediction confidence")
        if prediction_result.is_expired():
            limitations.append("Prediction may be outdated")
        
        return ExplanationResult(
            explanation_id=f"prediction_{datetime.utcnow().timestamp()}",
            explanation_type=ExplanationType.PREDICTION_EXPLANATION,
            style=style,
            title="Prediction Analysis",
            content=content,
            confidence_score=confidence_score,
            coverage_score=coverage_score,
            supporting_evidence=[f"Based on {len(prediction_result.features_used)} features",
                               f"Risk assessment: {len(prediction_result.risk_factors)} factors"],
            limitations=limitations
        )
    
    async def compare_success_failure(
        self,
        successful_sequences: List[ExecutionSequence],
        failed_sequences: List[ExecutionSequence],
        style: ExplanationStyle = ExplanationStyle.SIMPLE
    ) -> ExplanationResult:
        """Generate comparative analysis between successful and failed attempts."""
        if not successful_sequences and not failed_sequences:
            return self._create_empty_explanation(ExplanationType.COMPARATIVE_ANALYSIS, style)
        
        # Analyze differences
        success_patterns = self._extract_common_patterns(successful_sequences)
        failure_patterns = self._extract_common_patterns(failed_sequences)
        
        key_differences = self._identify_key_differences(success_patterns, failure_patterns)
        success_factors = self._identify_success_factors(successful_sequences)
        failure_factors = self._identify_failure_factors(failed_sequences)
        
        if style == ExplanationStyle.SIMPLE:
            content = f"Successful attempts ({len(successful_sequences)}) vs Failed attempts ({len(failed_sequences)}):\n\n"
            content += f"Key success factors: {', '.join(success_factors[:3])}\n"
            content += f"Common failure causes: {', '.join(failure_factors[:3])}\n"
            content += f"Main difference: {key_differences[0] if key_differences else 'No clear pattern identified'}"
        else:
            content = f"Comparative Analysis: Success vs Failure\n\n"
            content += f"Successful Executions: {len(successful_sequences)}\n"
            content += f"Failed Executions: {len(failed_sequences)}\n\n"
            content += f"Success Patterns:\n"
            for i, pattern in enumerate(success_patterns[:5], 1):
                content += f"  {i}. {pattern}\n"
            content += f"\nFailure Patterns:\n"
            for i, pattern in enumerate(failure_patterns[:5], 1):
                content += f"  {i}. {pattern}\n"
            content += f"\nKey Differentiators:\n"
            for i, diff in enumerate(key_differences[:3], 1):
                content += f"  {i}. {diff}\n"
            content += f"\nRecommendations:\n"
            recommendations = self._generate_recommendations(success_factors, failure_factors)
            for i, rec in enumerate(recommendations[:3], 1):
                content += f"  {i}. {rec}\n"
        
        total_sequences = len(successful_sequences) + len(failed_sequences)
        confidence_score = min(0.9, total_sequences / 20.0 + 0.3)
        coverage_score = min(1.0, min(len(successful_sequences), len(failed_sequences)) / 5.0)
        
        return ExplanationResult(
            explanation_id=f"comparison_{datetime.utcnow().timestamp()}",
            explanation_type=ExplanationType.COMPARATIVE_ANALYSIS,
            style=style,
            title="Success vs Failure Analysis",
            content=content,
            confidence_score=confidence_score,
            coverage_score=coverage_score,
            supporting_evidence=[f"{len(successful_sequences)} successful sequences",
                               f"{len(failed_sequences)} failed sequences"],
            limitations=["Analysis quality depends on sample size"] if total_sequences < 10 else []
        )
    
    def _create_empty_explanation(
        self,
        explanation_type: ExplanationType,
        style: ExplanationStyle
    ) -> ExplanationResult:
        """Create an explanation for when no data is available."""
        return ExplanationResult(
            explanation_id=f"empty_{datetime.utcnow().timestamp()}",
            explanation_type=explanation_type,
            style=style,
            title="No Data Available",
            content="Insufficient data available for analysis.",
            confidence_score=0.0,
            coverage_score=0.0,
            limitations=["No data provided for analysis"]
        )
    
    def _calculate_total_duration(self, sequences: List[ExecutionSequence]) -> str:
        """Calculate and format total duration of execution sequences."""
        total_ms = sum(seq.total_duration_ms for seq in sequences)
        if total_ms < 1000:
            return f"{total_ms}ms"
        elif total_ms < 60000:
            return f"{total_ms/1000:.1f}s"
        else:
            return f"{total_ms/60000:.1f}min"
    
    def _extract_primary_goal(self, epistemic_state: EpistemicState) -> str:
        """Extract the primary goal from epistemic state."""
        active_goals = [g for g in epistemic_state.goal_states if g.status == GoalStatus.ACTIVE]
        if active_goals:
            # Return highest priority active goal
            primary = max(active_goals, key=lambda g: g.priority)
            return primary.description
        elif epistemic_state.goal_states:
            return epistemic_state.goal_states[0].description
        return "No clear goal identified"
    
    def _extract_decision_steps(self, sequence: ExecutionSequence) -> List[str]:
        """Extract key decision steps from execution sequence."""
        decision_steps = []
        for step in sequence.steps:
            if any(keyword in step.action.lower() for keyword in ['decide', 'choose', 'select', 'determine']):
                decision_steps.append(step.action)
        return decision_steps or [step.action for step in sequence.steps[:3]]
    
    def _extract_influencing_factors(self, epistemic_states: List[EpistemicState]) -> List[str]:
        """Extract factors that influenced decisions."""
        factors = []
        for state in epistemic_states:
            # High confidence beliefs are likely influencing factors
            high_conf_beliefs = [b for b in state.belief_states if b.confidence > 0.7]
            factors.extend([b.content for b in high_conf_beliefs[:2]])
        return list(set(factors))[:5]  # Remove duplicates, limit to 5
    
    def _summarize_beliefs(self, beliefs: List[BeliefState]) -> str:
        """Summarize belief states into readable text."""
        if not beliefs:
            return "No clear beliefs identified"
        
        summaries = []
        for belief in beliefs:
            conf_desc = "strongly" if belief.confidence > 0.8 else "moderately" if belief.confidence > 0.5 else "weakly"
            summaries.append(f"{conf_desc} believes {belief.content}")
        
        return "; ".join(summaries)
    
    def _describe_outcome(self, sequence: ExecutionSequence) -> str:
        """Describe the outcome of an execution sequence."""
        if sequence.success_rate > 0.8:
            return f"Successful completion with {len(sequence.steps)} steps"
        elif sequence.success_rate > 0.5:
            return f"Partial success with some issues in {len(sequence.steps)} steps"
        else:
            return f"Failed execution with multiple issues in {len(sequence.steps)} steps"
    
    def _describe_strength(self, strength: float) -> str:
        """Describe causal relationship strength."""
        if strength > 0.8:
            return "very strong"
        elif strength > 0.6:
            return "strong"
        elif strength > 0.4:
            return "moderate"
        elif strength > 0.2:
            return "weak"
        else:
            return "very weak"
    
    def _describe_strength_detailed(self, strength: float) -> str:
        """Provide detailed description of causal relationship strength."""
        strength_desc = self._describe_strength(strength)
        return f"{strength_desc} (strength score: {strength:.2f})"
    
    def _describe_delay(self, delay_ms: int) -> str:
        """Describe time delay in human-readable format."""
        if delay_ms < 100:
            return "immediately"
        elif delay_ms < 1000:
            return f"{delay_ms}ms later"
        elif delay_ms < 60000:
            return f"{delay_ms/1000:.1f}s later"
        else:
            return f"{delay_ms/60000:.1f}min later"
    
    def _describe_delay_detailed(self, delay_ms: int) -> str:
        """Provide detailed description of time delay."""
        delay_desc = self._describe_delay(delay_ms)
        return f"{delay_desc} (delay: {delay_ms}ms)"
    
    def _summarize_evidence(self, evidence_list) -> str:
        """Summarize causal evidence."""
        if not evidence_list:
            return "No supporting evidence available"
        
        evidence_types = {}
        for evidence in evidence_list:
            ev_type = evidence.evidence_type.value
            if ev_type not in evidence_types:
                evidence_types[ev_type] = []
            evidence_types[ev_type].append(evidence.strength)
        
        summaries = []
        for ev_type, strengths in evidence_types.items():
            avg_strength = sum(strengths) / len(strengths)
            summaries.append(f"{ev_type} evidence (strength: {avg_strength:.2f})")
        
        return "; ".join(summaries)
    
    def _format_prediction_value(self, prediction: PredictionResult) -> str:
        """Format prediction value for simple display."""
        if prediction.prediction_type == PredictionType.SUCCESS_PROBABILITY:
            return f"{int(prediction.predicted_value * 100)}"
        elif prediction.prediction_type == PredictionType.COMPLETION_TIME:
            return f"{prediction.predicted_value:.1f}"
        else:
            return f"{prediction.predicted_value:.2f}"
    
    def _format_prediction_value_detailed(self, prediction: PredictionResult) -> str:
        """Format prediction value with confidence interval."""
        base_value = self._format_prediction_value(prediction)
        if prediction.confidence_interval:
            ci = prediction.confidence_interval
            return f"{base_value} (95% CI: {ci.lower_bound:.2f}-{ci.upper_bound:.2f})"
        return base_value
    
    def _describe_prediction_outcome(self, prediction_type: PredictionType) -> str:
        """Describe what the prediction is about."""
        type_descriptions = {
            PredictionType.SUCCESS_PROBABILITY: "success",
            PredictionType.COMPLETION_TIME: "completion",
            PredictionType.FAILURE_RISK: "failure",
            PredictionType.PERFORMANCE_SCORE: "good performance",
            PredictionType.RESOURCE_USAGE: "resource efficiency"
        }
        return type_descriptions.get(prediction_type, "positive outcome")
    
    def _format_risk_factors(self, risk_factors) -> str:
        """Format risk factors for display."""
        if not risk_factors:
            return "No significant risk factors identified"
        
        formatted = []
        for rf in risk_factors[:5]:  # Top 5 risk factors
            risk_score = rf.calculate_risk_score()
            formatted.append(f"• {rf.name}: {risk_score:.2f} risk score")
        
        return "\n".join(formatted)
    
    def _format_interventions(self, interventions) -> str:
        """Format recommended interventions."""
        if not interventions:
            return "No specific interventions recommended"
        
        formatted = []
        for intervention in interventions[:3]:  # Top 3 interventions
            formatted.append(f"• {intervention.name}: {intervention.description}")
        
        return "\n".join(formatted)
    
    def _extract_common_patterns(self, sequences: List[ExecutionSequence]) -> List[str]:
        """Extract common patterns from execution sequences."""
        if not sequences:
            return []
        
        patterns = []
        
        # Analyze action patterns
        action_counts = {}
        for seq in sequences:
            for step in seq.steps:
                action = step.action.lower()
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # Find most common actions
        common_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for action, count in common_actions:
            patterns.append(f"Frequently uses '{action}' ({count} times)")
        
        # Analyze timing patterns
        avg_duration = sum(seq.total_duration_ms for seq in sequences) / len(sequences)
        if avg_duration < 5000:
            patterns.append("Executes quickly (under 5 seconds)")
        elif avg_duration > 30000:
            patterns.append("Takes time for careful execution (over 30 seconds)")
        
        # Analyze step count patterns
        avg_steps = sum(len(seq.steps) for seq in sequences) / len(sequences)
        if avg_steps < 3:
            patterns.append("Uses simple, direct approaches")
        elif avg_steps > 8:
            patterns.append("Uses complex, multi-step approaches")
        
        return patterns
    
    def _identify_key_differences(self, success_patterns: List[str], failure_patterns: List[str]) -> List[str]:
        """Identify key differences between success and failure patterns."""
        differences = []
        
        # Simple pattern comparison
        success_keywords = set()
        failure_keywords = set()
        
        for pattern in success_patterns:
            success_keywords.update(pattern.lower().split())
        
        for pattern in failure_patterns:
            failure_keywords.update(pattern.lower().split())
        
        # Find unique keywords
        success_only = success_keywords - failure_keywords
        failure_only = failure_keywords - success_keywords
        
        if success_only:
            differences.append(f"Successful attempts tend to involve: {', '.join(list(success_only)[:3])}")
        
        if failure_only:
            differences.append(f"Failed attempts are characterized by: {', '.join(list(failure_only)[:3])}")
        
        if not differences:
            differences.append("No clear distinguishing patterns identified")
        
        return differences
    
    def _identify_success_factors(self, sequences: List[ExecutionSequence]) -> List[str]:
        """Identify factors that contribute to success."""
        if not sequences:
            return []
        
        factors = []
        
        # Analyze successful sequences
        high_success = [seq for seq in sequences if seq.success_rate > 0.8]
        if high_success:
            avg_steps = sum(len(seq.steps) for seq in high_success) / len(high_success)
            avg_duration = sum(seq.total_duration_ms for seq in high_success) / len(high_success)
            
            if avg_steps < 5:
                factors.append("keeping execution simple")
            if avg_duration < 10000:
                factors.append("quick decision making")
            
            # Look for common successful actions
            action_counts = {}
            for seq in high_success:
                for step in seq.steps:
                    if step.status == ExecutionStatus.SUCCESS:
                        action_counts[step.action] = action_counts.get(step.action, 0) + 1
            
            if action_counts:
                top_action = max(action_counts.items(), key=lambda x: x[1])[0]
                factors.append(f"effective use of '{top_action}'")
        
        return factors or ["consistent execution approach"]
    
    def _identify_failure_factors(self, sequences: List[ExecutionSequence]) -> List[str]:
        """Identify factors that contribute to failure."""
        if not sequences:
            return []
        
        factors = []
        
        # Analyze failed sequences
        low_success = [seq for seq in sequences if seq.success_rate < 0.5]
        if low_success:
            # Look for common failure patterns
            error_patterns = {}
            for seq in low_success:
                for step in seq.steps:
                    if step.status in [ExecutionStatus.FAILURE, ExecutionStatus.TIMEOUT]:
                        if step.error_message:
                            error_patterns[step.error_message] = error_patterns.get(step.error_message, 0) + 1
                        else:
                            error_patterns[step.action] = error_patterns.get(step.action, 0) + 1
            
            if error_patterns:
                top_error = max(error_patterns.items(), key=lambda x: x[1])[0]
                factors.append(f"issues with '{top_error}'")
            
            avg_duration = sum(seq.total_duration_ms for seq in low_success) / len(low_success)
            if avg_duration > 30000:
                factors.append("excessive execution time")
        
        return factors or ["inconsistent execution patterns"]
    
    def _generate_recommendations(self, success_factors: List[str], failure_factors: List[str]) -> List[str]:
        """Generate recommendations based on success and failure analysis."""
        recommendations = []
        
        if success_factors:
            recommendations.append(f"Focus on {success_factors[0]} as it correlates with success")
        
        if failure_factors:
            recommendations.append(f"Address {failure_factors[0]} to reduce failure rates")
        
        recommendations.append("Monitor execution patterns for early intervention opportunities")
        
        return recommendations
    
    async def generate_explanation(
        self,
        agent_id: str,
        behavior_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        explanation_type: str = "comprehensive",
        max_length: int = 500
    ) -> Dict[str, Any]:
        """Generate human-readable explanation of agent behavior."""
        try:
            # In a real implementation, this would gather data and generate explanations
            # For now, return a dummy explanation
            return {
                "agent_id": agent_id,
                "explanation_type": explanation_type,
                "explanation": f"Agent {agent_id} has been operating within normal parameters during the specified time period.",
                "confidence": 0.8,
                "generated_at": datetime.utcnow().isoformat(),
                "length": max_length
            }
        except Exception as e:
            self.logger.error(f"Failed to generate explanation for agent {agent_id}: {e}")
            raise

    async def get_explanation_quality_metrics(self, explanation: ExplanationResult) -> Dict[str, float]:
        """Calculate quality metrics for an explanation."""
        metrics = {
            "confidence": explanation.confidence_score,
            "coverage": explanation.coverage_score,
            "completeness": len(explanation.supporting_evidence) / 5.0,  # Normalize to 0-1
            "clarity": self._calculate_clarity_score(explanation.content),
            "actionability": self._calculate_actionability_score(explanation.content)
        }
        
        # Overall quality score
        metrics["overall_quality"] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    def _calculate_clarity_score(self, content: str) -> float:
        """Calculate clarity score based on content characteristics."""
        # Simple heuristics for clarity
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Prefer moderate sentence length (10-20 words)
        if 10 <= avg_sentence_length <= 20:
            length_score = 1.0
        elif 5 <= avg_sentence_length <= 30:
            length_score = 0.7
        else:
            length_score = 0.4
        
        # Check for technical jargon (simple heuristic)
        technical_words = ['algorithm', 'heuristic', 'optimization', 'correlation', 'regression']
        jargon_count = sum(1 for word in technical_words if word in content.lower())
        jargon_score = max(0.3, 1.0 - (jargon_count * 0.2))
        
        return (length_score + jargon_score) / 2
    
    def _calculate_actionability_score(self, content: str) -> float:
        """Calculate actionability score based on presence of actionable insights."""
        actionable_keywords = [
            'recommend', 'suggest', 'should', 'could', 'improve', 'optimize',
            'avoid', 'focus', 'consider', 'try', 'implement', 'address'
        ]
        
        actionable_count = sum(1 for keyword in actionable_keywords if keyword in content.lower())
        return min(1.0, actionable_count / 3.0)  # Normalize to 0-1, max at 3 actionable elements