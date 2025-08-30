"""
Epistemic State Extraction Engine for the ESCAI framework.

This module implements the EpistemicExtractor class that extracts epistemic states
from agent execution logs using NLP techniques, transformers, and graph analysis.
"""

import re
import math
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict, Counter
import logging

try:
    import networkx as nx
    import numpy as np_module
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    # Graceful degradation if optional dependencies are not available
    logging.warning(f"Optional dependency not available: {e}")
    nx = None
    np_module = None
    pipeline = None
    AutoTokenizer = None
    AutoModel = None
    torch = None
    SentenceTransformer = None

from ..models.epistemic_state import (
    EpistemicState, BeliefState, KnowledgeState, GoalState,
    BeliefType, GoalStatus
)
from ..instrumentation.events import AgentEvent, EventType


class EpistemicExtractor:
    """
    Extracts epistemic states from agent execution logs using NLP and ML techniques.
    
    This class provides methods to:
    - Extract beliefs from natural language using transformers
    - Parse confidence scores from text
    - Build knowledge graphs from relationships
    - Track goal progression
    - Quantify uncertainty using entropy measures
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 sentence_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the epistemic extractor.
        
        Args:
            model_name: Name of the transformer model for NLP tasks
            sentence_model: Name of the sentence transformer model for embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.sentence_model_name = sentence_model
        
        # Initialize models if dependencies are available
        self._init_models()
        
        # Confidence patterns for regex extraction
        self.confidence_patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)[%\s]",
            r"certainty[:\s]+(\d+(?:\.\d+)?)[%\s]",
            r"probability[:\s]+(\d+(?:\.\d+)?)[%\s]",
            r"(\d+(?:\.\d+)?)[%\s]+confident",
            r"(\d+(?:\.\d+)?)[%\s]+certain",
            r"(\d+(?:\.\d+)?)[%\s]+sure",
            r"I am (\d+(?:\.\d+)?)[%\s]",
            r"believe with (\d+(?:\.\d+)?)[%\s]",
        ]
        
        # Belief type patterns
        self.belief_type_patterns = {
            BeliefType.FACTUAL: [
                r"fact", r"true", r"certain", r"definitely", r"absolutely",
                r"confirmed", r"verified", r"established"
            ],
            BeliefType.PROBABILISTIC: [
                r"probably", r"likely", r"possibly", r"maybe", r"perhaps",
                r"chance", r"probability", r"odds"
            ],
            BeliefType.CONDITIONAL: [
                r"if", r"when", r"unless", r"provided", r"assuming",
                r"given that", r"in case", r"depends on"
            ],
            BeliefType.TEMPORAL: [
                r"will", r"going to", r"future", r"eventually", r"soon",
                r"later", r"tomorrow", r"next", r"after"
            ]
        }
        
        # Goal status patterns
        self.goal_status_patterns = {
            GoalStatus.ACTIVE: [
                r"working on", r"pursuing", r"attempting", r"trying to",
                r"in progress", r"currently", r"focusing on"
            ],
            GoalStatus.COMPLETED: [
                r"completed", r"finished", r"done", r"accomplished",
                r"achieved", r"succeeded", r"fulfilled"
            ],
            GoalStatus.FAILED: [
                r"failed", r"unsuccessful", r"couldn't", r"unable to",
                r"gave up", r"abandoned", r"impossible"
            ],
            GoalStatus.SUSPENDED: [
                r"paused", r"postponed", r"delayed", r"on hold",
                r"suspended", r"temporarily stopped"
            ]
        }
    
    def _init_models(self):
        """Initialize NLP models if dependencies are available."""
        self.sentiment_analyzer = None
        self.tokenizer = None
        self.model = None
        self.sentence_model = None
        
        if pipeline is not None:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    return_all_scores=True
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize sentiment analyzer: {e}")
        
        if AutoTokenizer is not None and AutoModel is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            except Exception as e:
                self.logger.warning(f"Could not initialize transformer model: {e}")
        
        if SentenceTransformer is not None:
            try:
                self.sentence_model = SentenceTransformer(self.sentence_model_name)
            except Exception as e:
                self.logger.warning(f"Could not initialize sentence transformer: {e}")
    
    async def extract_beliefs(self, agent_logs: List[AgentEvent]) -> List[BeliefState]:
        """
        Extract belief states from agent logs using NLP analysis.
        
        Args:
            agent_logs: List of agent events to analyze
            
        Returns:
            List of extracted belief states
        """
        beliefs = []
        
        for event in agent_logs:
            # Focus on events that might contain beliefs
            if event.event_type in [
                EventType.DECISION_START, EventType.DECISION_COMPLETE,
                EventType.BELIEF_UPDATE, EventType.KNOWLEDGE_UPDATE
            ]:
                event_beliefs = await self._extract_beliefs_from_event(event)
                beliefs.extend(event_beliefs)
        
        return beliefs
    
    async def _extract_beliefs_from_event(self, event: AgentEvent) -> List[BeliefState]:
        """Extract beliefs from a single event."""
        beliefs = []
        text_content = f"{event.message} {event.data.get('content', '')}"
        
        if not text_content.strip():
            return beliefs
        
        # Split text into sentences for individual belief extraction
        sentences = self._split_into_sentences(text_content)
        
        for sentence in sentences:
            if self._is_belief_sentence(sentence):
                belief = await self._create_belief_from_sentence(sentence, event)
                if belief:
                    beliefs.append(belief)
        
        return beliefs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_belief_sentence(self, sentence: str) -> bool:
        """Determine if a sentence contains a belief statement."""
        belief_indicators = [
            r"i believe", r"i think", r"i assume", r"i suppose",
            r"it seems", r"it appears", r"likely", r"probably",
            r"i'm confident", r"i'm certain", r"i know", r"confident"
        ]
        
        sentence_lower = sentence.lower()
        return any(re.search(pattern, sentence_lower) for pattern in belief_indicators)
    
    async def _create_belief_from_sentence(self, sentence: str, event: AgentEvent) -> Optional[BeliefState]:
        """Create a belief state from a sentence."""
        try:
            # Extract confidence score
            confidence = self._extract_confidence_score(sentence)
            
            # Determine belief type
            belief_type = self._classify_belief_type(sentence)
            
            # Extract evidence from event data
            evidence = self._extract_evidence(event)
            
            return BeliefState(
                content=sentence.strip(),
                belief_type=belief_type,
                confidence=confidence,
                evidence=evidence,
                timestamp=event.timestamp,
                source=f"{event.framework}:{event.component}"
            )
        except Exception as e:
            self.logger.error(f"Error creating belief from sentence: {e}")
            return None
    
    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score from text using regex and transformers."""
        # First try regex patterns
        for pattern in self.confidence_patterns:
            match = re.search(pattern, text.lower())
            if match:
                score = float(match.group(1))
                # Convert percentage to decimal if needed
                if score > 1.0:
                    score = score / 100.0
                return min(max(score, 0.0), 1.0)
        
        # If no explicit confidence found, use sentiment analysis
        if self.sentiment_analyzer:
            try:
                results = self.sentiment_analyzer(text)
                if results and len(results[0]) > 0:
                    # Use the confidence of the most confident prediction
                    max_score = max(result['score'] for result in results[0])
                    return max_score
            except Exception as e:
                self.logger.warning(f"Error in sentiment analysis: {e}")
        
        # Default confidence for belief statements
        return 0.7
    
    def _classify_belief_type(self, text: str) -> BeliefType:
        """Classify the type of belief based on text patterns."""
        text_lower = text.lower()
        
        # Count matches for each belief type
        type_scores: Dict[BeliefType, int] = {}
        for belief_type, patterns in self.belief_type_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            type_scores[belief_type] = score
        
        # Return the type with the highest score, default to FACTUAL
        if type_scores:
            return max(type_scores, key=type_scores.get)
        return BeliefType.FACTUAL
    
    def _extract_evidence(self, event: AgentEvent) -> List[str]:
        """Extract evidence from event data."""
        evidence = []
        
        # Add event message as evidence
        if event.message:
            evidence.append(event.message)
        
        # Extract evidence from event data
        if 'evidence' in event.data:
            if isinstance(event.data['evidence'], list):
                evidence.extend(event.data['evidence'])
            else:
                evidence.append(str(event.data['evidence']))
        
        # Add tool responses as evidence
        if event.event_type == EventType.TOOL_RESPONSE and 'response' in event.data:
            evidence.append(str(event.data['response']))
        
        return evidence
    
    async def extract_knowledge(self, agent_logs: List[AgentEvent]) -> KnowledgeState:
        """
        Extract knowledge state from agent logs.
        
        Args:
            agent_logs: List of agent events to analyze
            
        Returns:
            Extracted knowledge state
        """
        facts: List[str] = []
        rules: List[str] = []
        concepts: Dict[str, Any] = {}
        relationships: List[Dict[str, str]] = []
        
        for event in agent_logs:
            if event.event_type == EventType.KNOWLEDGE_UPDATE:
                # Extract facts
                event_facts = self._extract_facts_from_event(event)
                facts.extend(event_facts)
                
                # Extract rules
                event_rules = self._extract_rules_from_event(event)
                rules.extend(event_rules)
                
                # Extract concepts
                event_concepts = self._extract_concepts_from_event(event)
                concepts.update(event_concepts)
                
                # Extract relationships
                event_relationships = self._extract_relationships_from_event(event)
                relationships.extend(event_relationships)
        
        # Calculate overall confidence
        confidence_score = self._calculate_knowledge_confidence(facts, rules, concepts)
        
        return KnowledgeState(
            facts=list(set(facts)),  # Remove duplicates
            rules=list(set(rules)),
            concepts=concepts,
            relationships=relationships,
            confidence_score=confidence_score
        )
    
    def _extract_facts_from_event(self, event: AgentEvent) -> List[str]:
        """Extract factual statements from an event."""
        facts = []
        text = f"{event.message} {event.data.get('content', '')}"
        
        # Look for factual patterns
        fact_patterns = [
            r"fact[:\s]+(.+?)(?:\.|$)",
            r"true that (.+?)(?:\.|$)",
            r"confirmed that (.+?)(?:\.|$)",
            r"established that (.+?)(?:\.|$)"
        ]
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            facts.extend([match.strip() for match in matches])
        
        return facts
    
    def _extract_rules_from_event(self, event: AgentEvent) -> List[str]:
        """Extract rules from an event."""
        rules = []
        text = f"{event.message} {event.data.get('content', '')}"
        
        # Look for rule patterns
        rule_patterns = [
            r"if (.+?) then (.+?)(?:\.|$)",
            r"when (.+?), (.+?)(?:\.|$)",
            r"rule[:\s]+(.+?)(?:\.|$)",
            r"always (.+?)(?:\.|$)",
            r"never (.+?)(?:\.|$)"
        ]
        
        for pattern in rule_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                rules.extend([f"{match[0]} -> {match[1]}" for match in matches])
            else:
                rules.extend([match.strip() for match in matches])
        
        return rules
    
    def _extract_concepts_from_event(self, event: AgentEvent) -> Dict[str, Any]:
        """Extract concepts from an event."""
        concepts = {}
        
        # Extract from event data if available
        if 'concepts' in event.data:
            concepts.update(event.data['concepts'])
        
        # Extract named entities (simplified approach)
        text = f"{event.message} {event.data.get('content', '')}"
        
        # Simple concept extraction based on capitalized words
        concept_candidates = re.findall(r'\b[A-Z][a-z]+\b', text)
        for concept in concept_candidates:
            if concept not in concepts:
                concepts[concept] = {"mentions": 1, "context": text[:100]}
            else:
                concepts[concept]["mentions"] += 1
        
        return concepts
    
    def _extract_relationships_from_event(self, event: AgentEvent) -> List[Dict[str, str]]:
        """Extract relationships from an event."""
        relationships = []
        text = f"{event.message} {event.data.get('content', '')}"
        
        # Look for relationship patterns
        relationship_patterns = [
            (r"(.+?) is a (.+?)(?:\.|$)", "is_a"),
            (r"(.+?) has (.+?)(?:\.|$)", "has"),
            (r"(.+?) causes (.+?)(?:\.|$)", "causes"),
            (r"(.+?) leads to (.+?)(?:\.|$)", "leads_to"),
            (r"(.+?) depends on (.+?)(?:\.|$)", "depends_on")
        ]
        
        for pattern, relation_type in relationship_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    "subject": match[0].strip(),
                    "predicate": relation_type,
                    "object": match[1].strip()
                })
        
        return relationships
    
    def _calculate_knowledge_confidence(self, facts: List[str], rules: List[str], 
                                      concepts: Dict[str, Any]) -> float:
        """Calculate overall confidence in the knowledge state."""
        if not facts and not rules and not concepts:
            return 0.0
        
        # Simple heuristic based on quantity and diversity
        fact_score = min(len(facts) / 10.0, 1.0)  # Normalize to max 10 facts
        rule_score = min(len(rules) / 5.0, 1.0)   # Normalize to max 5 rules
        concept_score = min(len(concepts) / 20.0, 1.0)  # Normalize to max 20 concepts
        
        return (fact_score + rule_score + concept_score) / 3.0
    
    async def extract_goals(self, agent_logs: List[AgentEvent]) -> List[GoalState]:
        """
        Extract goal states from agent logs.
        
        Args:
            agent_logs: List of agent events to analyze
            
        Returns:
            List of extracted goal states
        """
        goals = []
        goal_tracker: Dict[str, GoalState] = {}  # Track goals by description
        
        for event in agent_logs:
            if event.event_type in [
                EventType.TASK_START, EventType.TASK_COMPLETE, EventType.TASK_FAIL,
                EventType.GOAL_UPDATE
            ]:
                event_goals = await self._extract_goals_from_event(event, goal_tracker)
                goals.extend(event_goals)
        
        return list(goal_tracker.values())
    
    async def _extract_goals_from_event(self, event: AgentEvent, 
                                      goal_tracker: Dict[str, GoalState]) -> List[GoalState]:
        """Extract goals from a single event and update tracker."""
        new_goals = []
        text = f"{event.message} {event.data.get('content', '')}"
        
        # Extract goal descriptions
        goal_descriptions = self._extract_goal_descriptions(text)
        
        for description in goal_descriptions:
            # Determine goal status
            status = self._classify_goal_status(text, event.event_type)
            
            # Calculate progress
            progress = self._calculate_goal_progress(text, status)
            
            # Determine priority (simplified)
            priority = self._extract_goal_priority(text)
            
            if description in goal_tracker:
                # Update existing goal
                goal = goal_tracker[description]
                goal.status = status
                goal.progress = progress
            else:
                # Create new goal
                goal = GoalState(
                    description=description,
                    status=status,
                    priority=priority,
                    progress=progress,
                    created_at=event.timestamp
                )
                goal_tracker[description] = goal
                new_goals.append(goal)
        
        return new_goals
    
    def _extract_goal_descriptions(self, text: str) -> List[str]:
        """Extract goal descriptions from text."""
        goal_patterns = [
            r"goal[:\s]+(.+?)(?:\.|$)",
            r"objective[:\s]+(.+?)(?:\.|$)",
            r"aim to (.+?)(?:\.|$)",
            r"trying to (.+?)(?:\.|$)",
            r"want to (.+?)(?:\.|$)",
            r"need to (.+?)(?:\.|$)",
            r"task[:\s]+(.+?)(?:\.|$)"
        ]
        
        descriptions = []
        for pattern in goal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            descriptions.extend([match.strip() for match in matches])
        
        return descriptions
    
    def _classify_goal_status(self, text: str, event_type: EventType) -> GoalStatus:
        """Classify goal status based on text and event type."""
        text_lower = text.lower()
        
        # Event type mapping
        if event_type == EventType.TASK_COMPLETE:
            return GoalStatus.COMPLETED
        elif event_type == EventType.TASK_FAIL:
            return GoalStatus.FAILED
        elif event_type == EventType.TASK_START:
            return GoalStatus.ACTIVE
        
        # Pattern matching
        for status, patterns in self.goal_status_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                return status
        
        return GoalStatus.ACTIVE  # Default
    
    def _calculate_goal_progress(self, text: str, status: GoalStatus) -> float:
        """Calculate goal progress from text and status."""
        # Look for explicit progress indicators
        progress_patterns = [
            r"(\d+(?:\.\d+)?)[%\s]+complete",
            r"(\d+(?:\.\d+)?)[%\s]+done",
            r"progress[:\s]+(\d+(?:\.\d+)?)[%\s]"
        ]
        
        for pattern in progress_patterns:
            match = re.search(pattern, text.lower())
            if match:
                progress = float(match.group(1))
                if progress > 1.0:
                    progress = progress / 100.0
                return min(max(progress, 0.0), 1.0)
        
        # Status-based progress
        if status == GoalStatus.COMPLETED:
            return 1.0
        elif status == GoalStatus.FAILED:
            return 0.0
        elif status == GoalStatus.ACTIVE:
            return 0.5  # Assume halfway if active
        else:  # SUSPENDED
            return 0.3  # Some progress made
    
    def _extract_goal_priority(self, text: str) -> int:
        """Extract goal priority from text."""
        priority_patterns = {
            r"high priority": 8,
            r"medium priority": 5,
            r"low priority": 2,
            r"urgent": 9,
            r"critical": 10
        }
        
        text_lower = text.lower()
        
        # Check for explicit priority numbers
        match = re.search(r"priority[:\s]+(\d+)", text_lower)
        if match:
            priority = int(match.group(1))
            return min(max(priority, 1), 10)
        
        # Check for priority keywords
        for pattern, priority in priority_patterns.items():
            if re.search(pattern, text_lower):
                return priority
        
        return 5  # Default medium priority
    
    async def build_knowledge_graph(self, knowledge_state: KnowledgeState) -> Optional[Any]:
        """
        Build a knowledge graph from the knowledge state using NetworkX.
        
        Args:
            knowledge_state: The knowledge state to build graph from
            
        Returns:
            NetworkX graph object or None if NetworkX not available
        """
        if nx is None:
            self.logger.warning("NetworkX not available, cannot build knowledge graph")
            return None
        
        try:
            graph = nx.DiGraph()
            
            # Add concepts as nodes
            for concept, data in knowledge_state.concepts.items():
                graph.add_node(concept, **data)
            
            # Add relationships as edges
            for relationship in knowledge_state.relationships:
                subject = relationship.get("subject", "")
                obj = relationship.get("object", "")
                predicate = relationship.get("predicate", "related_to")
                
                if subject and obj:
                    graph.add_edge(subject, obj, relation=predicate)
            
            # Add facts as nodes connected to a central "facts" node
            if knowledge_state.facts:
                graph.add_node("FACTS", node_type="fact_collection")
                for i, fact in enumerate(knowledge_state.facts):
                    fact_node = f"fact_{i}"
                    graph.add_node(fact_node, content=fact, node_type="fact")
                    graph.add_edge("FACTS", fact_node, relation="contains")
            
            # Add rules as structured relationships
            for i, rule in enumerate(knowledge_state.rules):
                rule_node = f"rule_{i}"
                graph.add_node(rule_node, content=rule, node_type="rule")
                
                # Try to parse rule structure (if -> then)
                if " -> " in rule:
                    condition, consequence = rule.split(" -> ", 1)
                    condition_node = f"condition_{i}"
                    consequence_node = f"consequence_{i}"
                    
                    graph.add_node(condition_node, content=condition.strip(), node_type="condition")
                    graph.add_node(consequence_node, content=consequence.strip(), node_type="consequence")
                    
                    graph.add_edge(condition_node, consequence_node, relation="implies")
                    graph.add_edge(rule_node, condition_node, relation="has_condition")
                    graph.add_edge(rule_node, consequence_node, relation="has_consequence")
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Error building knowledge graph: {e}")
            return None
    
    async def calculate_confidence(self, content: str) -> float:
        """
        Calculate confidence score for content using transformers.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not content.strip():
            return 0.0
        
        # First try regex-based extraction
        confidence = self._extract_confidence_score(content)
        if confidence != 0.7:  # Not the default value
            return confidence
        
        # Use transformer-based analysis if available
        if self.sentiment_analyzer:
            try:
                results = self.sentiment_analyzer(content)
                if results and len(results[0]) > 0:
                    # Use the confidence of the most confident prediction
                    max_score = max(result['score'] for result in results[0])
                    return max_score
            except Exception as e:
                self.logger.warning(f"Error in confidence calculation: {e}")
        
        # Fallback: analyze linguistic confidence indicators
        return self._analyze_linguistic_confidence(content)
    
    def _analyze_linguistic_confidence(self, content: str) -> float:
        """Analyze linguistic indicators of confidence."""
        content_lower = content.lower()
        
        # High confidence indicators
        high_confidence = [
            "definitely", "certainly", "absolutely", "clearly", "obviously",
            "undoubtedly", "without doubt", "for sure", "guaranteed"
        ]
        
        # Low confidence indicators
        low_confidence = [
            "maybe", "perhaps", "possibly", "might", "could be",
            "not sure", "uncertain", "unclear", "doubtful"
        ]
        
        # Medium confidence indicators
        medium_confidence = [
            "probably", "likely", "seems", "appears", "suggests",
            "indicates", "believe", "think"
        ]
        
        high_count = sum(1 for indicator in high_confidence if indicator in content_lower)
        low_count = sum(1 for indicator in low_confidence if indicator in content_lower)
        medium_count = sum(1 for indicator in medium_confidence if indicator in content_lower)
        
        if high_count > low_count and high_count > medium_count:
            return 0.9
        elif low_count > high_count and low_count > medium_count:
            return 0.3
        elif medium_count > 0:
            return 0.6
        else:
            return 0.5  # Neutral
    
    async def quantify_uncertainty(self, beliefs: List[BeliefState]) -> float:
        """
        Quantify uncertainty using entropy measures.
        
        Args:
            beliefs: List of belief states to analyze
            
        Returns:
            Uncertainty score between 0.0 and 1.0
        """
        if not beliefs:
            return 1.0  # Maximum uncertainty with no beliefs
        
        # Calculate entropy based on confidence distribution
        confidences = [belief.confidence for belief in beliefs]
        
        if not confidences:
            return 1.0
        
        # Calculate Shannon entropy
        entropy = self._calculate_shannon_entropy(confidences)
        
        # Normalize entropy to 0-1 range
        max_entropy = math.log2(len(confidences)) if len(confidences) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Also consider variance in confidence scores
        if np is not None:
            confidence_variance = float(np.var(confidences))
            # Combine entropy and variance
            uncertainty = (normalized_entropy + confidence_variance) / 2.0
        else:
            uncertainty = normalized_entropy
        
        return min(max(uncertainty, 0.0), 1.0)
    
    def _calculate_shannon_entropy(self, values: List[float]) -> float:
        """
        Calculate Shannon entropy for a list of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            Shannon entropy value
        """
        if not values:
            return 0.0
        
        # Convert to probabilities (normalize)
        total = sum(values)
        if total == 0:
            return 0.0
        
        probabilities = [v / total for v in values]
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    async def create_epistemic_state(self, agent_id: str, agent_logs: List[AgentEvent]) -> EpistemicState:
        """
        Create a complete epistemic state from agent logs.
        
        Args:
            agent_id: ID of the agent
            agent_logs: List of agent events to analyze
            
        Returns:
            Complete epistemic state
        """
        # Extract all components
        beliefs = await self.extract_beliefs(agent_logs)
        knowledge = await self.extract_knowledge(agent_logs)
        goals = await self.extract_goals(agent_logs)
        
        # Calculate overall confidence and uncertainty
        overall_confidence = await self._calculate_overall_confidence(beliefs, knowledge)
        uncertainty_score = await self.quantify_uncertainty(beliefs)
        
        # Extract decision context from recent events
        decision_context = self._extract_decision_context(agent_logs)
        
        return EpistemicState(
            agent_id=agent_id,
            timestamp=datetime.now(),
            belief_states=beliefs,
            knowledge_state=knowledge,
            goal_states=goals,
            confidence_level=overall_confidence,
            uncertainty_score=uncertainty_score,
            decision_context=decision_context
        )
    
    async def _calculate_overall_confidence(self, beliefs: List[BeliefState], 
                                          knowledge: KnowledgeState) -> float:
        """Calculate overall confidence from beliefs and knowledge."""
        if not beliefs and knowledge.confidence_score == 0.0:
            return 0.0
        
        # Weight belief confidence and knowledge confidence
        belief_confidence = sum(b.confidence for b in beliefs) / len(beliefs) if beliefs else 0.0
        knowledge_confidence = knowledge.confidence_score
        
        # Weighted average (beliefs weighted more heavily)
        if beliefs and knowledge_confidence > 0:
            return (belief_confidence * 0.7) + (knowledge_confidence * 0.3)
        elif beliefs:
            return belief_confidence
        else:
            return knowledge_confidence
    
    def _extract_decision_context(self, agent_logs: List[AgentEvent]) -> Dict[str, Any]:
        """Extract decision context from recent events."""
        context = {
            "recent_decisions": [],
            "active_tools": [],
            "conversation_context": [],
            "error_history": []
        }
        
        # Look at recent events (last 10)
        recent_events = agent_logs[-10:] if len(agent_logs) > 10 else agent_logs
        
        for event in recent_events:
            if event.event_type == EventType.DECISION_START:
                context["recent_decisions"].append({
                    "decision": event.message,
                    "timestamp": event.timestamp.isoformat()
                })
            elif event.event_type == EventType.TOOL_START:
                context["active_tools"].append(event.data.get("tool_name", "unknown"))
            elif event.event_type in [EventType.USER_MESSAGE, EventType.AGENT_MESSAGE]:
                context["conversation_context"].append({
                    "message": event.message,
                    "type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat()
                })
            elif event.event_type == EventType.ERROR:
                context["error_history"].append({
                    "error": event.message,
                    "timestamp": event.timestamp.isoformat()
                })
        
        
        return context

    
    async def get_current_state(self, agent_id: str) -> Optional[EpistemicState]:
        """Get current epistemic state for an agent."""
        try:
            # In a real implementation, this would query the database
            # For now, return None to indicate no state found
            return None
        except Exception as e:
            self.logger.error(f"Failed to get current state for agent {agent_id}: {e}")
            return None
    
    async def search_states(self, filters: Dict[str, Any], page: int = 1, size: int = 20) -> Dict[str, Any]:
        """Search epistemic states with filtering and pagination."""
        try:
            # In a real implementation, this would query the database with filters
            # For now, return empty results
            return {
                "items": [],
                "total": 0
            }
        except Exception as e:
            self.logger.error(f"Failed to search epistemic states: {e}")
            raise

    async def extract_epistemic_state(self, agent_id: str, agent_logs: List[AgentEvent]) -> EpistemicState:
        """
        Extract complete epistemic state from agent logs.
        
        Args:
            agent_id: ID of the agent
            agent_logs: List of agent events to analyze
            
        Returns:
            Complete epistemic state
        """
        if not agent_logs:
            return EpistemicState(
                agent_id=agent_id,
                timestamp=datetime.utcnow()
            )
        
        # Extract components
        belief_states = await self.extract_beliefs(agent_logs)
        knowledge_state = await self.extract_knowledge(agent_logs)
        goal_states = await self.extract_goals(agent_logs)
        
        # Calculate overall confidence
        if belief_states:
            confidence_level = sum(b.confidence for b in belief_states) / len(belief_states)
        else:
            confidence_level = knowledge_state.confidence_score
        
        # Calculate uncertainty
        uncertainty_score = await self.quantify_uncertainty(belief_states)
        
        # Extract decision context from recent events
        decision_context = self._extract_decision_context(agent_logs[-10:])  # Last 10 events
        
        return EpistemicState(
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            belief_states=belief_states,
            knowledge_state=knowledge_state,
            goal_states=goal_states,
            confidence_level=confidence_level,
            uncertainty_score=uncertainty_score,
            decision_context=decision_context
        )
    
    def _extract_decision_context(self, agent_logs: List[AgentEvent]) -> Dict[str, Any]:
        """Extract decision context from recent events."""
        context = {
            "recent_decisions": [],
            "active_tools": [],
            "conversation_context": [],
            "error_history": []
        }
        
        # Look at recent events (last 10)
        recent_events = agent_logs[-10:] if len(agent_logs) > 10 else agent_logs
        
        for event in recent_events:
            if event.event_type == EventType.DECISION_START:
                context["recent_decisions"].append({
                    "decision": event.message,
                    "timestamp": event.timestamp.isoformat()
                })
            elif event.event_type == EventType.TOOL_CALL:
                context["active_tools"].append(event.data.get("tool_name", "unknown"))
            elif event.event_type in [EventType.MESSAGE_SEND, EventType.MESSAGE_RECEIVE]:
                context["conversation_context"].append({
                    "message": event.message,
                    "type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat()
                })
            elif event.event_type == EventType.AGENT_ERROR:
                context["error_history"].append({
                    "error": event.message,
                    "timestamp": event.timestamp.isoformat()
                })
        
        return context