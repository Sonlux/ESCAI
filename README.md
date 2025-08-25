# ESCAI Framework

**Epistemic State and Causal Analysis Intelligence Framework**

A comprehensive observability system for monitoring autonomous agent cognition in real-time. The ESCAI framework provides deep insights into how AI agents think, decide, and behave during task execution by tracking epistemic states, analyzing behavioral patterns, discovering causal relationships, and predicting performance outcomes.

## Features

- **Real-time Epistemic State Monitoring**: Track agent beliefs, knowledge, and goals as they evolve
- **Multi-Framework Support**: Compatible with LangChain, AutoGen, CrewAI, and OpenAI Assistants
- **Behavioral Pattern Analysis**: Identify and analyze patterns in agent decision-making
- **Advanced Causal Inference**: Discover cause-effect relationships using temporal analysis and Granger causality testing
- **Intervention Analysis**: Estimate effects of hypothetical interventions with statistical confidence
- **Performance Prediction**: Forecast task outcomes and identify potential failures early
- **Human-Readable Explanations**: Generate natural language explanations of agent behavior

## Installation

### From Source

```bash
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e ".[dev]"
```

### Dependencies

The ESCAI framework requires the following key dependencies:

- **Core**: pandas, numpy, python-dateutil, pyyaml
- **Statistical Analysis**: scipy, scikit-learn, statsmodels
- **Causal Inference**: dowhy (for structural causal models)
- **Database**: sqlalchemy, asyncpg, psycopg2-binary, alembic
- **MongoDB**: pymongo, motor, pydantic (for unstructured data storage)
- **Redis**: redis (for caching and real-time data streaming)
- **InfluxDB**: influxdb-client (for time-series metrics storage)
- **Neo4j**: neo4j, networkx (for graph database and causal relationships)
- **Development**: pytest, black, isort, flake8, mypy

All dependencies are automatically installed when you install the framework.

## Quick Start

### Basic Usage

```python
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, BeliefType
from datetime import datetime

# Create a belief state
belief = BeliefState(
    content="The user wants to classify images",
    belief_type=BeliefType.FACTUAL,
    confidence=0.9,
    evidence=["user input", "task context"]
)

# Create an epistemic state
epistemic_state = EpistemicState(
    agent_id="image_classifier_agent",
    timestamp=datetime.utcnow(),
    belief_states=[belief],
    confidence_level=0.85,
    uncertainty_score=0.15
)

# Validate and serialize
if epistemic_state.validate():
    json_data = epistemic_state.to_json()
    print("Epistemic state captured successfully")
```

### Behavioral Pattern Analysis

```python
from escai_framework.models.behavioral_pattern import (
    BehavioralPattern, ExecutionSequence, ExecutionStep,
    PatternType, ExecutionStatus
)

# Create execution steps
step1 = ExecutionStep(
    step_id="step_001",
    action="load_model",
    timestamp=datetime.utcnow(),
    duration_ms=1500,
    status=ExecutionStatus.SUCCESS
)

step2 = ExecutionStep(
    step_id="step_002",
    action="process_image",
    timestamp=datetime.utcnow(),
    duration_ms=800,
    status=ExecutionStatus.SUCCESS
)

# Create execution sequence
sequence = ExecutionSequence(
    sequence_id="seq_001",
    agent_id="image_classifier",
    task_description="Image classification task",
    steps=[step1, step2]
)

# Create behavioral pattern
pattern = BehavioralPattern(
    pattern_id="classification_pattern",
    pattern_name="Standard Classification",
    pattern_type=PatternType.SEQUENTIAL,
    description="Standard image classification workflow",
    execution_sequences=[sequence]
)

pattern.calculate_statistics()
print(f"Pattern success rate: {pattern.success_rate}")
```

### Causal Relationship Analysis

```python
from escai_framework.models.causal_relationship import (
    CausalRelationship, CausalEvent, CausalEvidence,
    CausalType, EvidenceType
)

# Create cause and effect events
cause_event = CausalEvent(
    event_id="cause_001",
    event_type="parameter_change",
    description="Learning rate was reduced",
    timestamp=datetime.utcnow(),
    agent_id="training_agent"
)

effect_event = CausalEvent(
    event_id="effect_001",
    event_type="performance_improvement",
    description="Model accuracy increased",
    timestamp=datetime.utcnow(),
    agent_id="training_agent"
)

# Create evidence
evidence = CausalEvidence(
    evidence_type=EvidenceType.STATISTICAL,
    description="Strong correlation observed",
    strength=0.85,
    confidence=0.92,
    source="correlation_analysis"
)

# Create causal relationship
relationship = CausalRelationship(
    relationship_id="rel_001",
    cause_event=cause_event,
    effect_event=effect_event,
    causal_type=CausalType.DIRECT,
    strength=0.8,
    confidence=0.9,
    delay_ms=2000,
    evidence=[evidence]
)

print(f"Causal strength: {relationship.strength}")
```

### Advanced Causal Inference

```python
from escai_framework.core.causal_engine import CausalEngine, TemporalEvent
import pandas as pd
import numpy as np

# Initialize causal engine
engine = CausalEngine(min_observations=50, significance_threshold=0.05)

# Create temporal events for analysis
events = []
base_time = datetime.utcnow()

for i in range(100):
    # Decision event
    decision = TemporalEvent(
        event_id=f"decision_{i}",
        event_type="decision",
        timestamp=base_time + timedelta(seconds=i*10),
        agent_id="agent_1"
    )
    events.append(decision)

    # Action event (follows decision)
    action = TemporalEvent(
        event_id=f"action_{i}",
        event_type="action",
        timestamp=base_time + timedelta(seconds=i*10 + 2),
        agent_id="agent_1"
    )
    events.append(action)

# Discover causal relationships
relationships = await engine.discover_relationships(events)
print(f"Found {len(relationships)} causal relationships")

# Test Granger causality with time series data
data = pd.DataFrame({
    'cause_var': np.random.randn(100),
    'effect_var': np.random.randn(100)
})

granger_result = await engine.test_granger_causality(data, 'cause_var', 'effect_var')
print(f"Granger causality detected: {granger_result.is_causal}")
print(f"Confidence: {granger_result.confidence:.3f}")

# Analyze intervention effects
from escai_framework.core.causal_engine import CausalGraph

graph = CausalGraph()
graph.nodes.add("treatment")
graph.nodes.add("outcome")

intervention_data = pd.DataFrame({
    'treatment': np.random.randn(100),
    'outcome': 2.0 * np.random.randn(100) + np.random.randn(100) * 0.1
})

intervention_effect = await engine.analyze_interventions(
    graph=graph,
    intervention_variable="treatment",
    intervention_value=1.0,
    target_variable="outcome",
    data=intervention_data
)

print(f"Expected intervention effect: {intervention_effect.expected_effect:.3f}")
print(f"Confidence interval: {intervention_effect.confidence_interval}")
```

### Performance Prediction

```python
from escai_framework.models.prediction_result import (
    PredictionResult, RiskFactor, Intervention,
    PredictionType, RiskLevel, InterventionType
)

# Create risk factor
risk_factor = RiskFactor(
    factor_id="risk_001",
    name="High Task Complexity",
    description="Task complexity exceeds normal parameters",
    impact_score=0.7,
    probability=0.6,
    category="task_complexity"
)

# Create intervention
intervention = Intervention(
    intervention_id="int_001",
    intervention_type=InterventionType.PARAMETER_ADJUSTMENT,
    name="Reduce Batch Size",
    description="Reduce processing batch size to improve stability",
    expected_impact=0.6,
    implementation_cost=0.2,
    urgency=RiskLevel.MEDIUM
)

# Create prediction result
prediction = PredictionResult(
    prediction_id="pred_001",
    agent_id="processing_agent",
    prediction_type=PredictionType.SUCCESS_PROBABILITY,
    predicted_value=0.75,
    confidence_score=0.88,
    risk_factors=[risk_factor],
    recommended_interventions=[intervention]
)

print(f"Predicted success probability: {prediction.predicted_value}")
print(f"Overall risk score: {prediction.calculate_overall_risk_score()}")
```

### Human-Readable Explanations

```python
from escai_framework.core.explanation_engine import ExplanationEngine, ExplanationStyle

# Initialize explanation engine
engine = ExplanationEngine()

# Generate behavior explanation
behavior_explanation = await engine.explain_behavior(
    behavioral_patterns=[pattern],
    execution_sequences=[sequence],
    style=ExplanationStyle.DETAILED
)

print("Behavior Analysis:")
print(behavior_explanation.content)
print(f"Confidence: {behavior_explanation.confidence_score:.2f}")

# Generate decision pathway explanation
decision_explanation = await engine.explain_decision_pathway(
    epistemic_states=[epistemic_state],
    execution_sequence=sequence,
    style=ExplanationStyle.SIMPLE
)

print("\nDecision Pathway:")
print(decision_explanation.content)

# Generate causal explanation
causal_explanation = await engine.explain_causal_relationship(
    causal_relationship=relationship,
    style=ExplanationStyle.DETAILED
)

print("\nCausal Analysis:")
print(causal_explanation.content)

# Generate prediction explanation
prediction_explanation = await engine.explain_prediction(
    prediction_result=prediction,
    style=ExplanationStyle.DETAILED
)

print("\nPrediction Analysis:")
print(prediction_explanation.content)

# Compare successful vs failed attempts
comparison = await engine.compare_success_failure(
    successful_sequences=[successful_sequence],
    failed_sequences=[failed_sequence],
    style=ExplanationStyle.DETAILED
)

print("\nComparative Analysis:")
print(comparison.content)

# Get explanation quality metrics
metrics = await engine.get_explanation_quality_metrics(behavior_explanation)
print(f"\nExplanation Quality Metrics: {metrics}")
```

## Examples

The framework includes comprehensive examples demonstrating all capabilities:

### Basic Usage Example

```bash
python examples/basic_usage.py
```

Demonstrates core data models and basic functionality.

### Storage Examples

```bash
# PostgreSQL structured data storage
python examples/postgresql_storage_example.py

# MongoDB unstructured data storage
python examples/mongodb_storage_example.py

# Redis caching and real-time streaming
python examples/redis_storage_example.py

# InfluxDB time-series metrics
python examples/influxdb_storage_example.py

# Neo4j graph database for causal relationships
python examples/neo4j_storage_example.py
```

Shows comprehensive database integration including:

- **PostgreSQL**: Structured data with repository pattern and migrations
- **MongoDB**: Unstructured data with text search and aggregation
- **Redis**: Caching, session management, and real-time streaming
- **InfluxDB**: Time-series metrics with retention policies and dashboards
- **Neo4j**: Graph database with causal relationship analysis and advanced analytics
- Multi-database setup and configuration
- Data validation and serialization
- Advanced querying and aggregation
- Performance optimization and monitoring

### Causal Analysis Example

```bash
python examples/causal_analysis_example.py
```

Shows advanced causal inference capabilities including:

- Temporal causality detection
- Granger causality testing
- Intervention analysis
- Causal graph construction

### Explanation Engine Example

```bash
python examples/explanation_engine_example.py
```

Demonstrates human-readable explanation generation including:

- Behavior summaries with pattern analysis
- Decision pathway explanations
- Causal relationship explanations
- Prediction explanations with risk factors
- Comparative analysis between success and failure

## REST API

The ESCAI framework includes a comprehensive REST API with WebSocket support for real-time monitoring and analysis:

### Quick Start API

```bash
# Start the API server
uvicorn escai_framework.api.main:app --reload

# Test basic functionality
python test_api_basic.py

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Authentication

```python
import httpx

# Login to get access token
response = httpx.post("http://localhost:8000/api/v1/auth/login", json={
    "username": "admin",
    "password": "admin123"
})
token_data = response.json()
headers = {"Authorization": f"Bearer {token_data['access_token']}"}

# Access protected endpoints
user_info = httpx.get("http://localhost:8000/api/v1/auth/me", headers=headers)
print(user_info.json())
```

### Monitoring Endpoints

```python
# Start monitoring an agent
monitoring_request = {
    "agent_id": "my-agent-001",
    "framework": "langchain",
    "monitoring_config": {
        "capture_epistemic_states": True,
        "max_events_per_second": 100
    }
}

response = httpx.post(
    "http://localhost:8000/api/v1/monitor/start",
    json=monitoring_request,
    headers=headers
)
session_data = response.json()
session_id = session_data["session_id"]

# Get monitoring status
status = httpx.get(
    f"http://localhost:8000/api/v1/monitor/{session_id}/status",
    headers=headers
)
print(status.json())

# Stop monitoring
stop_response = httpx.post(
    f"http://localhost:8000/api/v1/monitor/{session_id}/stop",
    headers=headers
)
```

### Analysis Endpoints

```python
# Search epistemic states with pagination
search_query = {
    "agent_id": "my-agent-001",
    "confidence_min": 0.7,
    "start_time": "2024-01-01T00:00:00Z"
}

response = httpx.post(
    "http://localhost:8000/api/v1/epistemic/search",
    json=search_query,
    headers=headers,
    params={"page": 1, "size": 20}
)
epistemic_data = response.json()

# Analyze behavioral patterns
pattern_query = {
    "agent_id": "my-agent-001",
    "success_rate_min": 0.8
}

patterns = httpx.post(
    "http://localhost:8000/api/v1/patterns/analyze",
    json=pattern_query,
    headers=headers
)

# Generate performance prediction
prediction_request = {
    "agent_id": "my-agent-001",
    "prediction_horizon": 300,
    "include_risk_factors": True
}

prediction = httpx.post(
    "http://localhost:8000/api/v1/predictions/generate",
    json=prediction_request,
    headers=headers
)

# Get agent summary
summary = httpx.get(
    "http://localhost:8000/api/v1/agents/my-agent-001/summary?days=7",
    headers=headers
)
```

### WebSocket Real-time Interface

```python
import asyncio
import websockets
import json

async def monitor_agent():
    uri = f"ws://localhost:8000/ws/monitor/session_123?token={access_token}"

    async with websockets.connect(uri) as websocket:
        # Subscribe to epistemic updates
        await websocket.send(json.dumps({
            "type": "subscribe",
            "data": {
                "type": "epistemic_updates",
                "agent_id": "my-agent-001"
            }
        }))

        # Listen for real-time updates
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "epistemic_update":
                print(f"New epistemic state: {data['data']}")

# Run WebSocket client
asyncio.run(monitor_agent())
```

### API Features

- **JWT Authentication**: Secure token-based authentication with refresh tokens
- **Role-Based Access Control**: Admin, Researcher, Developer, and Viewer roles
- **Rate Limiting**: Configurable rate limits to prevent abuse
- **Request Validation**: Comprehensive input validation using Pydantic
- **Error Handling**: Detailed error responses with request tracking
- **Real-time Updates**: WebSocket subscriptions for live monitoring
- **Pagination**: Efficient handling of large datasets
- **Filtering**: Advanced filtering capabilities for all endpoints
- **Documentation**: Auto-generated OpenAPI/Swagger documentation

## Architecture

The ESCAI framework follows a modular architecture with the following key components:

- **Models**: Core data structures for epistemic states, behavioral patterns, causal relationships, and predictions
- **Instrumentation**: Framework-specific adapters for monitoring different agent systems
- **Core Processing**: Engines for extracting insights, causal inference, temporal analysis, and explanation generation
- **Analytics**: Machine learning models, statistical analysis, and Granger causality testing
- **API**: Comprehensive REST API with JWT authentication, rate limiting, and WebSocket real-time interface
- **Storage**: Hybrid multi-database architecture with PostgreSQL for structured data and MongoDB for unstructured data
- **Visualization**: Dashboard and reporting components

### Storage Architecture

The ESCAI framework uses a comprehensive multi-database storage approach:

- **PostgreSQL**: Structured data including agent metadata, epistemic states, behavioral patterns, causal relationships, and predictions
- **MongoDB**: Unstructured data including raw logs, processed events, explanations, configurations, and analytics results
- **Redis**: Caching, session management, and real-time data streaming with pub/sub capabilities
- **InfluxDB**: Time-series metrics for performance monitoring and system analytics
- **Neo4j**: Graph database for causal relationships, knowledge graphs, and complex relationship analysis

This multi-database approach provides optimal performance for different data types while maintaining data consistency and enabling complex queries across structured, unstructured, time-series, and graph data.

## Data Models

### EpistemicState

Represents an agent's complete epistemic state including beliefs, knowledge, goals, and confidence levels.

### BehavioralPattern

Captures recurring patterns in agent execution sequences with statistical analysis.

### CausalRelationship

Models cause-effect relationships between events with evidence and confidence measures.

### PredictionResult

Contains performance predictions with risk factors and recommended interventions.

### ExplanationEngine

Generates human-readable explanations for agent behavior, decisions, causal relationships, and predictions with configurable styles and quality metrics.

## Storage and Database

The ESCAI framework provides comprehensive data storage capabilities with support for both structured and unstructured data:

### Database Setup

```python
from escai_framework.storage.database import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

# Configure with both PostgreSQL and MongoDB
db_manager.initialize(
    database_url='postgresql://user:password@localhost:5432/escai',
    async_database_url='postgresql+asyncpg://user:password@localhost:5432/escai',
    mongo_url='mongodb://localhost:27017',
    mongo_db_name='escai_unstructured'
)

# Create tables and collections
await db_manager.create_tables()
```

### PostgreSQL Storage (Structured Data)

```python
from escai_framework.storage.repositories import AgentRepository, EpistemicStateRepository

# Use PostgreSQL repositories for structured data
async with db_manager.get_async_session() as session:
    agent_repo = AgentRepository()

    # Create agent record
    agent = await agent_repo.create(
        session,
        agent_id="demo_agent",
        name="Demo Agent",
        framework="langchain",
        version="1.0.0"
    )

    # Store epistemic state
    epistemic_repo = EpistemicStateRepository()
    state = await epistemic_repo.create(
        session,
        agent_id=agent.id,
        beliefs={"task": "classification"},
        goals=["complete_task"],
        confidence_level=0.85
    )
```

### MongoDB Storage (Unstructured Data)

```python
from escai_framework.storage.mongo_models import RawLogDocument, ProcessedEventDocument

# Access MongoDB repositories
mongo_manager = db_manager.mongo_manager

# Store raw logs
raw_log = RawLogDocument(
    agent_id="demo_agent",
    session_id="session_123",
    framework="langchain",
    log_level="INFO",
    message="Processing user request",
    metadata={"user_id": "user123"},
    timestamp=datetime.utcnow()
)

log_id = await mongo_manager.raw_logs.insert_one(raw_log)

# Store processed events
event = ProcessedEventDocument(
    agent_id="demo_agent",
    session_id="session_123",
    event_type="decision_made",
    event_data={"decision": "use_tool", "confidence": 0.9},
    timestamp=datetime.utcnow()
)

event_id = await mongo_manager.processed_events.insert_one(event)

# Query with advanced filters
recent_logs = await mongo_manager.raw_logs.find_by_agent(
    "demo_agent",
    start_time=datetime.utcnow() - timedelta(hours=1),
    log_level="ERROR"
)

# Text search capabilities
search_results = await mongo_manager.raw_logs.search_logs(
    "database connection",
    agent_id="demo_agent"
)

# Analytics and statistics
stats = await mongo_manager.raw_logs.get_log_statistics(
    agent_id="demo_agent",
    hours_back=24
)
```

### Storage Features

- **Automatic Indexing**: Optimized indexes for time-series queries, text search, and aggregations
- **Data Validation**: Pydantic models ensure data integrity and type safety
- **TTL Policies**: Automatic cleanup of old data with configurable retention periods
- **Connection Pooling**: Efficient connection management for high-throughput scenarios
- **Graceful Degradation**: System continues to function if MongoDB is unavailable
- **Migration Support**: Alembic integration for PostgreSQL schema migrations
- **Repository Pattern**: Clean separation of concerns with specialized repositories for different data types

### Neo4j Graph Database (Causal Relationships)

```python
from escai_framework.storage.neo4j_manager import Neo4jManager, create_causal_relationship_graph
from escai_framework.storage.neo4j_analytics import Neo4jAnalytics, CentralityMeasure
from escai_framework.storage.neo4j_models import CausalNode, CausalRelationship, AgentNode

# Initialize Neo4j manager
neo4j_manager = Neo4jManager(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="escai"
)

await neo4j_manager.connect()

# Create agent node
agent = AgentNode(
    node_id="agent_001",
    agent_name="Causal Analysis Agent",
    framework="langchain",
    capabilities=["reasoning", "causal_analysis"]
)
await neo4j_manager.create_node(agent)

# Create causal events
cause_event = {
    'node_id': 'event_001',
    'event_type': 'decision',
    'description': 'Agent decides to use search tool',
    'timestamp': datetime.utcnow(),
    'agent_id': 'agent_001'
}

effect_event = {
    'node_id': 'event_002',
    'event_type': 'action',
    'description': 'Agent executes search query',
    'timestamp': datetime.utcnow() + timedelta(seconds=2),
    'agent_id': 'agent_001'
}

# Create causal relationship
relationship_data = {
    'relationship_id': 'rel_001',
    'strength': 0.9,
    'delay_ms': 2000,
    'evidence': ['execution_trace', 'timing_analysis'],
    'confidence': 0.95
}

# Store causal relationship in graph
success = await create_causal_relationship_graph(
    neo4j_manager, cause_event, effect_event, relationship_data
)

# Perform graph analytics
analytics = Neo4jAnalytics(neo4j_manager)

# Find causal paths
paths = await neo4j_manager.find_causal_paths("event_001", "event_005", max_depth=5)
print(f"Found {len(paths)} causal paths")

# Calculate centrality measures
centrality = await analytics.calculate_centrality_measures(
    CentralityMeasure.PAGERANK, max_nodes=50
)
print(f"Most central nodes: {list(centrality.items())[:5]}")

# Discover causal patterns
patterns = await analytics.discover_causal_patterns(
    agent_id="agent_001", min_frequency=3, min_significance=0.7
)
print(f"Discovered {len(patterns)} causal patterns")

# Analyze temporal patterns
temporal_analysis = await analytics.analyze_temporal_patterns(
    time_window_hours=24, agent_id="agent_001"
)
print(f"Temporal analysis completed in {temporal_analysis.execution_time_ms:.2f}ms")

# Generate visualization data
viz_data = await analytics.generate_graph_visualization_data(
    agent_id="agent_001", max_nodes=100
)
print(f"Visualization: {len(viz_data['nodes'])} nodes, {len(viz_data['edges'])} edges")
```

### Available Repositories

**PostgreSQL Repositories:**

- `AgentRepository`: Agent metadata and configuration
- `MonitoringSessionRepository`: Session tracking and management
- `EpistemicStateRepository`: Structured epistemic state data
- `BehavioralPatternRepository`: Pattern analysis results
- `CausalRelationshipRepository`: Causal inference results
- `PredictionRepository`: Performance predictions and outcomes

**MongoDB Repositories:**

- `RawLogRepository`: Raw agent execution logs with text search
- `ProcessedEventRepository`: Structured events with timeline analysis
- `ExplanationRepository`: Generated explanations with confidence scoring
- `ConfigurationRepository`: System and user configurations with versioning
- `AnalyticsResultRepository`: ML model results and performance metrics

**Redis Operations:**

- Session management and caching with TTL policies
- Real-time data streaming using Redis Streams
- Rate limiting and temporary storage with counters
- Pub/sub messaging for real-time notifications

**InfluxDB Operations:**

- Time-series metrics storage with retention policies
- Performance monitoring and system analytics
- Batch data ingestion and querying
- Automated data aggregation and downsampling

**Neo4j Operations:**

- Graph-based causal relationship storage and analysis
- Advanced graph analytics (centrality, pattern discovery)
- Causal path finding and relationship traversal
- Knowledge graph construction and querying

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=escai_framework --cov-report=html

# Run specific test file
pytest tests/unit/test_epistemic_state.py

# Run with verbose output
pytest -v

# Test API endpoints specifically
pytest tests/integration/test_api_endpoints.py -v

# Quick API functionality test
python test_api_basic.py
```

### API Testing

The framework includes comprehensive API testing:

```bash
# Basic API functionality test
python test_api_basic.py

# Comprehensive integration tests
pytest tests/integration/test_api_endpoints.py -v

# Test specific API features
pytest tests/integration/test_api_endpoints.py::TestAuthenticationEndpoints -v
pytest tests/integration/test_api_endpoints.py::TestMonitoringEndpoints -v
pytest tests/integration/test_api_endpoints.py::TestAnalysisEndpoints -v
```

The API tests cover:

- Authentication and authorization
- Rate limiting and security
- All monitoring endpoints
- Analysis and prediction endpoints
- WebSocket real-time communication
- Error handling and validation
- Role-based access control

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run quality checks:

```bash
# Format code
black escai_framework tests

# Sort imports
isort escai_framework tests

# Lint code
flake8 escai_framework tests

# Type checking
mypy escai_framework

# Run all tests
pytest
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use the ESCAI framework in your research, please cite:

```bibtex
@software{escai_framework,
  title={ESCAI Framework: Epistemic State and Causal Analysis Intelligence},
  author={ESCAI Framework Team},
  year={2024},
  url={https://github.com/Sonlux/ESCAI}
}
```

## Support

- **Documentation**: [https://sonlux.github.io/ESCAI](https://sonlux.github.io/ESCAI)
- **Issues**: [GitHub Issues](https://github.com/Sonlux/ESCAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sonlux/ESCAI/discussions)

## Roadmap

- [x] **Causal Inference Engine**: Advanced causal analysis with Granger causality testing and intervention analysis
- [x] **Core Data Models**: Epistemic states, behavioral patterns, causal relationships, and predictions
- [x] **Temporal Analysis**: Event sequence analysis and pattern discovery
- [x] **Explanation Engine**: Human-readable explanations with natural language generation
- [x] **Database Storage Layer**: PostgreSQL for structured data and MongoDB for unstructured data with comprehensive repository pattern
- [x] **Redis Integration**: Caching, session management, and real-time data streaming
- [x] **InfluxDB Integration**: Time-series metrics storage with retention policies and dashboards
- [x] **Neo4j Integration**: Graph database for causal relationships with advanced analytics
- [x] **REST API Implementation**: Comprehensive FastAPI with JWT authentication, rate limiting, and monitoring endpoints
- [x] **WebSocket Real-time Interface**: Live monitoring and analysis updates with subscription management
- [x] **Analytics Components**: Pattern mining, failure analysis, and statistical analysis modules
- [ ] Complete remaining core processing engines
- [ ] Framework-specific instrumentors (LangChain, AutoGen, CrewAI, OpenAI)
- [ ] Machine learning models for prediction
- [ ] Visualization dashboard
- [ ] Production deployment tools
- [ ] Performance optimization
- [ ] Extended documentation
- [ ] Community examples
