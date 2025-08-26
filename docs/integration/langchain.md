# LangChain Integration Guide

This guide shows how to integrate ESCAI monitoring with LangChain agents, chains, and tools.

## Quick Start

### 1. Installation

```bash
pip install escai-framework langchain
```

### 2. Basic Integration

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from escai_framework import monitor_agent

# Set up your LangChain components
llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [DuckDuckGoSearchRun()]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Monitor the agent execution
with monitor_agent(
    agent_id="langchain-search-agent",
    framework="langchain",
    config={
        "capture_reasoning": True,
        "monitor_llm_calls": True,
        "track_token_usage": True,
        "capture_retrieval": True
    }
) as session:
    # Run your agent
    result = agent_executor.invoke({
        "input": "What are the latest developments in AI?"
    })

    # Access monitoring insights
    current_state = session.get_current_epistemic_state()
    print(f"Agent confidence: {current_state.confidence_level}")
    print(f"Reasoning steps: {len(current_state.decision_context.get('reasoning_steps', []))}")
```

## Detailed Integration

### 1. Chain Monitoring

Monitor individual chains and their execution steps:

```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from escai_framework.instrumentation import LangChainInstrumentor

# Create chains
first_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short summary about {topic}."
)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

second_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Create 3 key takeaways from this summary: {summary}"
)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[first_chain, second_chain],
    verbose=True
)

# Initialize instrumentor
instrumentor = LangChainInstrumentor()

# Start monitoring
session_id = instrumentor.start_monitoring(
    agent_id="sequential-chain-agent",
    config={
        "capture_chain_steps": True,
        "monitor_llm_calls": True,
        "track_intermediate_outputs": True
    }
)

try:
    # Run the chain
    result = overall_chain.run("artificial intelligence")

    # Get chain execution analysis
    chain_analysis = instrumentor.get_chain_analysis(session_id)
    print(f"Total steps: {chain_analysis['total_steps']}")
    print(f"Execution time: {chain_analysis['total_time_ms']}ms")
    print(f"Token usage: {chain_analysis['total_tokens']}")

finally:
    # Stop monitoring
    summary = instrumentor.stop_monitoring(session_id)
    print(f"Monitoring summary: {summary}")
```

### 2. Agent with Tools Monitoring

Monitor agents that use multiple tools:

```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from escai_framework import ESCAIClient

# Set up tools
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()
tools = [wikipedia, search]

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)

# Initialize ESCAI client
client = ESCAIClient(
    base_url="http://localhost:8000",
    username="your_username",
    password="your_password"
)

# Start monitoring with tool tracking
session = client.start_monitoring(
    agent_id="multi-tool-agent",
    framework="langchain",
    config={
        "capture_reasoning": True,
        "monitor_tools": True,
        "track_tool_selection": True,
        "capture_tool_outputs": True,
        "monitor_decision_process": True
    }
)

try:
    # Run agent with complex task
    result = agent_executor.invoke({
        "input": "Compare the population of Tokyo and New York City, and explain which factors contribute to their sizes."
    })

    # Analyze tool usage patterns
    tool_analysis = client.analyze_tool_usage(session.session_id)
    print("Tool Usage Analysis:")
    for tool_name, stats in tool_analysis.items():
        print(f"  {tool_name}:")
        print(f"    Uses: {stats['usage_count']}")
        print(f"    Success rate: {stats['success_rate']:.2%}")
        print(f"    Avg response time: {stats['avg_response_time_ms']}ms")

    # Get reasoning analysis
    reasoning = client.get_reasoning_analysis(session.session_id)
    print(f"\nReasoning Analysis:")
    print(f"  Decision points: {len(reasoning['decision_points'])}")
    print(f"  Tool selection confidence: {reasoning['tool_selection_confidence']:.2f}")

finally:
    summary = client.stop_monitoring(session.session_id)
```

### 3. Memory and Context Monitoring

Monitor agents with memory and context management:

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_conversational_retrieval_agent
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Set up vector store for retrieval
loader = TextLoader("knowledge_base.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Set up memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,  # Remember last 5 exchanges
    return_messages=True
)

# Create conversational agent
agent_executor = create_conversational_retrieval_agent(
    llm=llm,
    vectorstore=vectorstore,
    memory=memory,
    verbose=True
)

# Monitor with memory tracking
with monitor_agent(
    agent_id="conversational-rag-agent",
    framework="langchain",
    config={
        "capture_reasoning": True,
        "monitor_memory": True,
        "track_memory_updates": True,
        "capture_retrieval": True,
        "monitor_context_usage": True,
        "memory_tracking_depth": 10
    }
) as session:

    # Have a conversation
    questions = [
        "What is the main topic of the knowledge base?",
        "Can you elaborate on the key concepts?",
        "How does this relate to what we discussed earlier?",
        "Summarize our conversation so far."
    ]

    for i, question in enumerate(questions):
        print(f"\n--- Question {i+1}: {question} ---")

        response = agent_executor.invoke({"input": question})
        print(f"Response: {response['output']}")

        # Analyze memory state after each interaction
        memory_state = session.get_memory_analysis()
        print(f"Memory items: {memory_state['item_count']}")
        print(f"Context relevance: {memory_state['context_relevance']:.2f}")
        print(f"Memory utilization: {memory_state['utilization']:.2%}")
```

### 4. Custom Chain Monitoring

Monitor custom chains with specific business logic:

```python
from langchain.chains.base import Chain
from langchain.schema import BaseOutputParser
from typing import Dict, Any
import json

class BusinessAnalysisChain(Chain):
    """Custom chain for business analysis tasks."""

    llm: Any
    analysis_prompt: PromptTemplate
    summary_prompt: PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        return ["business_data", "analysis_type"]

    @property
    def output_keys(self) -> List[str]:
        return ["analysis", "recommendations", "confidence"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Custom business logic here
        business_data = inputs["business_data"]
        analysis_type = inputs["analysis_type"]

        # Step 1: Analyze data
        analysis_result = self.llm.predict(
            self.analysis_prompt.format(
                data=business_data,
                type=analysis_type
            )
        )

        # Step 2: Generate recommendations
        recommendations = self.llm.predict(
            self.summary_prompt.format(
                analysis=analysis_result
            )
        )

        # Step 3: Calculate confidence
        confidence = self._calculate_confidence(analysis_result)

        return {
            "analysis": analysis_result,
            "recommendations": recommendations,
            "confidence": confidence
        }

    def _calculate_confidence(self, analysis: str) -> float:
        # Custom confidence calculation
        return 0.85  # Simplified

# Create custom chain
analysis_prompt = PromptTemplate(
    input_variables=["data", "type"],
    template="Analyze this {type} data: {data}"
)

summary_prompt = PromptTemplate(
    input_variables=["analysis"],
    template="Based on this analysis: {analysis}, provide 3 key recommendations."
)

custom_chain = BusinessAnalysisChain(
    llm=llm,
    analysis_prompt=analysis_prompt,
    summary_prompt=summary_prompt
)

# Monitor custom chain
with monitor_agent(
    agent_id="business-analysis-chain",
    framework="langchain",
    config={
        "capture_reasoning": True,
        "monitor_custom_chains": True,
        "track_business_metrics": True,
        "capture_confidence_scores": True
    }
) as session:

    result = custom_chain.run({
        "business_data": "Q4 sales data: Revenue $2M, Growth 15%",
        "analysis_type": "financial"
    })

    # Get custom chain insights
    chain_insights = session.get_custom_chain_analysis()
    print(f"Chain execution steps: {chain_insights['step_count']}")
    print(f"Business confidence: {result['confidence']}")
```

## Configuration Options

### LangChain-Specific Configuration

```python
langchain_config = {
    # Chain monitoring
    "capture_chain_steps": True,        # Monitor individual chain steps
    "track_intermediate_outputs": True, # Capture intermediate results
    "monitor_chain_composition": True,  # Track chain relationships

    # LLM monitoring
    "monitor_llm_calls": True,          # Track LLM API calls
    "track_token_usage": True,          # Monitor token consumption
    "capture_llm_responses": True,      # Store LLM responses
    "monitor_llm_latency": True,        # Track response times

    # Tool monitoring
    "monitor_tools": True,              # Track tool usage
    "track_tool_selection": True,       # Monitor tool selection process
    "capture_tool_outputs": True,       # Store tool results
    "monitor_tool_errors": True,        # Track tool failures

    # Memory monitoring
    "monitor_memory": True,             # Track memory operations
    "track_memory_updates": True,       # Monitor memory changes
    "memory_tracking_depth": 5,         # How many memory items to track
    "capture_memory_retrieval": True,   # Monitor memory access

    # Retrieval monitoring
    "capture_retrieval": True,          # Monitor retrieval operations
    "track_document_relevance": True,   # Score document relevance
    "monitor_embedding_operations": True, # Track embedding calls
    "capture_retrieval_context": True,  # Store retrieval context

    # Agent monitoring
    "monitor_agent_decisions": True,    # Track agent decision points
    "capture_agent_reasoning": True,    # Store reasoning traces
    "track_agent_state": True,          # Monitor agent state changes
    "monitor_planning_process": True,   # Track planning steps

    # Performance settings
    "sampling_rate": 1.0,               # Fraction of events to capture
    "batch_size": 100,                  # Events per processing batch
    "async_processing": True,           # Use async processing
    "buffer_size": 1000,                # Event buffer size

    # Error handling
    "capture_exceptions": True,         # Capture and analyze errors
    "monitor_retry_attempts": True,     # Track retry behavior
    "track_fallback_usage": True,       # Monitor fallback mechanisms
}
```

## Advanced Features

### 1. Real-time Monitoring Dashboard

```python
from escai_framework.visualization import LangChainDashboard

# Create real-time dashboard
dashboard = LangChainDashboard(session_id=session.session_id)

# Start dashboard server
dashboard.start(port=8080)
print("Dashboard available at http://localhost:8080")

# The dashboard will show:
# - Real-time chain execution flow
# - LLM call patterns and latency
# - Tool usage statistics
# - Memory utilization graphs
# - Error rates and patterns
```

### 2. Custom Metrics Collection

```python
from escai_framework.metrics import CustomMetric

# Define custom metrics
business_metrics = [
    CustomMetric(
        name="customer_satisfaction_score",
        description="Predicted customer satisfaction",
        extractor=lambda context: extract_satisfaction_score(context)
    ),
    CustomMetric(
        name="task_complexity",
        description="Estimated task complexity",
        extractor=lambda context: calculate_complexity(context)
    )
]

# Add to monitoring configuration
config["custom_metrics"] = business_metrics

with monitor_agent(agent_id, "langchain", config) as session:
    # Your agent code
    result = agent_executor.invoke({"input": user_query})

    # Custom metrics are automatically collected
    metrics = session.get_custom_metrics()
    print(f"Customer satisfaction: {metrics['customer_satisfaction_score']}")
    print(f"Task complexity: {metrics['task_complexity']}")
```

### 3. Integration with LangSmith

```python
from langsmith import Client as LangSmithClient
from escai_framework.integrations import LangSmithIntegration

# Set up LangSmith integration
langsmith_client = LangSmithClient()
langsmith_integration = LangSmithIntegration(langsmith_client)

# Configure ESCAI to work with LangSmith
config["integrations"] = {
    "langsmith": {
        "enabled": True,
        "sync_traces": True,
        "cross_reference_runs": True
    }
}

with monitor_agent(agent_id, "langchain", config) as session:
    # Both ESCAI and LangSmith will capture data
    result = agent_executor.invoke({"input": user_query})

    # Access combined insights
    combined_analysis = session.get_langsmith_analysis()
    print(f"ESCAI confidence: {combined_analysis['escai_confidence']}")
    print(f"LangSmith run ID: {combined_analysis['langsmith_run_id']}")
```

## Best Practices

### 1. Performance Optimization

```python
# For production environments
production_config = {
    "sampling_rate": 0.1,               # Sample 10% of events
    "capture_chain_steps": True,        # Keep essential monitoring
    "track_intermediate_outputs": False, # Reduce data volume
    "monitor_llm_calls": True,          # Keep for cost tracking
    "track_token_usage": True,          # Important for billing
    "async_processing": True,           # Better performance
    "batch_size": 500,                  # Larger batches
}
```

### 2. Error Handling

```python
from escai_framework.exceptions import LangChainInstrumentationError

try:
    with monitor_agent(agent_id, "langchain", config) as session:
        result = agent_executor.invoke({"input": user_query})
except LangChainInstrumentationError as e:
    print(f"Monitoring error: {e}")
    # Fallback to unmonitored execution
    result = agent_executor.invoke({"input": user_query})
except Exception as e:
    print(f"Agent execution error: {e}")
    # Handle agent errors
```

### 3. Testing

```python
import pytest
from escai_framework.testing import MockLangChainInstrumentor

def test_langchain_agent_monitoring():
    with MockLangChainInstrumentor() as mock_instrumentor:
        # Your test code
        result = agent_executor.invoke({"input": "test query"})

        # Verify monitoring captured expected events
        events = mock_instrumentor.get_captured_events()
        assert any(e.type == "chain_start" for e in events)
        assert any(e.type == "llm_call" for e in events)
        assert any(e.type == "chain_end" for e in events)
```

## Troubleshooting

### Common Issues

1. **Callback Registration Issues**

   ```python
   # Ensure callbacks are properly registered
   from langchain.callbacks import get_openai_callback

   with get_openai_callback() as cb:
       with monitor_agent(agent_id, "langchain", config) as session:
           result = agent_executor.invoke({"input": query})
           print(f"Tokens used: {cb.total_tokens}")
   ```

2. **Memory Monitoring Not Working**

   ```python
   # Ensure memory is properly configured
   memory = ConversationBufferWindowMemory(
       memory_key="chat_history",
       return_messages=True,  # Required for monitoring
       k=5
   )
   ```

3. **High Monitoring Overhead**
   ```python
   # Reduce monitoring overhead
   config = {
       "sampling_rate": 0.1,           # Sample fewer events
       "capture_chain_steps": False,   # Disable detailed step tracking
       "track_intermediate_outputs": False,
       "async_processing": True        # Use async processing
   }
   ```

### Performance Monitoring

```python
# Monitor the monitoring overhead
import time

start_time = time.time()

with monitor_agent(agent_id, "langchain", config) as session:
    agent_start = time.time()
    result = agent_executor.invoke({"input": query})
    agent_end = time.time()

    agent_time = agent_end - agent_start
    total_time = time.time() - start_time
    overhead = (total_time - agent_time) / agent_time * 100

    print(f"Agent execution time: {agent_time:.2f}s")
    print(f"Monitoring overhead: {overhead:.1f}%")
```

## Next Steps

1. **Explore Advanced Features**: Try custom metrics and real-time dashboards
2. **Integration Testing**: Test with your specific LangChain setup
3. **Performance Tuning**: Optimize configuration for your use case
4. **Visualization**: Set up dashboards for ongoing monitoring
5. **Alerts**: Configure alerts for important events and anomalies

For more examples, see the [LangChain examples directory](../../examples/langchain/) in the repository.
