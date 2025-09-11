# ESCAI Framework CLI Research Examples

This document provides comprehensive examples for using the ESCAI CLI for epistemic state monitoring and causal inference research.

## 🎓 Research Workflow Examples

### Complete Research Session

Here's a complete workflow for conducting epistemic state monitoring research:

```bash
# 1. Initial Setup and Configuration
python -m escai_framework.cli.main config setup
python -m escai_framework.cli.main config set research mode enabled
python -m escai_framework.cli.main config set monitoring overhead_limit 0.05

# 2. Start Monitoring Research Agent
python -m escai_framework.cli.main monitor start \
    --agent-id "research-agent-001" \
    --framework langchain \
    --config research_config.yaml

# 3. Real-time Monitoring During Execution
python -m escai_framework.cli.main monitor epistemic \
    --agent-id "research-agent-001" \
    --refresh 2

# 4. Collect Session Data
SESSION_ID=$(python -m escai_framework.cli.main session list --format json | jq -r '.[0].session_id')

# 5. Comprehensive Analysis
python -m escai_framework.cli.main analyze patterns \
    --agent-id "research-agent-001" \
    --timeframe 1h \
    --interactive

# 6. Causal Relationship Discovery
python -m escai_framework.cli.main analyze causal \
    --min-strength 0.6 \
    --interactive \
    --export causal_results.json

# 7. Export Research Data
python -m escai_framework.cli.main analyze export \
    --type all \
    --agent-id "research-agent-001" \
    --format json \
    --output research_data_$(date +%Y%m%d).json
```

## 🧠 Epistemic State Monitoring Examples

### Real-time Epistemic State Tracking

Monitor how agent beliefs evolve during task execution:

```bash
# Start monitoring with detailed epistemic state capture
python -m escai_framework.cli.main monitor start \
    --agent-id "epistemic-study-agent" \
    --framework langchain \
    --capture-beliefs \
    --capture-knowledge \
    --capture-goals

# View real-time epistemic state changes
python -m escai_framework.cli.main monitor epistemic \
    --agent-id "epistemic-study-agent" \
    --refresh 1 \
    --show-confidence \
    --show-uncertainty

# Analyze epistemic state evolution over time
python -m escai_framework.cli.main analyze epistemic \
    --agent-id "epistemic-study-agent" \
    --timeframe 30m \
    --show-transitions \
    --confidence-threshold 0.7
```

### Belief State Analysis

Analyze how agent beliefs change and their confidence levels:

```bash
# Analyze belief confidence patterns
python -m escai_framework.cli.main analyze stats \
    --field belief_confidence \
    --agent-id "epistemic-study-agent" \
    --timeframe 1h \
    --show-distribution

# Track belief stability over time
python -m escai_framework.cli.main analyze timeseries \
    --metric belief_stability \
    --agent-id "epistemic-study-agent" \
    --timeframe 2h \
    --show-trends

# Interactive belief exploration
python -m escai_framework.cli.main analyze interactive \
    --agent-id "epistemic-study-agent" \
    --focus beliefs \
    --filter "confidence > 0.8"
```

### Knowledge Base Evolution

Track how agent knowledge grows and changes:

```bash
# Monitor knowledge base growth
python -m escai_framework.cli.main analyze timeline \
    --agent-id "epistemic-study-agent" \
    --metric knowledge_size \
    --timeframe 1h

# Analyze knowledge acquisition patterns
python -m escai_framework.cli.main analyze patterns \
    --agent-id "epistemic-study-agent" \
    --pattern-type knowledge_acquisition \
    --min-frequency 3

# Export knowledge evolution data
python -m escai_framework.cli.main analyze export \
    --type knowledge \
    --agent-id "epistemic-study-agent" \
    --format csv \
    --output knowledge_evolution.csv
```

## 🔗 Causal Inference Examples

### Discovering Causal Relationships

Find cause-effect relationships in agent behavior:

```bash
# Interactive causal relationship discovery
python -m escai_framework.cli.main analyze causal \
    --interactive \
    --min-strength 0.7 \
    --min-evidence 5 \
    --temporal-window 10s

# Visualize causal network
python -m escai_framework.cli.main analyze causal-network \
    --agent-id "research-agent-001" \
    --layout force \
    --show-strength \
    --min-strength 0.6

# Generate causal relationship tree
python -m escai_framework.cli.main analyze tree \
    --root-cause "high_confidence_belief" \
    --max-depth 4 \
    --show-evidence
```

### Temporal Causality Analysis

Analyze time-based causal relationships:

```bash
# Time-lagged causal analysis
python -m escai_framework.cli.main analyze causal \
    --temporal-analysis \
    --max-lag 30s \
    --significance-level 0.05 \
    --agent-id "research-agent-001"

# Granger causality testing
python -m escai_framework.cli.main analyze causal \
    --method granger \
    --lag-order 5 \
    --confidence-interval 0.95

# Export temporal causal data
python -m escai_framework.cli.main analyze export \
    --type temporal_causal \
    --format json \
    --output temporal_causality.json
```

### Intervention Analysis

Analyze the effects of interventions on agent behavior:

```bash
# Analyze intervention effects
python -m escai_framework.cli.main analyze causal \
    --intervention-analysis \
    --intervention-type "confidence_boost" \
    --before-window 60s \
    --after-window 120s

# Compare pre/post intervention patterns
python -m escai_framework.cli.main analyze patterns \
    --compare-periods \
    --period1 "before_intervention" \
    --period2 "after_intervention" \
    --agent-id "research-agent-001"
```

## 📊 Behavioral Pattern Analysis Examples

### Pattern Discovery and Analysis

Identify recurring patterns in agent behavior:

```bash
# Comprehensive pattern analysis
python -m escai_framework.cli.main analyze pattern-analysis \
    --agent-id "research-agent-001" \
    --timeframe 24h \
    --min-frequency 5 \
    --pattern-types "decision,execution,error" \
    --statistical-significance 0.05

# Interactive pattern exploration
python -m escai_framework.cli.main analyze patterns \
    --interactive \
    --clustering \
    --similarity-threshold 0.8 \
    --show-transitions

# Pattern frequency heatmap
python -m escai_framework.cli.main analyze heatmap \
    --metric pattern_frequency \
    --timeframe 7d \
    --granularity hourly \
    --agent-id "research-agent-001"
```

### Sequential Pattern Mining

Discover sequences of actions and decisions:

```bash
# Mine sequential patterns
python -m escai_framework.cli.main analyze patterns \
    --sequential \
    --min-support 0.3 \
    --max-pattern-length 5 \
    --agent-id "research-agent-001"

# Analyze decision sequences
python -m escai_framework.cli.main analyze patterns \
    --pattern-type decision_sequence \
    --window-size 10 \
    --overlap 0.5

# Export sequential patterns
python -m escai_framework.cli.main analyze export \
    --type sequential_patterns \
    --format json \
    --output sequential_patterns.json
```

## 🎯 Performance Prediction Examples

### Predictive Analysis

Predict agent performance and identify failure modes:

```bash
# Generate performance predictions
python -m escai_framework.cli.main analyze predictions \
    --agent-id "research-agent-001" \
    --horizon 2h \
    --confidence-interval 0.95 \
    --include-risk-factors

# View prediction trends
python -m escai_framework.cli.main analyze prediction-trends \
    --timeframe 7d \
    --metric success_probability \
    --show-accuracy

# Analyze prediction accuracy
python -m escai_framework.cli.main analyze stats \
    --field prediction_accuracy \
    --timeframe 30d \
    --show-distribution \
    --compare-models
```

### Risk Assessment

Identify and analyze risk factors:

```bash
# Risk factor analysis
python -m escai_framework.cli.main analyze predictions \
    --risk-analysis \
    --risk-threshold 0.3 \
    --agent-id "research-agent-001"

# Failure mode analysis
python -m escai_framework.cli.main analyze patterns \
    --pattern-type failure \
    --severity-threshold high \
    --timeframe 7d

# Generate risk report
python -m escai_framework.cli.main analyze report \
    --type risk_assessment \
    --agent-id "research-agent-001" \
    --format pdf \
    --output risk_report.pdf
```

## 📈 Advanced Visualization Examples

### ASCII Visualizations for Research

Create publication-ready ASCII visualizations:

```bash
# Epistemic state evolution chart
python -m escai_framework.cli.main analyze visualize \
    --chart-type line \
    --metric confidence \
    --agent-id "research-agent-001" \
    --timeframe 1h \
    --title "Epistemic State Confidence Evolution"

# Causal strength scatter plot
python -m escai_framework.cli.main analyze causal-scatter \
    --x-axis temporal_lag \
    --y-axis causal_strength \
    --color-by evidence_count \
    --min-strength 0.5

# Pattern frequency histogram
python -m escai_framework.cli.main analyze visualize \
    --chart-type histogram \
    --metric pattern_frequency \
    --bins 20 \
    --timeframe 24h

# Multi-metric dashboard
python -m escai_framework.cli.main analyze visualize \
    --chart-type dashboard \
    --metrics "confidence,performance,pattern_count" \
    --layout grid \
    --refresh 5
```

### Interactive Visualizations

Launch interactive exploration interfaces:

```bash
# Interactive causal relationship explorer
python -m escai_framework.cli.main analyze tree-explorer \
    --interactive \
    --expandable \
    --show-evidence \
    --color-by-strength

# Interactive data exploration
python -m escai_framework.cli.main analyze interactive \
    --agent-id "research-agent-001" \
    --enable-filtering \
    --enable-search \
    --enable-export

# Live monitoring dashboard
python -m escai_framework.cli.main monitor dashboard \
    --layout research \
    --metrics "epistemic,causal,patterns,predictions" \
    --refresh 3
```

## 📊 Data Export and Analysis Examples

### Research Data Export

Export data in formats suitable for academic analysis:

```bash
# Export complete dataset for statistical analysis
python -m escai_framework.cli.main analyze export \
    --type complete \
    --agent-id "research-agent-001" \
    --format csv \
    --include-metadata \
    --output research_dataset.csv

# Export causal relationships for network analysis
python -m escai_framework.cli.main analyze export \
    --type causal \
    --format graphml \
    --include-attributes \
    --output causal_network.graphml

# Export epistemic states for time series analysis
python -m escai_framework.cli.main analyze export \
    --type epistemic \
    --format json \
    --time-series \
    --granularity 1s \
    --output epistemic_timeseries.json
```

### Report Generation for Research

Generate comprehensive reports for research documentation:

```bash
# Generate comprehensive research report
python -m escai_framework.cli.main analyze report \
    --type comprehensive \
    --agent-id "research-agent-001" \
    --timeframe 24h \
    --include-statistics \
    --include-visualizations \
    --format pdf \
    --output research_report.pdf

# Create custom research report
python -m escai_framework.cli.main analyze custom-report \
    --sections "epistemic,causal,patterns,predictions" \
    --statistical-tests \
    --confidence-intervals \
    --format latex \
    --output custom_report.tex

# Schedule automated research reports
python -m escai_framework.cli.main analyze schedule-report \
    --frequency daily \
    --time "23:00" \
    --format json \
    --email research@university.edu
```

## 🔬 Multi-Agent Research Examples

### Comparative Analysis

Compare multiple agents or experimental conditions:

```bash
# Compare multiple agents
python -m escai_framework.cli.main analyze patterns \
    --compare-agents \
    --agent-ids "agent-A,agent-B,agent-C" \
    --timeframe 2h \
    --statistical-test anova

# Compare experimental conditions
python -m escai_framework.cli.main analyze causal \
    --compare-conditions \
    --condition1 "baseline" \
    --condition2 "intervention" \
    --significance-test t-test

# Cross-agent causal analysis
python -m escai_framework.cli.main analyze causal \
    --cross-agent \
    --source-agent "agent-A" \
    --target-agent "agent-B" \
    --temporal-window 30s
```

### Longitudinal Studies

Analyze agent behavior over extended periods:

```bash
# Long-term pattern evolution
python -m escai_framework.cli.main analyze patterns \
    --longitudinal \
    --timeframe 30d \
    --granularity daily \
    --trend-analysis \
    --agent-id "research-agent-001"

# Epistemic state development over time
python -m escai_framework.cli.main analyze epistemic \
    --longitudinal \
    --timeframe 7d \
    --show-development \
    --milestone-detection

# Performance trend analysis
python -m escai_framework.cli.main analyze timeseries \
    --metric performance \
    --timeframe 30d \
    --trend-detection \
    --seasonal-analysis \
    --forecast 7d
```

## 🎯 Research Validation Examples

### Statistical Validation

Validate research findings with statistical tests:

```bash
# Statistical significance testing
python -m escai_framework.cli.main analyze stats \
    --statistical-tests \
    --field causal_strength \
    --test-type "t-test,anova,chi-square" \
    --confidence-level 0.95

# Cross-validation of patterns
python -m escai_framework.cli.main analyze patterns \
    --cross-validation \
    --folds 5 \
    --validation-metric accuracy \
    --agent-id "research-agent-001"

# Reproducibility analysis
python -m escai_framework.cli.main analyze patterns \
    --reproducibility-test \
    --replications 10 \
    --similarity-threshold 0.9
```

### Model Validation

Validate predictive models and causal inferences:

```bash
# Prediction model validation
python -m escai_framework.cli.main analyze predictions \
    --validate-model \
    --test-period 7d \
    --metrics "accuracy,precision,recall,f1" \
    --cross-validation

# Causal model validation
python -m escai_framework.cli.main analyze causal \
    --validate-causality \
    --method "randomization,instrumental" \
    --robustness-checks \
    --sensitivity-analysis
```

## 📋 Research Documentation Examples

### Automated Documentation

Generate documentation for research reproducibility:

```bash
# Generate methodology documentation
python -m escai_framework.cli.main analyze report \
    --type methodology \
    --include-parameters \
    --include-code \
    --format markdown \
    --output methodology.md

# Create reproducibility guide
python -m escai_framework.cli.main session export \
    --session-id "research-session-001" \
    --format reproducibility \
    --include-commands \
    --include-data \
    --output reproducibility_guide.md

# Generate data dictionary
python -m escai_framework.cli.main analyze export \
    --type schema \
    --format documentation \
    --output data_dictionary.md
```

These examples provide a comprehensive foundation for conducting epistemic state monitoring and causal inference research using the ESCAI CLI. Each example can be adapted to specific research questions and experimental designs.
