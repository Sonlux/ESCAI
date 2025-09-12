# ESCAI Framework

A comprehensive observability system for monitoring autonomous agent cognition in real-time. ESCAI (Epistemic State and Causal Analysis Intelligence) provides deep insights into how AI agents think, decide, and behave during task execution, enabling researchers and developers to understand agent behavior patterns, causal relationships, and performance characteristics.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them:

- **Python 3.10 or higher**
- **4GB RAM minimum** (8GB recommended for large datasets)
- **1GB free disk space** for installation and data storage
- **Git** for version control

```bash
# Check Python version
python --version

# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')"
```

### Installing

A step by step series of examples that tell you how to get a development environment running:

**Step 1: Clone the repository**

```bash
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
```

**Step 2: Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install dependencies**

```bash
# Install basic dependencies
pip install -e .

# Or install with full research dependencies
pip install -e ".[full]"
```

**Step 4: Verify installation**

```bash
escai --version
escai config check
```

End with an example of getting some data out of the system:

```bash
# Start monitoring a sample agent
escai monitor start --agent-id demo-agent --framework langchain

# View real-time epistemic states
escai monitor epistemic --agent-id demo-agent --refresh 2
```

## Running the tests

Explain how to run the automated tests for this system:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest --cov=escai_framework tests/
```

### Break down into end to end tests

Explain what these tests test and why:

```bash
# Test complete monitoring workflow
python -m pytest tests/e2e/test_complete_workflows.py
```

These tests verify that the entire monitoring pipeline works correctly from agent instrumentation through data analysis and visualization.

### And coding style tests

Explain what these tests test and why:

```bash
# Run linting and type checking
python -m pytest tests/unit/test_code_quality.py
mypy escai_framework/
flake8 escai_framework/
```

These tests ensure code quality, type safety, and adherence to Python coding standards.

## Deployment

Add additional notes about how to deploy this on a live system:

### Production Deployment

```bash
# Using Docker
docker-compose up -d

# Using Kubernetes
kubectl apply -f k8s/

# Using Helm
helm install escai ./helm/escai
```

### Environment Configuration

```bash
# Set production environment variables
export ESCAI_ENV=production
export ESCAI_DATABASE_URL=postgresql://user:pass@host:5432/escai
export ESCAI_REDIS_URL=redis://host:6379/0
```

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used for REST API
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM and management
- [Redis](https://redis.io/) - Used for caching and real-time data
- [PostgreSQL](https://www.postgresql.org/) - Primary database for structured data
- [MongoDB](https://www.mongodb.com/) - Document storage for unstructured data
- [Neo4j](https://neo4j.com/) - Graph database for causal relationships
- [InfluxDB](https://www.influxdata.com/) - Time series database for metrics
- [Streamlit](https://streamlit.io/) - Dashboard framework for visualization
- [Click](https://click.palletsprojects.com/) - Command line interface framework

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/Sonlux/ESCAI/tags).

## Authors

- **ESCAI Research Team** - _Initial work_ - [Sonlux](https://github.com/Sonlux)

See also the list of [contributors](https://github.com/Sonlux/ESCAI/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hat tip to the autonomous agent research community for inspiration
- Thanks to the open-source frameworks that made this possible: LangChain, AutoGen, CrewAI
- Inspired by the need for better observability in AI agent systems
- Special thanks to contributors and early adopters who provided valuable feedback
