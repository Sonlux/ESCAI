# Changelog

All notable changes to the ESCAI Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Continuous Integration/Continuous Deployment (CI/CD) pipeline
- Automated PyPI publishing workflow
- Comprehensive GitHub Actions for testing, building, and deployment

### Changed

- Updated CI/CD workflow for better reliability and PyPI integration

## [0.2.0] - 2025-09-26

### Added

- Initial project structure and core data models
- EpistemicState model for tracking agent beliefs, knowledge, and goals
- BehavioralPattern model for analyzing execution sequences
- CausalRelationship model for discovering cause-effect relationships
- PredictionResult model for performance forecasting and risk analysis
- Comprehensive validation utilities with custom exceptions
- JSON serialization utilities with support for complex types
- Dictionary conversion utilities for all data models
- Comprehensive unit test suite with 175+ tests
- Development setup with pyproject.toml configuration
- Documentation with usage examples and API reference
- Multi-framework support (LangChain, AutoGen, CrewAI, OpenAI)
- Rich CLI with 40+ commands using Click + Rich
- FastAPI REST endpoints with WebSocket support
- Multi-database architecture (PostgreSQL, MongoDB, Redis, InfluxDB, Neo4j)
- Framework instrumentation for real-time agent monitoring
- Docker Compose for full-stack development
- Custom CLI test runner for comprehensive testing
- Security scanning with Bandit
- Type checking with MyPy
- Code formatting with Black and isort
- Production-ready CI/CD pipeline

### Changed

- N/A (initial release)

### Deprecated

- N/A (initial release)

### Removed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Security

- N/A (initial release)

## [0.1.0] - 2024-01-XX

### Added

- Initial release of ESCAI Framework
- Core data models for epistemic state monitoring
- Behavioral pattern analysis capabilities
- Causal relationship discovery
- Performance prediction and risk analysis
- Validation and serialization utilities
- Comprehensive test suite
- Project documentation and setup files

[Unreleased]: https://github.com/Sonlux/ESCAI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Sonlux/ESCAI/releases/tag/v0.1.0
