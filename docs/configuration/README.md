# Configuration Management

The ESCAI framework provides a comprehensive configuration management system that supports environment-specific settings, validation, encryption, versioning, and hot-reloading.

## Quick Start

### Initialize Configuration

```bash
# Initialize configuration for development
escai config init --environment development

# Initialize for production
escai config init --environment production --format yaml
```

### Validate Configuration

```bash
# Validate current configuration
escai config validate

# Validate specific file with detailed report
escai config validate --config-file config/config.production.yaml --detailed
```

### Manage Configuration Values

```bash
# Show current configuration
escai config show

# Show specific configuration section
escai config show --key database

# Set configuration value
escai config set api.port 9000

# Set nested configuration value
escai config set database.postgres_pool_size 20
```

## Configuration Structure

### Environment-Specific Configuration

The framework supports four environments:

- **development**: Local development with debug features
- **testing**: Automated testing with minimal overhead
- **staging**: Pre-production testing environment
- **production**: Production deployment with security and performance optimizations

### Configuration Schema

```yaml
# Environment settings
environment: development
debug: true
log_level: DEBUG

# Database configuration
database:
  # PostgreSQL
  postgres_host: localhost
  postgres_port: 5432
  postgres_database: escai
  postgres_username: escai_user
  postgres_password: secure_password
  postgres_pool_size: 10
  postgres_max_overflow: 20

  # MongoDB
  mongodb_host: localhost
  mongodb_port: 27017
  mongodb_database: escai_logs

  # Redis
  redis_host: localhost
  redis_port: 6379
  redis_database: 0
  redis_pool_size: 10

  # InfluxDB
  influxdb_host: localhost
  influxdb_port: 8086
  influxdb_database: escai_metrics
  influxdb_retention_policy: 30d

  # Neo4j
  neo4j_uri: bolt://localhost:7687
  neo4j_username: neo4j
  neo4j_password: secure_password
  neo4j_database: neo4j
  neo4j_pool_size: 10

# API configuration
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false
  rate_limit_requests: 100
  rate_limit_window: 60
  cors_origins: ["*"]
  max_request_size: 10485760
  request_timeout: 30
  websocket_max_connections: 100
  websocket_heartbeat_interval: 30

# Security configuration
security:
  jwt_secret_key: your_secret_key_here
  jwt_algorithm: HS256
  jwt_access_token_expire_minutes: 30
  jwt_refresh_token_expire_days: 7
  tls_enabled: false
  password_min_length: 8
  password_require_uppercase: true
  password_require_lowercase: true
  password_require_numbers: true
  password_require_symbols: true
  session_timeout_minutes: 60
  max_login_attempts: 5
  lockout_duration_minutes: 15
  pii_detection_enabled: true
  pii_masking_enabled: true
  pii_sensitivity_level: medium
  audit_enabled: true
  audit_retention_days: 90

# Monitoring configuration
monitoring:
  monitoring_enabled: true
  monitoring_overhead_threshold: 0.1
  sampling_rate: 1.0
  metrics_enabled: true
  metrics_interval_seconds: 60
  health_check_enabled: true
  health_check_interval_seconds: 30
  alerting_enabled: true
  alert_thresholds:
    cpu_usage: 80.0
    memory_usage: 85.0
    disk_usage: 90.0
    error_rate: 5.0

# Machine Learning configuration
ml:
  model_cache_enabled: true
  model_cache_size: 100
  model_update_interval_hours: 24
  training_enabled: true
  training_batch_size: 32
  training_epochs: 10
  validation_split: 0.2
  prediction_confidence_threshold: 0.7
  ensemble_size: 5

# Custom application settings
custom_settings: {}
```

## Configuration Encryption

### Enable Encryption

```python
from escai_framework.config import ConfigManager

# Initialize with encryption enabled
manager = ConfigManager(
    config_dir="config",
    enable_encryption=True
)

# Load configuration (sensitive values will be encrypted when saved)
config = manager.load_config()
```

### Encrypt Configuration File

```bash
# Encrypt configuration with new key
escai config encrypt --config-file config.yaml --generate-key --key-file .master_key

# Encrypt with existing key
escai config encrypt --config-file config.yaml --key-file .master_key --output config.encrypted.yaml
```

### Decrypt Configuration File

```bash
# Decrypt configuration
escai config decrypt --config-file config.encrypted.yaml --key-file .master_key --output config.decrypted.yaml
```

### Sensitive Field Detection

The system automatically detects and encrypts fields containing:

- `password`
- `secret`
- `key`
- `token`
- `credential`
- `private`
- `auth`
- `cert`
- `ssl`
- `tls`

## Configuration Versioning

### View Version History

```bash
# Show configuration version history
escai config history

# Show limited number of versions
escai config history --limit 5
```

### Compare Versions

```bash
# Compare two configuration versions
escai config diff version1_id version2_id
```

### Rollback Configuration

```bash
# Rollback to previous version
escai config rollback version_id
```

### Version Management in Code

```python
from escai_framework.config import ConfigManager

manager = ConfigManager(enable_versioning=True)

# Load configuration (automatically versioned)
config = manager.load_config()

# Update configuration (creates new version)
manager.update_config({"api": {"port": 9000}})

# Get version history
history = manager.get_config_history()

# Rollback to previous version
if len(history) > 1:
    manager.rollback_config(history[1]["id"])
```

## Hot-Reloading

### Enable Hot-Reloading

```python
from escai_framework.config import ConfigManager

# Initialize with hot-reload enabled
manager = ConfigManager(
    config_dir="config",
    enable_hot_reload=True
)

# Register callback for configuration changes
def on_config_change(new_config):
    print(f"Configuration updated: {new_config.environment}")
    # Restart services, update caches, etc.

manager.add_change_callback(on_config_change)

# Load configuration
config = manager.load_config()

# Configuration will automatically reload when files change
```

### Configuration Change Callbacks

```python
# Multiple callbacks can be registered
def restart_api_server(config):
    # Restart API server with new configuration
    pass

def update_database_connections(config):
    # Update database connection pools
    pass

def refresh_security_settings(config):
    # Update security middleware
    pass

manager.add_change_callback(restart_api_server)
manager.add_change_callback(update_database_connections)
manager.add_change_callback(refresh_security_settings)
```

## Configuration Validation

### Validation Rules

The configuration validator enforces:

1. **Schema Validation**: Type checking and required fields
2. **Network Settings**: Valid IP addresses and port ranges
3. **Security Requirements**: Strong passwords, TLS in production
4. **Database Settings**: Valid connection parameters
5. **Environment Consistency**: Environment-specific requirements
6. **Resource Limits**: Reasonable resource allocation
7. **Dependencies**: Configuration interdependencies

### Custom Validation

```python
from escai_framework.config import ConfigValidator

validator = ConfigValidator()

# Validate configuration data
config_data = {...}
is_valid, errors = validator.validate_config(config_data)

if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")

# Generate detailed validation report
report = validator.generate_validation_report(config_data)
print(f"Valid: {report['valid']}")
print(f"Warnings: {report['warnings']}")
print(f"Recommendations: {report['recommendations']}")
```

## Deployment Templates

### Generate Docker Compose

```bash
# Generate Docker Compose for development
escai config generate-deploy --environment development --format docker-compose

# Generate for production
escai config generate-deploy --environment production --format docker-compose --output-dir deploy/prod
```

### Generate Kubernetes Manifests

```bash
# Generate Kubernetes manifests
escai config generate-deploy --environment production --format kubernetes --output-dir k8s/
```

### Template Customization

```python
from escai_framework.config import ConfigTemplates

templates = ConfigTemplates()

# Generate custom configuration template
config_data = templates.generate_config_template(Environment.PRODUCTION)

# Customize for your needs
config_data["api"]["workers"] = 16
config_data["database"]["postgres_pool_size"] = 50

# Generate deployment templates
docker_compose = templates.generate_docker_compose_template(Environment.PRODUCTION)
k8s_manifests = templates.generate_kubernetes_template(Environment.PRODUCTION)
```

## Environment Variables

### Configuration Override

Environment variables can override configuration values:

```bash
# Override database settings
export ESCAI_DATABASE_POSTGRES_HOST=prod-db.example.com
export ESCAI_DATABASE_POSTGRES_PASSWORD=secure_prod_password

# Override API settings
export ESCAI_API_PORT=8080
export ESCAI_API_WORKERS=8

# Override security settings
export ESCAI_SECURITY_JWT_SECRET_KEY=production_jwt_secret
export ESCAI_SECURITY_TLS_ENABLED=true
```

### Environment Variable Naming

Environment variables follow the pattern: `ESCAI_<SECTION>_<KEY>`

Examples:

- `ESCAI_DATABASE_POSTGRES_HOST` → `database.postgres_host`
- `ESCAI_API_PORT` → `api.port`
- `ESCAI_SECURITY_TLS_ENABLED` → `security.tls_enabled`

## Best Practices

### Security

1. **Use Strong Secrets**: Generate cryptographically secure secrets
2. **Enable Encryption**: Always encrypt sensitive configuration values
3. **Rotate Keys**: Regularly rotate encryption keys and secrets
4. **Restrict Access**: Limit access to configuration files and keys
5. **Audit Changes**: Enable audit logging for configuration changes

### Performance

1. **Connection Pooling**: Configure appropriate database connection pools
2. **Monitoring Overhead**: Keep monitoring overhead below 10%
3. **Resource Limits**: Set reasonable resource limits for containers
4. **Caching**: Enable caching for frequently accessed configuration

### Deployment

1. **Environment Separation**: Use separate configurations for each environment
2. **Version Control**: Version configuration changes
3. **Validation**: Always validate configuration before deployment
4. **Rollback Plan**: Have a rollback plan for configuration changes
5. **Health Checks**: Configure proper health checks and readiness probes

### Development

1. **Local Overrides**: Use local configuration overrides for development
2. **Hot-Reload**: Enable hot-reload for faster development cycles
3. **Validation**: Validate configuration changes early and often
4. **Documentation**: Document custom configuration options

## Troubleshooting

### Common Issues

#### Configuration Not Loading

```bash
# Check if configuration file exists
ls -la config/

# Validate configuration syntax
escai config validate --config-file config/config.yaml

# Check file permissions
chmod 644 config/config.yaml
```

#### Encryption/Decryption Errors

```bash
# Verify encryption key
escai config encrypt --config-file config.yaml --key-file .master_key

# Check key file permissions
chmod 600 .master_key

# Regenerate encryption key if corrupted
escai config encrypt --generate-key --key-file .master_key_new
```

#### Validation Failures

```bash
# Get detailed validation report
escai config validate --detailed

# Check environment-specific requirements
escai config validate --environment production

# Fix common validation issues
escai config set security.jwt_secret_key "$(openssl rand -base64 32)"
escai config set security.tls_enabled true
```

#### Hot-Reload Not Working

```python
# Check if hot-reload is enabled
manager = ConfigManager(enable_hot_reload=True)

# Verify file system permissions
import os
os.access("config/", os.R_OK | os.W_OK)

# Check for file system events
# Ensure watchdog package is installed
pip install watchdog
```

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from escai_framework.config import ConfigManager

manager = ConfigManager(
    config_dir="config",
    enable_hot_reload=True,
    enable_encryption=True,
    enable_versioning=True
)

# Debug information will be logged
config = manager.load_config()
```

## API Reference

See the [Configuration API Reference](api-reference.md) for detailed API documentation.

## Examples

See the [Configuration Examples](examples/) directory for complete examples and use cases.
