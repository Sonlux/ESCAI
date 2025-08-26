# ESCAI Framework Deployment Guide

This guide provides comprehensive instructions for deploying the ESCAI Framework in different environments.

## Deployment Options

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Enterprise Deployment](#enterprise-deployment)

## Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Kubernetes cluster (for K8s deployment)
- Cloud provider account (for cloud deployment)

### Minimum System Requirements

| Component | Minimum  | Recommended |
| --------- | -------- | ----------- |
| CPU       | 2 cores  | 4+ cores    |
| RAM       | 4 GB     | 8+ GB       |
| Storage   | 20 GB    | 100+ GB     |
| Network   | 100 Mbps | 1 Gbps      |

## Local Development

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/escai-framework/escai.git
cd escai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Database Setup

```bash
# Start local databases with Docker Compose
docker-compose -f docker-compose.dev.yml up -d postgres mongodb redis influxdb neo4j

# Run database migrations
alembic upgrade head

# Seed test data (optional)
python scripts/seed_data.py
```

### 3. Configuration

Create a local configuration file:

```bash
# Create config directory
mkdir -p ~/.escai

# Copy example configuration
cp config/local.example.yaml ~/.escai/config.yaml
```

Edit `~/.escai/config.yaml`:

```yaml
# Local development configuration
database:
  postgresql:
    host: localhost
    port: 5432
    database: escai_dev
    username: escai_user
    password: escai_password

  mongodb:
    host: localhost
    port: 27017
    database: escai_logs

  redis:
    host: localhost
    port: 6379
    database: 0

  influxdb:
    host: localhost
    port: 8086
    database: escai_metrics
    username: escai_user
    password: escai_password

  neo4j:
    host: localhost
    port: 7687
    username: neo4j
    password: escai_password

api:
  host: 0.0.0.0
  port: 8000
  debug: true
  reload: true

security:
  jwt_secret: "your-development-secret-key"
  jwt_expiration: 3600
  enable_cors: true
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8080"

logging:
  level: DEBUG
  format: detailed
  file: logs/escai.log
```

### 4. Start Services

```bash
# Start the ESCAI server
escai server start

# Or run directly with Python
python -m escai_framework.api.main

# Start the CLI (in another terminal)
escai cli

# Start the dashboard (optional)
streamlit run escai_framework/visualization/dashboard.py
```

### 5. Verify Installation

```bash
# Check server health
curl http://localhost:8000/health

# Run basic tests
pytest tests/integration/test_basic_functionality.py

# Test CLI
escai status
```

## Docker Deployment

### 1. Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/escai-framework/escai.git
cd escai

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f escai-api
```

### 2. Custom Docker Configuration

Create a custom `docker-compose.override.yml`:

```yaml
version: "3.8"

services:
  escai-api:
    environment:
      - ESCAI_LOG_LEVEL=INFO
      - ESCAI_WORKERS=4
    ports:
      - "8000:8000"
    volumes:
      - ./custom-config.yaml:/app/config/config.yaml
      - ./logs:/app/logs

  postgres:
    environment:
      - POSTGRES_DB=escai_production
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups

  mongodb:
    volumes:
      - mongodb_data:/data/db
      - ./backups:/backups

  redis:
    volumes:
      - redis_data:/data

  influxdb:
    volumes:
      - influxdb_data:/var/lib/influxdb

  neo4j:
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  postgres_data:
  mongodb_data:
  redis_data:
  influxdb_data:
  neo4j_data:
  neo4j_logs:
```

### 3. Production Docker Configuration

```yaml
# docker-compose.prod.yml
version: "3.8"

services:
  escai-api:
    image: escai/escai-framework:latest
    restart: unless-stopped
    environment:
      - ESCAI_ENV=production
      - ESCAI_LOG_LEVEL=INFO
      - ESCAI_WORKERS=8
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - mongodb
      - redis
      - influxdb
      - neo4j
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - escai-api

  postgres:
    image: postgres:13
    restart: unless-stopped
    environment:
      - POSTGRES_DB=escai_production
      - POSTGRES_USER=escai_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U escai_user"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Additional services...
```

## Kubernetes Deployment

### 1. Using Helm Charts

```bash
# Add ESCAI Helm repository
helm repo add escai https://charts.escai.dev
helm repo update

# Install ESCAI with default values
helm install escai escai/escai-framework

# Or with custom values
helm install escai escai/escai-framework -f values.yaml
```

### 2. Custom Helm Values

Create `values.yaml`:

```yaml
# Helm values for ESCAI deployment
replicaCount: 3

image:
  repository: escai/escai-framework
  tag: "latest"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: escai.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: escai-tls
      hosts:
        - escai.yourdomain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    postgresPassword: "secure-password"
    database: "escai_production"
  primary:
    persistence:
      enabled: true
      size: 100Gi

mongodb:
  enabled: true
  auth:
    enabled: true
    rootPassword: "secure-password"
  persistence:
    enabled: true
    size: 100Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: "secure-password"
  master:
    persistence:
      enabled: true
      size: 20Gi

influxdb:
  enabled: true
  auth:
    admin:
      password: "secure-password"
  persistence:
    enabled: true
    size: 50Gi

neo4j:
  enabled: true
  auth:
    password: "secure-password"
  core:
    persistentVolume:
      enabled: true
      size: 50Gi

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "secure-password"

security:
  networkPolicies:
    enabled: true
  podSecurityPolicy:
    enabled: true
  rbac:
    create: true
```

### 3. Manual Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace escai

# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/mongodb.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/influxdb.yaml
kubectl apply -f k8s/neo4j.yaml
kubectl apply -f k8s/escai-api.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n escai
kubectl get services -n escai
kubectl get ingress -n escai
```

## Cloud Deployment

### AWS Deployment

#### 1. Using AWS EKS

```bash
# Create EKS cluster
eksctl create cluster --name escai-cluster --region us-west-2 --nodes 3

# Deploy ESCAI using Helm
helm install escai escai/escai-framework -f values-aws.yaml
```

#### 2. Using AWS ECS

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name escai-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service --cluster escai-cluster --service-name escai-service --task-definition escai-task
```

### Google Cloud Platform

#### 1. Using GKE

```bash
# Create GKE cluster
gcloud container clusters create escai-cluster --num-nodes=3 --zone=us-central1-a

# Deploy ESCAI
helm install escai escai/escai-framework -f values-gcp.yaml
```

### Azure Deployment

#### 1. Using AKS

```bash
# Create AKS cluster
az aks create --resource-group escai-rg --name escai-cluster --node-count 3

# Deploy ESCAI
helm install escai escai/escai-framework -f values-azure.yaml
```

## Enterprise Deployment

### High Availability Setup

```yaml
# values-ha.yaml
replicaCount: 5

affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
            - key: app.kubernetes.io/name
              operator: In
              values:
                - escai-framework
        topologyKey: kubernetes.io/hostname

postgresql:
  architecture: replication
  primary:
    persistence:
      size: 500Gi
  readReplicas:
    replicaCount: 2
    persistence:
      size: 500Gi

mongodb:
  architecture: replicaset
  replicaCount: 3
  persistence:
    size: 500Gi

redis:
  architecture: replication
  sentinel:
    enabled: true
```

### Security Hardening

```yaml
# values-security.yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 2000

podSecurityContext:
  seccompProfile:
    type: RuntimeDefault

networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000

serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/escai-service-role

encryption:
  enabled: true
  provider: vault
  vault:
    address: https://vault.company.com
    role: escai-role
```

### Monitoring and Observability

```yaml
# values-monitoring.yaml
monitoring:
  enabled: true

  prometheus:
    enabled: true
    retention: 30d
    storageClass: fast-ssd
    storage: 100Gi

  grafana:
    enabled: true
    persistence:
      enabled: true
      size: 10Gi
    dashboards:
      - escai-overview
      - escai-performance
      - escai-errors

  alertmanager:
    enabled: true
    config:
      route:
        group_by: ["alertname"]
        group_wait: 10s
        group_interval: 10s
        repeat_interval: 1h
        receiver: "web.hook"
      receivers:
        - name: "web.hook"
          webhook_configs:
            - url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

logging:
  enabled: true
  elasticsearch:
    enabled: true
    replicas: 3
    storage: 100Gi
  kibana:
    enabled: true
  fluentd:
    enabled: true
```

## Configuration Management

### Environment-Specific Configurations

```bash
# Development
cp config/environments/development.yaml ~/.escai/config.yaml

# Staging
cp config/environments/staging.yaml ~/.escai/config.yaml

# Production
cp config/environments/production.yaml ~/.escai/config.yaml
```

### Configuration Validation

```bash
# Validate configuration
escai config validate

# Test database connections
escai config test-connections

# Check security settings
escai config security-check
```

## Backup and Recovery

### Database Backups

```bash
# PostgreSQL backup
pg_dump -h localhost -U escai_user escai_production > backup_$(date +%Y%m%d).sql

# MongoDB backup
mongodump --host localhost --port 27017 --db escai_logs --out backup_$(date +%Y%m%d)

# Redis backup
redis-cli --rdb backup_$(date +%Y%m%d).rdb

# InfluxDB backup
influxd backup -portable backup_$(date +%Y%m%d)

# Neo4j backup
neo4j-admin backup --backup-dir=backup_$(date +%Y%m%d) --name=graph.db-backup
```

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# PostgreSQL
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER $POSTGRES_DB > $BACKUP_DIR/postgres.sql

# MongoDB
mongodump --host $MONGODB_HOST --port $MONGODB_PORT --db $MONGODB_DB --out $BACKUP_DIR/mongodb

# Upload to cloud storage
aws s3 sync $BACKUP_DIR s3://escai-backups/$(date +%Y%m%d)/

# Cleanup old backups
find /backups -type d -mtime +30 -exec rm -rf {} \;
```

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
```

### Application Tuning

```yaml
# config/performance.yaml
api:
  workers: 8
  worker_class: uvicorn.workers.UvicornWorker
  max_requests: 1000
  max_requests_jitter: 100
  timeout: 30
  keepalive: 5

database:
  postgresql:
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
    pool_recycle: 3600

  mongodb:
    max_pool_size: 100
    min_pool_size: 10
    max_idle_time_ms: 30000

  redis:
    max_connections: 50
    retry_on_timeout: true
    socket_keepalive: true

caching:
  enabled: true
  default_timeout: 300
  key_prefix: "escai:"

monitoring:
  sampling_rate: 0.1
  batch_size: 1000
  async_processing: true
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**

   ```bash
   # Check database connectivity
   escai config test-connections

   # Check database logs
   docker-compose logs postgres
   kubectl logs -l app=postgresql
   ```

2. **High Memory Usage**

   ```bash
   # Monitor memory usage
   kubectl top pods
   docker stats

   # Adjust memory limits
   helm upgrade escai escai/escai-framework --set resources.limits.memory=8Gi
   ```

3. **Performance Issues**

   ```bash
   # Check API performance
   curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

   # Monitor database performance
   kubectl exec -it postgres-0 -- psql -U escai_user -c "SELECT * FROM pg_stat_activity;"
   ```

### Health Checks

```bash
# API health check
curl http://localhost:8000/health

# Database health checks
curl http://localhost:8000/health/database

# Detailed system status
escai status --detailed
```

## Security Considerations

### Network Security

- Use TLS 1.3 for all communications
- Implement network policies in Kubernetes
- Use VPC/VNET isolation in cloud deployments
- Configure firewalls and security groups

### Authentication and Authorization

- Enable JWT authentication
- Implement role-based access control (RBAC)
- Use strong passwords and rotate regularly
- Enable audit logging

### Data Protection

- Encrypt data at rest and in transit
- Implement PII detection and masking
- Regular security audits and penetration testing
- Backup encryption

## Maintenance

### Regular Maintenance Tasks

```bash
# Update ESCAI Framework
helm upgrade escai escai/escai-framework --version 1.2.0

# Database maintenance
escai maintenance vacuum-databases
escai maintenance optimize-indexes
escai maintenance cleanup-old-data

# Security updates
escai security update-certificates
escai security rotate-secrets
```

### Monitoring and Alerts

Set up monitoring for:

- API response times
- Database performance
- Memory and CPU usage
- Error rates
- Security events

## Support

- **Documentation**: [https://docs.escai.dev](https://docs.escai.dev)
- **GitHub Issues**: [https://github.com/escai-framework/escai/issues](https://github.com/escai-framework/escai/issues)
- **Community Forum**: [https://community.escai.dev](https://community.escai.dev)
- **Enterprise Support**: support@escai.dev
