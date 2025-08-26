# ESCAI Framework Quick Start Deployment Guide

This guide will help you deploy the ESCAI Framework quickly in different environments.

## Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (for K8s deployment)
- Helm 3.x (for Helm deployment)
- kubectl configured

## Option 1: Local Development with Docker Compose

### 1. Clone and Setup

```bash
git clone https://github.com/escai-framework/escai.git
cd escai
```

### 2. Start All Services

```bash
# Start the complete stack
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f escai-api
```

### 3. Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Access services
echo "API: http://localhost:8000"
echo "Grafana: http://localhost:3000 (admin/admin_password)"
echo "Prometheus: http://localhost:9090"
echo "Neo4j: http://localhost:7474 (neo4j/escai_neo4j)"
```

### 4. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Option 2: Kubernetes with Helm (Recommended)

### 1. Add Helm Repository

```bash
# Add ESCAI Helm repository (when available)
helm repo add escai https://charts.escai.dev
helm repo update

# Or use local chart
cd helm/escai
```

### 2. Install ESCAI

```bash
# Create namespace
kubectl create namespace escai

# Install with default values
helm install escai ./helm/escai --namespace escai

# Or install with custom values
helm install escai ./helm/escai \
  --namespace escai \
  --values values-production.yaml
```

### 3. Check Deployment Status

```bash
# Check pods
kubectl get pods -n escai

# Check services
kubectl get services -n escai

# Check ingress
kubectl get ingress -n escai

# View logs
kubectl logs -f deployment/escai -n escai
```

### 4. Access the API

```bash
# Port forward for local access
kubectl port-forward svc/escai 8000:8000 -n escai

# Or configure ingress for external access
# Update values.yaml with your domain
```

## Option 3: Manual Kubernetes Deployment

### 1. Apply Manifests

```bash
# Apply in order
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/escai-api.yaml

# Apply security policies (optional)
kubectl apply -f k8s/security/
```

### 2. Verify Deployment

```bash
kubectl get all -n escai
kubectl describe deployment escai-api -n escai
```

## Configuration Options

### Environment Variables

Key environment variables you can customize:

```yaml
# Application
ESCAI_ENV: "production"
LOG_LEVEL: "INFO"

# Database URLs
DATABASE_URL: "postgresql://user:pass@host:5432/db"
MONGODB_URL: "mongodb://host:27017/db"
REDIS_URL: "redis://host:6379/0"
INFLUXDB_URL: "http://host:8086"
NEO4J_URL: "bolt://host:7687"

# Security
JWT_SECRET_KEY: "your-secret-key"
```

### Resource Allocation

Adjust resources based on your needs:

```yaml
resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: 500m
    memory: 1Gi
```

### Scaling

Enable autoscaling:

```yaml
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## Health Checks

The API provides several health check endpoints:

- `/health` - Comprehensive health check
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe
- `/metrics` - Prometheus metrics

## Monitoring

### Prometheus Metrics

Access Prometheus at `http://localhost:9090` (Docker Compose) or configure ingress for Kubernetes.

Key metrics to monitor:

- `escai_requests_total`
- `escai_active_agents`
- `escai_events_processed_total`
- `escai_prediction_accuracy`

### Grafana Dashboards

Access Grafana at `http://localhost:3000` with credentials `admin/admin_password`.

Pre-configured dashboards:

- ESCAI Overview
- Agent Monitoring
- Database Performance
- API Metrics

## Troubleshooting

### Common Issues

1. **Pods not starting**:

```bash
kubectl describe pod <pod-name> -n escai
kubectl logs <pod-name> -n escai
```

2. **Database connection issues**:

```bash
# Check database pod status
kubectl get pods -l app=postgres -n escai

# Test connection
kubectl exec -it <api-pod> -n escai -- python -c "
from escai_framework.storage.database import get_db_connection
print(get_db_connection())
"
```

3. **High resource usage**:

```bash
kubectl top pods -n escai
kubectl top nodes
```

### Performance Tuning

1. **Adjust resource limits** based on actual usage
2. **Enable horizontal pod autoscaling** for API services
3. **Optimize database configurations** for your workload
4. **Use faster storage classes** for production

## Security Considerations

### Production Checklist

- [ ] Change default passwords
- [ ] Use TLS certificates
- [ ] Enable network policies
- [ ] Configure RBAC
- [ ] Use secrets for sensitive data
- [ ] Enable audit logging
- [ ] Regular security updates

### TLS Configuration

```yaml
ingress:
  tls:
    - secretName: escai-tls
      hosts:
        - api.escai.example.com
```

## Next Steps

1. **Configure monitoring** with Prometheus and Grafana
2. **Set up alerting** for critical issues
3. **Implement backup strategies** for databases
4. **Configure log aggregation** (ELK stack, Fluentd)
5. **Set up CI/CD pipelines** for automated deployments

## Support

For issues and questions:

- Check the [troubleshooting guide](troubleshooting.md)
- Review logs and metrics
- Open an issue on GitHub
- Consult the [full deployment documentation](README.md)
