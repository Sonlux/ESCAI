"""
Configuration templates module for the ESCAI framework.

This module provides pre-configured templates for different deployment scenarios
and environments, making it easy to set up the framework in various contexts.
"""

from typing import Dict, Any
from .config_schema import Environment, LogLevel


class ConfigTemplates:
    """Configuration template generator for different deployment scenarios."""
    
    def generate_config_template(self, environment: Environment) -> Dict[str, Any]:
        """
        Generate configuration template for specified environment.
        
        Args:
            environment: Target environment
            
        Returns:
            Configuration template dictionary
        """
        base_config = self._get_base_config()
        
        if environment == Environment.DEVELOPMENT:
            return self._apply_development_overrides(base_config)
        elif environment == Environment.TESTING:
            return self._apply_testing_overrides(base_config)
        elif environment == Environment.STAGING:
            return self._apply_staging_overrides(base_config)
        elif environment == Environment.PRODUCTION:
            return self._apply_production_overrides(base_config)
        else:
            return base_config
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration template."""
        return {
            "environment": "development",
            "debug": False,
            "log_level": "INFO",
            
            "database": {
                "postgres_host": "localhost",
                "postgres_port": 5432,
                "postgres_database": "escai",
                "postgres_username": "escai_user",
                "postgres_password": "change_me_in_production",
                "postgres_pool_size": 10,
                "postgres_max_overflow": 20,
                
                "mongodb_host": "localhost",
                "mongodb_port": 27017,
                "mongodb_database": "escai_logs",
                "mongodb_username": None,
                "mongodb_password": None,
                "mongodb_replica_set": None,
                
                "redis_host": "localhost",
                "redis_port": 6379,
                "redis_database": 0,
                "redis_password": None,
                "redis_pool_size": 10,
                
                "influxdb_host": "localhost",
                "influxdb_port": 8086,
                "influxdb_database": "escai_metrics",
                "influxdb_username": None,
                "influxdb_password": None,
                "influxdb_retention_policy": "30d",
                
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_username": "neo4j",
                "neo4j_password": "change_me_in_production",
                "neo4j_database": "neo4j",
                "neo4j_pool_size": 10
            },
            
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "reload": False,
                "rate_limit_requests": 100,
                "rate_limit_window": 60,
                "cors_origins": ["*"],
                "cors_methods": ["GET", "POST", "PUT", "DELETE"],
                "cors_headers": ["*"],
                "max_request_size": 10485760,
                "request_timeout": 30,
                "websocket_max_connections": 100,
                "websocket_heartbeat_interval": 30
            },
            
            "security": {
                "jwt_secret_key": "change_me_to_a_secure_random_key_in_production",
                "jwt_algorithm": "HS256",
                "jwt_access_token_expire_minutes": 30,
                "jwt_refresh_token_expire_days": 7,
                "tls_enabled": False,
                "tls_cert_file": None,
                "tls_key_file": None,
                "tls_ca_file": None,
                "password_min_length": 8,
                "password_require_uppercase": True,
                "password_require_lowercase": True,
                "password_require_numbers": True,
                "password_require_symbols": True,
                "session_timeout_minutes": 60,
                "max_login_attempts": 5,
                "lockout_duration_minutes": 15,
                "pii_detection_enabled": True,
                "pii_masking_enabled": True,
                "pii_sensitivity_level": "medium",
                "audit_enabled": True,
                "audit_retention_days": 90
            },
            
            "monitoring": {
                "monitoring_enabled": True,
                "monitoring_overhead_threshold": 0.1,
                "sampling_rate": 1.0,
                "metrics_enabled": True,
                "metrics_interval_seconds": 60,
                "health_check_enabled": True,
                "health_check_interval_seconds": 30,
                "alerting_enabled": True,
                "alert_thresholds": {
                    "cpu_usage": 80.0,
                    "memory_usage": 85.0,
                    "disk_usage": 90.0,
                    "error_rate": 5.0
                }
            },
            
            "ml": {
                "model_cache_enabled": True,
                "model_cache_size": 100,
                "model_update_interval_hours": 24,
                "training_enabled": True,
                "training_batch_size": 32,
                "training_epochs": 10,
                "validation_split": 0.2,
                "prediction_confidence_threshold": 0.7,
                "ensemble_size": 5
            },
            
            "custom_settings": {}
        }
    
    def _apply_development_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply development environment overrides."""
        config.update({
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG"
        })
        
        config["api"].update({
            "reload": True,
            "workers": 1,
            "cors_origins": ["http://localhost:3000", "http://localhost:8080"]
        })
        
        config["security"].update({
            "tls_enabled": False,
            "jwt_access_token_expire_minutes": 60,
            "audit_enabled": False
        })
        
        config["monitoring"].update({
            "sampling_rate": 1.0,
            "metrics_interval_seconds": 30,
            "alerting_enabled": False
        })
        
        return config
    
    def _apply_testing_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply testing environment overrides."""
        config.update({
            "environment": "testing",
            "debug": False,
            "log_level": "WARNING"
        })
        
        # Use test databases
        config["database"].update({
            "postgres_database": "escai_test",
            "mongodb_database": "escai_logs_test",
            "influxdb_database": "escai_metrics_test",
            "redis_database": 1
        })
        
        config["api"].update({
            "reload": False,
            "workers": 1,
            "port": 8001
        })
        
        config["security"].update({
            "tls_enabled": False,
            "audit_enabled": False,
            "jwt_access_token_expire_minutes": 15
        })
        
        config["monitoring"].update({
            "sampling_rate": 0.1,
            "metrics_enabled": False,
            "alerting_enabled": False
        })
        
        config["ml"].update({
            "training_enabled": False,
            "model_cache_size": 10
        })
        
        return config
    
    def _apply_staging_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply staging environment overrides."""
        config.update({
            "environment": "staging",
            "debug": False,
            "log_level": "INFO"
        })
        
        # Use staging databases
        config["database"].update({
            "postgres_database": "escai_staging",
            "mongodb_database": "escai_logs_staging",
            "influxdb_database": "escai_metrics_staging"
        })
        
        config["api"].update({
            "reload": False,
            "workers": 2,
            "cors_origins": ["https://staging.example.com"]
        })
        
        config["security"].update({
            "tls_enabled": True,
            "audit_enabled": True,
            "jwt_access_token_expire_minutes": 30
        })
        
        config["monitoring"].update({
            "sampling_rate": 0.5,
            "alerting_enabled": True
        })
        
        return config
    
    def _apply_production_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply production environment overrides."""
        config.update({
            "environment": "production",
            "debug": False,
            "log_level": "WARNING"
        })
        
        config["api"].update({
            "reload": False,
            "workers": 8,
            "cors_origins": ["https://app.example.com"],
            "rate_limit_requests": 1000,
            "websocket_max_connections": 500
        })
        
        config["security"].update({
            "tls_enabled": True,
            "audit_enabled": True,
            "jwt_access_token_expire_minutes": 15,
            "session_timeout_minutes": 30,
            "pii_sensitivity_level": "high"
        })
        
        config["monitoring"].update({
            "sampling_rate": 0.1,
            "monitoring_overhead_threshold": 0.05,
            "alerting_enabled": True,
            "alert_thresholds": {
                "cpu_usage": 70.0,
                "memory_usage": 80.0,
                "disk_usage": 85.0,
                "error_rate": 2.0
            }
        })
        
        config["ml"].update({
            "model_cache_size": 500,
            "training_batch_size": 64,
            "ensemble_size": 10
        })
        
        return config
    
    def generate_docker_compose_template(self, environment: Environment) -> str:
        """
        Generate Docker Compose template for specified environment.
        
        Args:
            environment: Target environment
            
        Returns:
            Docker Compose YAML content
        """
        if environment == Environment.DEVELOPMENT:
            return self._get_development_docker_compose()
        elif environment == Environment.PRODUCTION:
            return self._get_production_docker_compose()
        else:
            return self._get_basic_docker_compose()
    
    def _get_development_docker_compose(self) -> str:
        """Get Docker Compose template for development."""
        return """version: '3.8'

services:
  escai-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ESCAI_ENV=development
      - ESCAI_DEBUG=true
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      - postgres
      - mongodb
      - redis
      - influxdb
      - neo4j
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: escai
      POSTGRES_USER: escai_user
      POSTGRES_PASSWORD: escai_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  mongodb:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: password
      DOCKER_INFLUXDB_INIT_ORG: escai
      DOCKER_INFLUXDB_INIT_BUCKET: metrics
    volumes:
      - influxdb_data:/var/lib/influxdb2
    restart: unless-stopped

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  mongodb_data:
  redis_data:
  influxdb_data:
  neo4j_data:
"""
    
    def _get_production_docker_compose(self) -> str:
        """Get Docker Compose template for production."""
        return """version: '3.8'

services:
  escai-api:
    image: escai/framework:latest
    ports:
      - "8000:8000"
    environment:
      - ESCAI_ENV=production
      - ESCAI_DEBUG=false
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./certs:/app/certs:ro
    depends_on:
      - postgres
      - mongodb
      - redis
      - influxdb
      - neo4j
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: escai
      POSTGRES_USER: escai_user
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    secrets:
      - postgres_password
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  mongodb:
    image: mongo:6
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD_FILE: /run/secrets/mongodb_password
    volumes:
      - mongodb_data:/data/db
      - ./mongodb/mongod.conf:/etc/mongod.conf:ro
    secrets:
      - mongodb_password
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  redis:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf --requirepass $(cat /run/secrets/redis_password)
    volumes:
      - redis_data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
    secrets:
      - redis_password
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  influxdb:
    image: influxdb:2.7
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD_FILE: /run/secrets/influxdb_password
      DOCKER_INFLUXDB_INIT_ORG: escai
      DOCKER_INFLUXDB_INIT_BUCKET: metrics
    volumes:
      - influxdb_data:/var/lib/influxdb2
      - ./influxdb/config.yml:/etc/influxdb2/config.yml:ro
    secrets:
      - influxdb_password
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  neo4j:
    image: neo4j:5-enterprise
    environment:
      NEO4J_AUTH: neo4j/$(cat /run/secrets/neo4j_password)
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"
    volumes:
      - neo4j_data:/data
      - ./neo4j/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf:ro
    secrets:
      - neo4j_password
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  mongodb_password:
    file: ./secrets/mongodb_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
  influxdb_password:
    file: ./secrets/influxdb_password.txt
  neo4j_password:
    file: ./secrets/neo4j_password.txt

volumes:
  postgres_data:
  mongodb_data:
  redis_data:
  influxdb_data:
  neo4j_data:

networks:
  default:
    driver: overlay
    attachable: true
"""
    
    def _get_basic_docker_compose(self) -> str:
        """Get basic Docker Compose template."""
        return """version: '3.8'

services:
  escai-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ESCAI_ENV=staging
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      - postgres
      - mongodb
      - redis
      - influxdb
      - neo4j
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: escai
      POSTGRES_USER: escai_user
      POSTGRES_PASSWORD: escai_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  mongodb:
    image: mongo:6
    volumes:
      - mongodb_data:/data/db
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  influxdb:
    image: influxdb:2.7
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: password
      DOCKER_INFLUXDB_INIT_ORG: escai
      DOCKER_INFLUXDB_INIT_BUCKET: metrics
    volumes:
      - influxdb_data:/var/lib/influxdb2
    restart: unless-stopped

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  mongodb_data:
  redis_data:
  influxdb_data:
  neo4j_data:
"""
    
    def generate_kubernetes_template(self, environment: Environment) -> Dict[str, str]:
        """
        Generate Kubernetes manifests for specified environment.
        
        Args:
            environment: Target environment
            
        Returns:
            Dictionary of Kubernetes manifest files
        """
        manifests = {}
        
        # ConfigMap
        manifests['configmap.yaml'] = self._get_kubernetes_configmap(environment)
        
        # Deployment
        manifests['deployment.yaml'] = self._get_kubernetes_deployment(environment)
        
        # Service
        manifests['service.yaml'] = self._get_kubernetes_service(environment)
        
        # Ingress (for production)
        if environment == Environment.PRODUCTION:
            manifests['ingress.yaml'] = self._get_kubernetes_ingress()
        
        return manifests
    
    def _get_kubernetes_configmap(self, environment: Environment) -> str:
        """Get Kubernetes ConfigMap manifest."""
        config = self.generate_config_template(environment)
        
        return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: escai-config
  namespace: escai
data:
  config.yaml: |
    environment: {config['environment']}
    debug: {str(config['debug']).lower()}
    log_level: {config['log_level']}
    
    api:
      host: "0.0.0.0"
      port: 8000
      workers: {config['api']['workers']}
      
    monitoring:
      monitoring_enabled: {str(config['monitoring']['monitoring_enabled']).lower()}
      sampling_rate: {config['monitoring']['sampling_rate']}
"""
    
    def _get_kubernetes_deployment(self, environment: Environment) -> str:
        """Get Kubernetes Deployment manifest."""
        replicas = 1 if environment == Environment.DEVELOPMENT else 3
        
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: escai-api
  namespace: escai
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: escai-api
  template:
    metadata:
      labels:
        app: escai-api
    spec:
      containers:
      - name: escai-api
        image: escai/framework:latest
        ports:
        - containerPort: 8000
        env:
        - name: ESCAI_ENV
          value: "{environment.value}"
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: escai-config
"""
    
    def _get_kubernetes_service(self, environment: Environment) -> str:
        """Get Kubernetes Service manifest."""
        return """apiVersion: v1
kind: Service
metadata:
  name: escai-api-service
  namespace: escai
spec:
  selector:
    app: escai-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
"""
    
    def _get_kubernetes_ingress(self) -> str:
        """Get Kubernetes Ingress manifest for production."""
        return """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: escai-ingress
  namespace: escai
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.escai.example.com
    secretName: escai-tls
  rules:
  - host: api.escai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: escai-api-service
            port:
              number: 80
"""