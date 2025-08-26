#!/usr/bin/env python3
"""
Simple deployment validation script for ESCAI Framework.
"""

import os
import yaml
import json
from pathlib import Path

def validate_docker_compose():
    """Validate Docker Compose configuration."""
    print("🐳 Validating Docker Compose configuration...")
    
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    try:
        with open(compose_file) as f:
            compose_config = yaml.safe_load(f)
        
        # Check required services
        required_services = [
            "escai-api", "postgres", "mongo", "redis", 
            "influxdb", "neo4j", "prometheus", "grafana"
        ]
        
        services = compose_config.get("services", {})
        missing_services = [svc for svc in required_services if svc not in services]
        
        if missing_services:
            print(f"❌ Missing services: {missing_services}")
            return False
        
        print("✅ Docker Compose configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating Docker Compose: {e}")
        return False

def validate_kubernetes_manifests():
    """Validate Kubernetes manifest files."""
    print("☸️  Validating Kubernetes manifests...")
    
    k8s_files = [
        "k8s/namespace.yaml",
        "k8s/configmap.yaml", 
        "k8s/postgres.yaml",
        "k8s/escai-api.yaml"
    ]
    
    all_valid = True
    for k8s_file in k8s_files:
        file_path = Path(k8s_file)
        if not file_path.exists():
            print(f"⚠️  {k8s_file} not found")
            continue
            
        try:
            with open(file_path) as f:
                yaml.safe_load_all(f)
            print(f"✅ {k8s_file} is valid YAML")
        except Exception as e:
            print(f"❌ {k8s_file} validation failed: {e}")
            all_valid = False
    
    return all_valid

def validate_helm_chart():
    """Validate Helm chart structure."""
    print("⛵ Validating Helm chart...")
    
    chart_file = Path("helm/escai/Chart.yaml")
    values_file = Path("helm/escai/values.yaml")
    
    if not chart_file.exists():
        print("❌ helm/escai/Chart.yaml not found")
        return False
        
    if not values_file.exists():
        print("❌ helm/escai/values.yaml not found")
        return False
    
    try:
        with open(chart_file) as f:
            chart_config = yaml.safe_load(f)
        
        required_fields = ["apiVersion", "name", "description", "version"]
        missing_fields = [field for field in required_fields if field not in chart_config]
        
        if missing_fields:
            print(f"❌ Missing Chart.yaml fields: {missing_fields}")
            return False
            
        with open(values_file) as f:
            yaml.safe_load(f)
            
        print("✅ Helm chart structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating Helm chart: {e}")
        return False

def validate_dockerfile():
    """Validate Dockerfile exists and has basic structure."""
    print("🐋 Validating Dockerfile...")
    
    dockerfile = Path("Dockerfile")
    if not dockerfile.exists():
        print("❌ Dockerfile not found")
        return False
    
    try:
        with open(dockerfile) as f:
            content = f.read()
        
        required_instructions = ["FROM", "WORKDIR", "COPY", "RUN", "EXPOSE"]
        missing_instructions = [
            instr for instr in required_instructions 
            if instr not in content
        ]
        
        if missing_instructions:
            print(f"❌ Missing Dockerfile instructions: {missing_instructions}")
            return False
            
        print("✅ Dockerfile structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating Dockerfile: {e}")
        return False

def validate_ci_cd_pipeline():
    """Validate CI/CD pipeline configuration."""
    print("🔄 Validating CI/CD pipeline...")
    
    workflow_file = Path(".github/workflows/ci-cd.yml")
    if not workflow_file.exists():
        print("❌ .github/workflows/ci-cd.yml not found")
        return False
    
    try:
        with open(workflow_file) as f:
            workflow_config = yaml.safe_load(f)
        
        required_jobs = ["test", "build"]
        jobs = workflow_config.get("jobs", {})
        missing_jobs = [job for job in required_jobs if job not in jobs]
        
        if missing_jobs:
            print(f"❌ Missing CI/CD jobs: {missing_jobs}")
            return False
            
        print("✅ CI/CD pipeline configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating CI/CD pipeline: {e}")
        return False

def validate_health_endpoints():
    """Validate health check endpoints exist in API."""
    print("🏥 Validating health check endpoints...")
    
    api_file = Path("escai_framework/api/main.py")
    if not api_file.exists():
        print("❌ API main file not found")
        return False
    
    try:
        with open(api_file) as f:
            content = f.read()
        
        required_endpoints = ["/health", "/health/ready", "/health/live", "/metrics"]
        missing_endpoints = [
            endpoint for endpoint in required_endpoints 
            if endpoint not in content
        ]
        
        if missing_endpoints:
            print(f"❌ Missing health endpoints: {missing_endpoints}")
            return False
            
        print("✅ Health check endpoints are present")
        return True
        
    except Exception as e:
        print(f"❌ Error validating health endpoints: {e}")
        return False

def main():
    """Run all deployment validations."""
    print("🚀 ESCAI Framework Deployment Validation")
    print("=" * 50)
    
    validations = [
        ("Docker Compose", validate_docker_compose),
        ("Kubernetes Manifests", validate_kubernetes_manifests),
        ("Helm Chart", validate_helm_chart),
        ("Dockerfile", validate_dockerfile),
        ("CI/CD Pipeline", validate_ci_cd_pipeline),
        ("Health Endpoints", validate_health_endpoints),
    ]
    
    results = []
    for name, validation_func in validations:
        print(f"\n📋 {name}")
        try:
            result = validation_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Validation Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 All deployment validations passed!")
        print("\n📝 Next steps:")
        print("   1. Build Docker image: docker build -t escai:latest .")
        print("   2. Test locally: docker-compose up -d")
        print("   3. Deploy to Kubernetes: kubectl apply -f k8s/")
        print("   4. Or use Helm: helm install escai ./helm/escai")
        return 0
    else:
        print("⚠️  Some validations failed. Please review and fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())