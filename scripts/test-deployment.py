#!/usr/bin/env python3
"""
Test script for ESCAI Framework deployment validation.
"""

import asyncio
import subprocess
import sys
import time
import requests
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_docker_compose_config():
    """Test Docker Compose configuration validity."""
    print("Testing Docker Compose configuration...")
    success, stdout, stderr = run_command("docker-compose config --quiet")
    
    if success:
        print("✅ Docker Compose configuration is valid")
        return True
    else:
        print(f"❌ Docker Compose configuration error: {stderr}")
        return False

def test_dockerfile_build():
    """Test Dockerfile build process."""
    print("Testing Dockerfile build...")
    success, stdout, stderr = run_command("docker build -t escai-test:latest .")
    
    if success:
        print("✅ Dockerfile builds successfully")
        return True
    else:
        print(f"❌ Dockerfile build failed: {stderr}")
        return False

def test_kubernetes_manifests():
    """Test Kubernetes manifest validity."""
    print("Testing Kubernetes manifests...")
    k8s_files = [
        "k8s/namespace.yaml",
        "k8s/configmap.yaml",
        "k8s/postgres.yaml",
        "k8s/escai-api.yaml"
    ]
    
    all_valid = True
    for k8s_file in k8s_files:
        if Path(k8s_file).exists():
            success, stdout, stderr = run_command(f"kubectl apply --dry-run=client -f {k8s_file}")
            if success:
                print(f"✅ {k8s_file} is valid")
            else:
                print(f"❌ {k8s_file} validation failed: {stderr}")
                all_valid = False
        else:
            print(f"⚠️  {k8s_file} not found")
    
    return all_valid

def test_helm_chart():
    """Test Helm chart validity."""
    print("Testing Helm chart...")
    success, stdout, stderr = run_command("helm lint helm/escai")
    
    if success:
        print("✅ Helm chart is valid")
        return True
    else:
        print(f"❌ Helm chart validation failed: {stderr}")
        return False

def test_api_health_check():
    """Test API health check endpoint."""
    print("Testing API health check...")
    try:
        # This would require the API to be running
        # For now, just check if the endpoint exists in the code
        with open("escai_framework/api/main.py", "r") as f:
            content = f.read()
            if "/health" in content and "health_check" in content:
                print("✅ Health check endpoint exists in API")
                return True
            else:
                print("❌ Health check endpoint not found in API")
                return False
    except Exception as e:
        print(f"❌ Error checking API health endpoint: {e}")
        return False

def main():
    """Run all deployment tests."""
    print("🚀 ESCAI Framework Deployment Validation")
    print("=" * 50)
    
    tests = [
        ("Docker Compose Configuration", test_docker_compose_config),
        ("Dockerfile Build", test_dockerfile_build),
        ("Kubernetes Manifests", test_kubernetes_manifests),
        ("Helm Chart", test_helm_chart),
        ("API Health Check", test_api_health_check),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All deployment tests passed!")
        return 0
    else:
        print("⚠️  Some deployment tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())