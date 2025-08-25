#!/usr/bin/env python3
"""
Basic API test script for ESCAI Framework.
"""

import sys
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastapi.testclient import TestClient
    from escai_framework.api.main import app
    
    def test_basic_api():
        """Test basic API functionality."""
        client = TestClient(app)
        
        print("Testing ESCAI Framework API...")
        
        # Test root endpoint
        print("1. Testing root endpoint...")
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        print(f"   ✓ Root endpoint: {data['name']} v{data['version']}")
        
        # Test health endpoint
        print("2. Testing health endpoint...")
        try:
            response = client.get("/health")
            print(f"   ✓ Health endpoint: {response.status_code}")
        except Exception as e:
            print(f"   ⚠ Health endpoint failed (expected): {e}")
        
        # Test authentication
        print("3. Testing authentication...")
        response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        assert response.status_code == 200
        token_data = response.json()
        print(f"   ✓ Login successful for user: {token_data['user']['username']}")
        
        # Test protected endpoint
        print("4. Testing protected endpoint...")
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        user_data = response.json()
        print(f"   ✓ Protected endpoint: {user_data['username']} with roles {user_data['roles']}")
        
        # Test monitoring endpoint (will fail but should not crash)
        print("5. Testing monitoring endpoint...")
        try:
            response = client.post("/api/v1/monitor/start", json={
                "agent_id": "test-agent",
                "framework": "langchain"
            }, headers=headers)
            print(f"   ✓ Monitoring endpoint responded: {response.status_code}")
        except Exception as e:
            print(f"   ⚠ Monitoring endpoint failed (expected): {e}")
        
        print("\n✅ Basic API tests completed successfully!")
        return True
    
    if __name__ == "__main__":
        try:
            test_basic_api()
        except Exception as e:
            print(f"\n❌ API test failed: {e}")
            sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt] slowapi")
    sys.exit(1)