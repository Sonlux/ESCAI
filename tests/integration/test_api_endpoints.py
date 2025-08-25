"""
Integration tests for ESCAI Framework API endpoints.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from fastapi import status

from escai_framework.api.main import app
from escai_framework.api.auth import auth_manager, UserRole
from escai_framework.models.epistemic_state import EpistemicState
from escai_framework.models.behavioral_pattern import BehavioralPattern
from escai_framework.models.prediction_result import PredictionResult

# Test client
client = TestClient(app)

# Test data
TEST_USER_CREDENTIALS = {
    "username": "admin",
    "password": "admin123"
}

TEST_RESEARCHER_CREDENTIALS = {
    "username": "researcher", 
    "password": "research123"
}

@pytest.fixture
def auth_headers():
    """Get authentication headers for admin user."""
    response = client.post("/api/v1/auth/login", json=TEST_USER_CREDENTIALS)
    assert response.status_code == status.HTTP_200_OK
    token_data = response.json()
    return {"Authorization": f"Bearer {token_data['access_token']}"}

@pytest.fixture
def researcher_headers():
    """Get authentication headers for researcher user."""
    response = client.post("/api/v1/auth/login", json=TEST_RESEARCHER_CREDENTIALS)
    assert response.status_code == status.HTTP_200_OK
    token_data = response.json()
    return {"Authorization": f"Bearer {token_data['access_token']}"}

class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_login_success(self):
        """Test successful login."""
        response = client.post("/api/v1/auth/login", json=TEST_USER_CREDENTIALS)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["username"] == "admin"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "wrongpassword"
        })
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_refresh_token(self):
        """Test token refresh."""
        # Login first
        login_response = client.post("/api/v1/auth/login", json=TEST_USER_CREDENTIALS)
        login_data = login_response.json()
        
        # Refresh token
        response = client.post("/api/v1/auth/refresh", json={
            "refresh_token": login_data["refresh_token"]
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    def test_get_current_user(self, auth_headers):
        """Test getting current user info."""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == "admin"
        assert "admin" in data["roles"]
    
    def test_unauthorized_access(self):
        """Test accessing protected endpoint without auth."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_create_user_admin_only(self, auth_headers):
        """Test creating user (admin only)."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpass123",
            "roles": ["viewer"]
        }
        
        response = client.post("/api/v1/auth/users", json=user_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"

class TestMonitoringEndpoints:
    """Test monitoring endpoints."""
    
    def test_start_monitoring_success(self, auth_headers):
        """Test starting monitoring session."""
        monitoring_request = {
            "agent_id": "test-agent-001",
            "framework": "langchain",
            "config": {},
            "monitoring_config": {
                "capture_epistemic_states": True,
                "max_events_per_second": 50
            }
        }
        
        with patch('escai_framework.api.monitoring.instrumentors') as mock_instrumentors:
            mock_instrumentor = AsyncMock()
            mock_instrumentors.__getitem__.return_value = mock_instrumentor
            
            response = client.post(
                "/api/v1/monitor/start", 
                json=monitoring_request, 
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["agent_id"] == "test-agent-001"
            assert data["framework"] == "langchain"
            assert data["status"] == "active"
            assert "session_id" in data
    
    def test_start_monitoring_unsupported_framework(self, auth_headers):
        """Test starting monitoring with unsupported framework."""
        monitoring_request = {
            "agent_id": "test-agent-001",
            "framework": "unsupported",
            "config": {}
        }
        
        response = client.post(
            "/api/v1/monitor/start", 
            json=monitoring_request, 
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_monitoring_status(self, auth_headers):
        """Test getting monitoring status."""
        # First start monitoring
        monitoring_request = {
            "agent_id": "test-agent-002",
            "framework": "langchain",
            "config": {}
        }
        
        with patch('escai_framework.api.monitoring.instrumentors') as mock_instrumentors:
            mock_instrumentor = AsyncMock()
            mock_instrumentor.get_monitoring_stats.return_value = {
                "events_captured": 10,
                "performance_overhead": 0.05,
                "error_count": 0,
                "last_activity": datetime.utcnow()
            }
            mock_instrumentors.__getitem__.return_value = mock_instrumentor
            
            start_response = client.post(
                "/api/v1/monitor/start", 
                json=monitoring_request, 
                headers=auth_headers
            )
            session_id = start_response.json()["session_id"]
            
            # Get status
            response = client.get(
                f"/api/v1/monitor/{session_id}/status", 
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["session_id"] == session_id
            assert data["agent_id"] == "test-agent-002"
            assert data["events_captured"] == 10
    
    def test_stop_monitoring(self, auth_headers):
        """Test stopping monitoring session."""
        # First start monitoring
        monitoring_request = {
            "agent_id": "test-agent-003",
            "framework": "langchain",
            "config": {}
        }
        
        with patch('escai_framework.api.monitoring.instrumentors') as mock_instrumentors:
            mock_instrumentor = AsyncMock()
            mock_instrumentor.stop_monitoring.return_value = AsyncMock()
            mock_instrumentors.__getitem__.return_value = mock_instrumentor
            
            start_response = client.post(
                "/api/v1/monitor/start", 
                json=monitoring_request, 
                headers=auth_headers
            )
            session_id = start_response.json()["session_id"]
            
            # Stop monitoring
            response = client.post(
                f"/api/v1/monitor/{session_id}/stop", 
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["session_id"] == session_id
            assert data["status"] == "stopped"
    
    def test_list_monitoring_sessions(self, auth_headers):
        """Test listing monitoring sessions."""
        response = client.get("/api/v1/monitor/sessions", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

class TestAnalysisEndpoints:
    """Test analysis endpoints."""
    
    def test_get_current_epistemic_state_not_found(self, auth_headers):
        """Test getting current epistemic state when none exists."""
        response = client.get(
            "/api/v1/epistemic/nonexistent-agent/current", 
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_search_epistemic_states(self, auth_headers):
        """Test searching epistemic states."""
        query = {
            "agent_id": "test-agent",
            "confidence_min": 0.5,
            "confidence_max": 1.0
        }
        
        response = client.post(
            "/api/v1/epistemic/search", 
            json=query, 
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
    
    def test_analyze_behavioral_patterns(self, researcher_headers):
        """Test analyzing behavioral patterns."""
        query = {
            "agent_id": "test-agent",
            "success_rate_min": 0.7
        }
        
        response = client.post(
            "/api/v1/patterns/analyze", 
            json=query, 
            headers=researcher_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    def test_analyze_causal_relationships(self, researcher_headers):
        """Test analyzing causal relationships."""
        analysis_request = {
            "agent_id": "test-agent",
            "min_confidence": 0.6,
            "max_relationships": 50
        }
        
        response = client.post(
            "/api/v1/causal/analyze", 
            json=analysis_request, 
            headers=researcher_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_current_predictions_not_found(self, auth_headers):
        """Test getting current predictions when none exist."""
        response = client.get(
            "/api/v1/predictions/nonexistent-agent/current", 
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_generate_prediction(self, researcher_headers):
        """Test generating performance prediction."""
        prediction_request = {
            "agent_id": "test-agent",
            "prediction_horizon": 120,
            "include_risk_factors": True,
            "include_interventions": True
        }
        
        response = client.post(
            "/api/v1/predictions/generate", 
            json=prediction_request, 
            headers=researcher_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "test-agent"
        assert "predicted_value" in data
        assert "confidence_score" in data
    
    def test_explain_behavior(self, auth_headers):
        """Test generating behavior explanation."""
        explanation_request = {
            "agent_id": "test-agent",
            "explanation_type": "comprehensive",
            "max_length": 1000
        }
        
        response = client.post(
            "/api/v1/explain/behavior", 
            json=explanation_request, 
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "test-agent"
        assert "explanation" in data
    
    def test_get_agent_summary(self, auth_headers):
        """Test getting agent summary."""
        response = client.get(
            "/api/v1/agents/test-agent/summary?days=7", 
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "test-agent"
        assert "summary_period" in data
        assert "behavioral_patterns" in data

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiting_login(self):
        """Test rate limiting on login endpoint."""
        # Make multiple rapid requests
        for i in range(6):  # Limit is 5/minute
            response = client.post("/api/v1/auth/login", json={
                "username": "admin",
                "password": "wrongpassword"
            })
            
            if i < 5:
                assert response.status_code == status.HTTP_401_UNAUTHORIZED
            else:
                assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_json(self, auth_headers):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/monitor/start",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, auth_headers):
        """Test validation of required fields."""
        response = client.post(
            "/api/v1/monitor/start",
            json={"agent_id": "test"},  # Missing framework
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_field_values(self, auth_headers):
        """Test validation of field values."""
        query = {
            "confidence_min": 1.5,  # Invalid: > 1.0
            "confidence_max": -0.5   # Invalid: < 0.0
        }
        
        response = client.post(
            "/api/v1/epistemic/search",
            json=query,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

class TestHealthCheck:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "ESCAI Framework API"
        assert data["version"] == "1.0.0"
    
    @patch('escai_framework.api.main.DatabaseManager')
    def test_health_check_healthy(self, mock_db_manager):
        """Test health check when services are healthy."""
        mock_db_instance = AsyncMock()
        mock_db_instance.health_check.return_value = {"overall": True}
        mock_db_manager.return_value = mock_db_instance
        
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
    
    @patch('escai_framework.api.main.DatabaseManager')
    def test_health_check_unhealthy(self, mock_db_manager):
        """Test health check when services are unhealthy."""
        mock_db_instance = AsyncMock()
        mock_db_instance.health_check.return_value = {"overall": False}
        mock_db_manager.return_value = mock_db_instance
        
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

class TestPermissions:
    """Test role-based access control."""
    
    def test_researcher_access_to_analysis(self, researcher_headers):
        """Test researcher can access analysis endpoints."""
        response = client.post(
            "/api/v1/patterns/analyze",
            json={"agent_id": "test"},
            headers=researcher_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_viewer_cannot_start_monitoring(self):
        """Test viewer cannot start monitoring (requires developer role)."""
        # Create viewer user first (would need admin)
        # For this test, we'll simulate by modifying the auth system temporarily
        pass  # This would require more complex setup
    
    def test_admin_can_access_all(self, auth_headers):
        """Test admin can access all endpoints."""
        # Test monitoring (requires developer)
        response = client.post(
            "/api/v1/monitor/start",
            json={
                "agent_id": "test",
                "framework": "langchain"
            },
            headers=auth_headers
        )
        
        # Should not fail due to permissions (may fail for other reasons)
        assert response.status_code != status.HTTP_403_FORBIDDEN

if __name__ == "__main__":
    pytest.main([__file__, "-v"])