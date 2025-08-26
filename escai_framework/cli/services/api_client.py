"""
API client for ESCAI CLI
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

import httpx
from rich.console import Console

class ESCAIAPIClient:
    """Client for interacting with ESCAI API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.console = Console()
        
        # Load config if available
        config_file = Path.home() / '.escai' / 'config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    api_config = config.get('api', {})
                    self.base_url = f"http://{api_config.get('host', 'localhost')}:{api_config.get('port', 8000)}"
            except Exception:
                pass
    
    async def start_monitoring(self, agent_id: str, framework: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start monitoring an agent"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/monitor/start",
                    json={
                        "agent_id": agent_id,
                        "framework": framework,
                        "config": config
                    },
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return {}
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return {}
    
    async def stop_monitoring(self, session_id: str) -> Dict[str, Any]:
        """Stop monitoring a session"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/monitor/{session_id}/stop",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return {}
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return {}
    
    async def get_agent_status(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get agent status information"""
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}/api/v1/agents"
                if agent_id:
                    url += f"/{agent_id}"
                
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                
                data = response.json()
                return data if isinstance(data, list) else [data]
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return []
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return []
    
    async def get_epistemic_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current epistemic state for an agent"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/epistemic/{agent_id}/current",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return {}
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return {}
    
    async def analyze_patterns(self, agent_id: Optional[str] = None, timeframe: str = "24h") -> List[Dict[str, Any]]:
        """Analyze behavioral patterns"""
        async with httpx.AsyncClient() as client:
            try:
                params = {"timeframe": timeframe}
                if agent_id:
                    params["agent_id"] = agent_id
                
                response = await client.get(
                    f"{self.base_url}/api/v1/patterns/analyze",
                    params=params,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return []
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return []
    
    async def analyze_causal_relationships(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Analyze causal relationships"""
        async with httpx.AsyncClient() as client:
            try:
                params = {}
                if agent_id:
                    params["agent_id"] = agent_id
                
                response = await client.post(
                    f"{self.base_url}/api/v1/causal/analyze",
                    json=params,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return []
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return []
    
    async def get_predictions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get performance predictions for an agent"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/predictions/{agent_id}/current",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return []
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return []
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers   
 
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generic GET request to API endpoint"""
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}{endpoint}"
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return {}
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return {}
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generic POST request to API endpoint"""
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}{endpoint}"
                response = await client.post(url, json=data, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.console.print(f"[error]API request failed: {e}[/error]")
                return {}
            except httpx.HTTPStatusError as e:
                self.console.print(f"[error]API error {e.response.status_code}: {e.response.text}[/error]")
                return {}
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers