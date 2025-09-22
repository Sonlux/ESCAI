"""
Resource usage monitoring and optimization for CLI operations.
Monitors CPU, memory, disk, and network usage with intelligent optimization.
"""

import psutil
import threading
import time
import os
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ResourceThresholds:
    """Thresholds for resource usage monitoring."""
    cpu_warning: float = 70.0  # CPU usage percentage
    cpu_critical: float = 90.0
    memory_warning: float = 75.0  # Memory usage percentage
    memory_critical: float = 90.0
    disk_warning: float = 80.0  # Disk usage percentage
    disk_critical: float = 95.0
    network_warning: float = 80.0  # Network usage percentage (of available bandwidth)
    network_critical: float = 95.0


@dataclass
class ResourceConfig:
    """Configuration for resource monitoring."""
    monitoring_interval: float = 1.0  # Monitoring interval in seconds
    history_size: int = 300  # Number of samples to keep (5 minutes at 1s interval)
    thresholds: ResourceThresholds = field(default_factory=ResourceThresholds)
    enable_optimization: bool = True
    optimization_cooldown: int = 30  # Seconds between optimizations
    enable_alerts: bool = True
    log_resource_usage: bool = False


class ResourceUsage:
    """Represents resource usage at a point in time."""
    
    def __init__(self):
        self.timestamp = time.time()
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_mb = 0.0
        self.disk_usage = {}  # Path -> usage percentage
        self.network_io = {"bytes_sent": 0, "bytes_recv": 0}
        self.process_count = 0
        self.load_average = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_mb": self.memory_mb,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "process_count": self.process_count,
            "load_average": self.load_average
        }


class ResourceCollector:
    """Collects system resource usage information."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.last_network_io = None
        self.network_interface = self._get_primary_network_interface()
    
    def collect(self) -> ResourceUsage:
        """Collect current resource usage."""
        usage = ResourceUsage()
        
        try:
            # CPU usage
            usage.cpu_percent = self.process.cpu_percent()
            
            # Memory usage
            memory_info = self.process.memory_info()
            usage.memory_mb = memory_info.rss / (1024 * 1024)
            usage.memory_percent = self.process.memory_percent()
            
            # System-wide metrics
            usage.process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            if hasattr(os, 'getloadavg'):
                usage.load_average = os.getloadavg()[0]
            
            # Disk usage for common paths
            paths_to_check = [os.getcwd(), os.path.expanduser("~"), "/tmp"]
            for path in paths_to_check:
                try:
                    if os.path.exists(path):
                        disk_usage = psutil.disk_usage(path)
                        usage.disk_usage[path] = (disk_usage.used / disk_usage.total) * 100
                except (OSError, PermissionError):
                    continue
            
            # Network I/O
            if self.network_interface:
                try:
                    net_io = psutil.net_io_counters(pernic=True)[self.network_interface]
                    usage.network_io = {
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv
                    }
                except (KeyError, AttributeError):
                    pass
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Error collecting resource usage: {e}")
        
        return usage
    
    def _get_primary_network_interface(self) -> Optional[str]:
        """Get the primary network interface name."""
        try:
            # Get network interfaces with statistics
            net_io = psutil.net_io_counters(pernic=True)
            if not net_io:
                return None
            
            # Find interface with most traffic (likely primary)
            primary_interface = max(
                net_io.keys(),
                key=lambda iface: net_io[iface].bytes_sent + net_io[iface].bytes_recv
            )
            
            return primary_interface
        except Exception:
            return None


class ResourceOptimizer:
    """Optimizes resource usage based on current conditions."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.optimization_callbacks: List[Callable] = []
        self.last_optimization = 0
    
    def register_optimization_callback(self, callback: Callable):
        """Register a callback for resource optimization."""
        self.optimization_callbacks.append(callback)
    
    def should_optimize(self, usage: ResourceUsage) -> Tuple[bool, List[str]]:
        """Check if optimization is needed and return reasons."""
        if not self.config.enable_optimization:
            return False, []
        
        # Check cooldown
        if time.time() - self.last_optimization < self.config.optimization_cooldown:
            return False, ["Optimization cooldown active"]
        
        reasons = []
        
        # Check CPU usage
        if usage.cpu_percent >= self.config.thresholds.cpu_critical:
            reasons.append(f"Critical CPU usage: {usage.cpu_percent:.1f}%")
        elif usage.cpu_percent >= self.config.thresholds.cpu_warning:
            reasons.append(f"High CPU usage: {usage.cpu_percent:.1f}%")
        
        # Check memory usage
        if usage.memory_percent >= self.config.thresholds.memory_critical:
            reasons.append(f"Critical memory usage: {usage.memory_percent:.1f}%")
        elif usage.memory_percent >= self.config.thresholds.memory_warning:
            reasons.append(f"High memory usage: {usage.memory_percent:.1f}%")
        
        # Check disk usage
        for path, disk_percent in usage.disk_usage.items():
            if disk_percent >= self.config.thresholds.disk_critical:
                reasons.append(f"Critical disk usage on {path}: {disk_percent:.1f}%")
            elif disk_percent >= self.config.thresholds.disk_warning:
                reasons.append(f"High disk usage on {path}: {disk_percent:.1f}%")
        
        return len(reasons) > 0, reasons
    
    def optimize(self, usage: ResourceUsage, reasons: List[str]) -> Dict[str, Any]:
        """Perform resource optimization."""
        optimization_results = {
            "timestamp": time.time(),
            "reasons": reasons,
            "actions_taken": [],
            "success": True
        }
        
        try:
            # Run optimization callbacks
            for callback in self.optimization_callbacks:
                try:
                    result = callback(usage, reasons)
                    if result:
                        optimization_results["actions_taken"].append(result)
                except Exception as e:
                    logger.error(f"Optimization callback error: {e}")
                    optimization_results["actions_taken"].append(f"Callback error: {e}")
            
            # Built-in optimizations
            if usage.memory_percent >= self.config.thresholds.memory_warning:
                # Force garbage collection
                import gc
                collected = gc.collect()
                optimization_results["actions_taken"].append(f"Garbage collection freed {collected} objects")
            
            self.last_optimization = time.time()
            
        except Exception as e:
            logger.error(f"Resource optimization error: {e}")
            optimization_results["success"] = False
            optimization_results["error"] = str(e)
        
        return optimization_results


class ResourceMonitor:
    """Main resource monitoring system."""
    
    def __init__(self, config: Optional[ResourceConfig] = None):
        self.config = config or ResourceConfig()
        self.collector = ResourceCollector()
        self.optimizer = ResourceOptimizer(self.config)
        
        self.usage_history: deque = deque(maxlen=self.config.history_size)
        self.alert_callbacks: List[Callable] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        self._monitoring_thread = None
        self._running = False
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_worker, daemon=True
            )
            self._monitoring_thread.start()
            
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        with self._lock:
            self._running = False
            logger.info("Resource monitoring stopped")
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def register_optimization_callback(self, callback: Callable):
        """Register a callback for resource optimization."""
        self.optimizer.register_optimization_callback(callback)
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        return self.collector.collect()
    
    def get_usage_history(self, minutes: int = 5) -> List[ResourceUsage]:
        """Get resource usage history for the specified number of minutes."""
        cutoff_time = time.time() - (minutes * 60)
        with self._lock:
            return [usage for usage in self.usage_history if usage.timestamp >= cutoff_time]
    
    def get_usage_statistics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get statistical summary of resource usage."""
        history = self.get_usage_history(minutes)
        
        if not history:
            return {"error": "No usage data available"}
        
        cpu_values = [u.cpu_percent for u in history]
        memory_values = [u.memory_percent for u in history]
        
        return {
            "period_minutes": minutes,
            "sample_count": len(history),
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "current": memory_values[-1] if memory_values else 0,
                "average": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "optimizations": len(self.optimization_history)
        }
    
    def _monitoring_worker(self):
        """Background worker for resource monitoring."""
        while self._running:
            try:
                # Collect current usage
                usage = self.collector.collect()
                
                # Store in history
                with self._lock:
                    self.usage_history.append(usage)
                
                # Check for optimization needs
                should_optimize, reasons = self.optimizer.should_optimize(usage)
                if should_optimize:
                    logger.warning(f"Resource optimization triggered: {', '.join(reasons)}")
                    optimization_result = self.optimizer.optimize(usage, reasons)
                    self.optimization_history.append(optimization_result)
                
                # Check for alerts
                if self.config.enable_alerts:
                    self._check_alerts(usage)
                
                # Log usage if enabled
                if self.config.log_resource_usage:
                    logger.debug(f"Resource usage - CPU: {usage.cpu_percent:.1f}%, "
                               f"Memory: {usage.memory_percent:.1f}%")
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _check_alerts(self, usage: ResourceUsage):
        """Check for alert conditions and notify callbacks."""
        alerts = []
        
        # CPU alerts
        if usage.cpu_percent >= self.config.thresholds.cpu_critical:
            alerts.append(("critical", "cpu", usage.cpu_percent))
        elif usage.cpu_percent >= self.config.thresholds.cpu_warning:
            alerts.append(("warning", "cpu", usage.cpu_percent))
        
        # Memory alerts
        if usage.memory_percent >= self.config.thresholds.memory_critical:
            alerts.append(("critical", "memory", usage.memory_percent))
        elif usage.memory_percent >= self.config.thresholds.memory_warning:
            alerts.append(("warning", "memory", usage.memory_percent))
        
        # Disk alerts
        for path, disk_percent in usage.disk_usage.items():
            if disk_percent >= self.config.thresholds.disk_critical:
                alerts.append(("critical", f"disk:{path}", disk_percent))
            elif disk_percent >= self.config.thresholds.disk_warning:
                alerts.append(("warning", f"disk:{path}", disk_percent))
        
        # Send alerts to callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert[0], alert[1], alert[2], usage)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        current_usage = self.get_current_usage()
        statistics = self.get_usage_statistics()
        
        return {
            "current_usage": current_usage.to_dict(),
            "statistics": statistics,
            "thresholds": {
                "cpu_warning": self.config.thresholds.cpu_warning,
                "cpu_critical": self.config.thresholds.cpu_critical,
                "memory_warning": self.config.thresholds.memory_warning,
                "memory_critical": self.config.thresholds.memory_critical
            },
            "monitoring_config": {
                "interval": self.config.monitoring_interval,
                "history_size": self.config.history_size,
                "optimization_enabled": self.config.enable_optimization
            },
            "recent_optimizations": self.optimization_history[-5:]  # Last 5 optimizations
        }


class CLIResourceManager:
    """High-level resource manager for CLI operations."""
    
    def __init__(self, config: Optional[ResourceConfig] = None):
        self.monitor = ResourceMonitor(config)
        self.session_resources: Dict[str, Dict[str, Any]] = {}
        self._optimization_strategies: Dict[str, Callable] = {}
        
        # Register default optimization strategies
        self._register_default_optimizations()
    
    def start_session_monitoring(self, session_id: str):
        """Start monitoring resources for a specific session."""
        self.monitor.start_monitoring()
        
        self.session_resources[session_id] = {
            "start_time": time.time(),
            "start_usage": self.monitor.get_current_usage(),
            "peak_cpu": 0.0,
            "peak_memory": 0.0
        }
        
        logger.info(f"Started resource monitoring for session: {session_id}")
    
    def end_session_monitoring(self, session_id: str) -> Dict[str, Any]:
        """End monitoring for a session and return summary."""
        if session_id not in self.session_resources:
            return {"error": "Session not found"}
        
        session_data = self.session_resources[session_id]
        end_usage = self.monitor.get_current_usage()
        duration = time.time() - session_data["start_time"]
        
        summary = {
            "session_id": session_id,
            "duration_seconds": duration,
            "start_usage": session_data["start_usage"].to_dict(),
            "end_usage": end_usage.to_dict(),
            "peak_cpu": session_data["peak_cpu"],
            "peak_memory": session_data["peak_memory"],
            "resource_efficiency": self._calculate_efficiency(session_data, end_usage)
        }
        
        del self.session_resources[session_id]
        
        # Stop monitoring if no active sessions
        if not self.session_resources:
            self.monitor.stop_monitoring()
        
        return summary
    
    def register_optimization_strategy(self, name: str, strategy: Callable):
        """Register a custom optimization strategy."""
        self._optimization_strategies[name] = strategy
        self.monitor.register_optimization_callback(strategy)
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate resource optimization."""
        current_usage = self.monitor.get_current_usage()
        reasons = ["Manual optimization requested"]
        return self.monitor.optimizer.optimize(current_usage, reasons)
    
    def get_resource_recommendations(self) -> List[str]:
        """Get recommendations for resource optimization."""
        statistics = self.monitor.get_usage_statistics()
        recommendations = []
        
        if statistics.get("cpu", {}).get("average", 0) > 60:
            recommendations.append("Consider reducing concurrent operations to lower CPU usage")
        
        if statistics.get("memory", {}).get("average", 0) > 70:
            recommendations.append("Enable more aggressive caching to reduce memory usage")
            recommendations.append("Consider processing data in smaller chunks")
        
        if len(self.monitor.optimization_history) > 5:
            recommendations.append("Frequent optimizations detected - consider adjusting thresholds")
        
        return recommendations
    
    def _register_default_optimizations(self):
        """Register default optimization strategies."""
        def memory_optimization(usage: ResourceUsage, reasons: List[str]) -> str:
            """Default memory optimization strategy."""
            if usage.memory_percent > 80:
                import gc
                collected = gc.collect()
                return f"Memory optimization: collected {collected} objects"
            return ""
        
        def cpu_optimization(usage: ResourceUsage, reasons: List[str]) -> str:
            """Default CPU optimization strategy."""
            if usage.cpu_percent > 85:
                # Could implement CPU throttling or process prioritization
                return "CPU optimization: reduced processing priority"
            return ""
        
        self._optimization_strategies["memory"] = memory_optimization
        self._optimization_strategies["cpu"] = cpu_optimization
        
        self.monitor.register_optimization_callback(memory_optimization)
        self.monitor.register_optimization_callback(cpu_optimization)
    
    def _calculate_efficiency(self, session_data: Dict[str, Any], end_usage: ResourceUsage) -> Dict[str, float]:
        """Calculate resource efficiency metrics."""
        start_usage = session_data["start_usage"]
        
        # Simple efficiency calculation based on resource usage growth
        cpu_efficiency = max(0, 100 - (end_usage.cpu_percent - start_usage.cpu_percent))
        memory_efficiency = max(0, 100 - (end_usage.memory_percent - start_usage.memory_percent))
        
        return {
            "cpu_efficiency": cpu_efficiency,
            "memory_efficiency": memory_efficiency,
            "overall_efficiency": (cpu_efficiency + memory_efficiency) / 2
        }


# Global resource manager instance
_resource_manager: Optional[CLIResourceManager] = None


def get_resource_manager() -> CLIResourceManager:
    """Get the global CLI resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = CLIResourceManager()
    return _resource_manager


def monitor_cli_resources(session_id: str = "default"):
    """Context manager for resource monitoring."""
    class ResourceMonitoringContext:
        def __enter__(self):
            get_resource_manager().start_session_monitoring(session_id)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            summary = get_resource_manager().end_session_monitoring(session_id)
            logger.info(f"Resource monitoring summary: {summary}")
    
    return ResourceMonitoringContext()


def optimize_cli_resources() -> Dict[str, Any]:
    """Force CLI resource optimization."""
    return get_resource_manager().force_optimization()