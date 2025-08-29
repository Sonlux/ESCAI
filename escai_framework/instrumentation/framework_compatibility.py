"""
Framework compatibility and version management for robust integration.

This module provides version checking, compatibility validation, and automatic
adaptation mechanisms for different framework versions and configurations.
"""

import asyncio
import importlib
import logging
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, cast
from packaging import version
import inspect

from ..utils.exceptions import FrameworkNotSupportedError, InstrumentationError
from ..utils.retry import retry_async, RetryConfig, BackoffStrategy
from ..utils.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig


logger = logging.getLogger(__name__)


class FrameworkStatus(Enum):
    """Status of framework availability and compatibility."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


class AdaptationStrategy(Enum):
    """Strategies for adapting to framework changes."""
    VERSION_SPECIFIC = "version_specific"
    FEATURE_DETECTION = "feature_detection"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    COMPATIBILITY_LAYER = "compatibility_layer"


@dataclass
class FrameworkVersion:
    """Framework version information."""
    name: str
    version: str
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    
    @classmethod
    def from_string(cls, name: str, version_str: str) -> 'FrameworkVersion':
        """Create FrameworkVersion from version string."""
        try:
            parsed = version.parse(version_str)
            return cls(
                name=name,
                version=version_str,
                major=parsed.major,
                minor=parsed.minor,
                patch=parsed.micro,
                pre_release=str(parsed.pre) if parsed.pre else None
            )
        except Exception as e:
            logger.warning(f"Failed to parse version {version_str} for {name}: {e}")
            return cls(
                name=name,
                version=version_str,
                major=0,
                minor=0,
                patch=0
            )
    
    def is_compatible_with(self, min_version: str, max_version: Optional[str] = None) -> bool:
        """Check if this version is compatible with the given range."""
        try:
            current = version.parse(self.version)
            min_ver = version.parse(min_version)
            
            if current < min_ver:
                return False
            
            if max_version:
                max_ver = version.parse(max_version)
                if current > max_ver:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Version compatibility check failed: {e}")
            return False
    
    def __str__(self) -> str:
        return f"{self.name} {self.version}"


@dataclass
class CompatibilityRequirement:
    """Compatibility requirement for a framework."""
    min_version: str
    max_version: Optional[str] = None
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)
    deprecated_features: List[str] = field(default_factory=list)
    breaking_changes: Dict[str, str] = field(default_factory=dict)  # version -> description


@dataclass
class FrameworkInfo:
    """Complete information about a framework."""
    name: str
    status: FrameworkStatus
    version: Optional[FrameworkVersion] = None
    compatibility: Optional[CompatibilityRequirement] = None
    available_features: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.FEATURE_DETECTION
    error_message: Optional[str] = None


class FrameworkDetector(ABC):
    """Abstract base class for framework detection and validation."""
    
    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return the name of the framework this detector handles."""
        pass
    
    @abstractmethod
    async def detect_framework(self) -> FrameworkInfo:
        """Detect and validate the framework."""
        pass
    
    @abstractmethod
    def get_compatibility_requirements(self) -> CompatibilityRequirement:
        """Get compatibility requirements for this framework."""
        pass
    
    @abstractmethod
    async def validate_features(self, framework_module: Any) -> Tuple[List[str], List[str]]:
        """Validate available and missing features."""
        pass


class LangChainDetector(FrameworkDetector):
    """Detector for LangChain framework."""
    
    @property
    def framework_name(self) -> str:
        return "langchain"
    
    async def detect_framework(self) -> FrameworkInfo:
        """Detect LangChain framework and version."""
        try:
            import langchain
            version_str = getattr(langchain, '__version__', '0.0.0')
            framework_version = FrameworkVersion.from_string("langchain", version_str)
            
            # Check compatibility
            requirements = self.get_compatibility_requirements()
            is_compatible = framework_version.is_compatible_with(
                requirements.min_version, 
                requirements.max_version
            )
            
            if not is_compatible:
                return FrameworkInfo(
                    name=self.framework_name,
                    status=FrameworkStatus.INCOMPATIBLE,
                    version=framework_version,
                    compatibility=requirements,
                    error_message=f"LangChain version {version_str} is not compatible. "
                                f"Required: {requirements.min_version}+"
                )
            
            # Validate features
            available_features, missing_features = await self.validate_features(langchain)
            
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.AVAILABLE,
                version=framework_version,
                compatibility=requirements,
                available_features=available_features,
                missing_features=missing_features,
                adaptation_strategy=AdaptationStrategy.FEATURE_DETECTION
            )
            
        except ImportError as e:
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.UNAVAILABLE,
                error_message=f"LangChain not installed: {e}"
            )
        except Exception as e:
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.INCOMPATIBLE,
                error_message=f"LangChain detection failed: {e}"
            )
    
    def get_compatibility_requirements(self) -> CompatibilityRequirement:
        """Get LangChain compatibility requirements."""
        return CompatibilityRequirement(
            min_version="0.0.200",
            max_version="0.3.0",
            required_features=[
                "callbacks.base.BaseCallbackHandler",
                "schema.BaseMessage",
                "schema.AgentAction",
                "schema.AgentFinish"
            ],
            optional_features=[
                "schema.output.LLMResult",
                "schema.document.Document",
                "agents.AgentExecutor"
            ],
            deprecated_features=[
                "llms.base.LLM"  # Deprecated in favor of chat models
            ],
            breaking_changes={
                "0.1.0": "Major API restructuring",
                "0.2.0": "Schema changes for messages"
            }
        )
    
    async def validate_features(self, langchain_module: Any) -> Tuple[List[str], List[str]]:
        """Validate LangChain features."""
        requirements = self.get_compatibility_requirements()
        available_features = []
        missing_features = []
        
        all_features = requirements.required_features + requirements.optional_features
        
        for feature in all_features:
            try:
                # Navigate nested module path
                parts = feature.split('.')
                current = langchain_module
                
                for part in parts:
                    current = getattr(current, part)
                
                available_features.append(feature)
                logger.debug(f"LangChain feature available: {feature}")
                
            except AttributeError:
                missing_features.append(feature)
                logger.warning(f"LangChain feature missing: {feature}")
        
        return available_features, missing_features


class AutoGenDetector(FrameworkDetector):
    """Detector for AutoGen framework."""
    
    @property
    def framework_name(self) -> str:
        return "autogen"
    
    async def detect_framework(self) -> FrameworkInfo:
        """Detect AutoGen framework and version."""
        try:
            import autogen
            version_str = getattr(autogen, '__version__', '0.1.0')
            framework_version = FrameworkVersion.from_string("autogen", version_str)
            
            # Check compatibility
            requirements = self.get_compatibility_requirements()
            is_compatible = framework_version.is_compatible_with(
                requirements.min_version,
                requirements.max_version
            )
            
            if not is_compatible:
                return FrameworkInfo(
                    name=self.framework_name,
                    status=FrameworkStatus.INCOMPATIBLE,
                    version=framework_version,
                    compatibility=requirements,
                    error_message=f"AutoGen version {version_str} is not compatible. "
                                f"Required: {requirements.min_version}+"
                )
            
            # Validate features
            available_features, missing_features = await self.validate_features(autogen)
            
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.AVAILABLE,
                version=framework_version,
                compatibility=requirements,
                available_features=available_features,
                missing_features=missing_features,
                adaptation_strategy=AdaptationStrategy.VERSION_SPECIFIC
            )
            
        except ImportError as e:
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.UNAVAILABLE,
                error_message=f"AutoGen not installed: {e}"
            )
        except Exception as e:
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.INCOMPATIBLE,
                error_message=f"AutoGen detection failed: {e}"
            )
    
    def get_compatibility_requirements(self) -> CompatibilityRequirement:
        """Get AutoGen compatibility requirements."""
        return CompatibilityRequirement(
            min_version="0.1.0",
            max_version="0.3.0",
            required_features=[
                "ConversableAgent",
                "GroupChat",
                "GroupChatManager"
            ],
            optional_features=[
                "agentchat.conversable_agent.ConversableAgent",
                "coding.LocalCommandLineCodeExecutor"
            ],
            breaking_changes={
                "0.2.0": "Agent initialization changes"
            }
        )
    
    async def validate_features(self, autogen_module: Any) -> Tuple[List[str], List[str]]:
        """Validate AutoGen features."""
        requirements = self.get_compatibility_requirements()
        available_features = []
        missing_features = []
        
        all_features = requirements.required_features + requirements.optional_features
        
        for feature in all_features:
            try:
                parts = feature.split('.')
                current = autogen_module
                
                for part in parts:
                    current = getattr(current, part)
                
                available_features.append(feature)
                logger.debug(f"AutoGen feature available: {feature}")
                
            except AttributeError:
                missing_features.append(feature)
                logger.warning(f"AutoGen feature missing: {feature}")
        
        return available_features, missing_features


class CrewAIDetector(FrameworkDetector):
    """Detector for CrewAI framework."""
    
    @property
    def framework_name(self) -> str:
        return "crewai"
    
    async def detect_framework(self) -> FrameworkInfo:
        """Detect CrewAI framework and version."""
        try:
            import crewai
            version_str = getattr(crewai, '__version__', '0.1.0')
            framework_version = FrameworkVersion.from_string("crewai", version_str)
            
            # Check compatibility
            requirements = self.get_compatibility_requirements()
            is_compatible = framework_version.is_compatible_with(
                requirements.min_version,
                requirements.max_version
            )
            
            if not is_compatible:
                return FrameworkInfo(
                    name=self.framework_name,
                    status=FrameworkStatus.INCOMPATIBLE,
                    version=framework_version,
                    compatibility=requirements,
                    error_message=f"CrewAI version {version_str} is not compatible. "
                                f"Required: {requirements.min_version}+"
                )
            
            # Validate features
            available_features, missing_features = await self.validate_features(crewai)
            
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.AVAILABLE,
                version=framework_version,
                compatibility=requirements,
                available_features=available_features,
                missing_features=missing_features,
                adaptation_strategy=AdaptationStrategy.COMPATIBILITY_LAYER
            )
            
        except ImportError as e:
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.UNAVAILABLE,
                error_message=f"CrewAI not installed: {e}"
            )
        except Exception as e:
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.INCOMPATIBLE,
                error_message=f"CrewAI detection failed: {e}"
            )
    
    def get_compatibility_requirements(self) -> CompatibilityRequirement:
        """Get CrewAI compatibility requirements."""
        return CompatibilityRequirement(
            min_version="0.1.0",
            required_features=[
                "Agent",
                "Task",
                "Crew"
            ],
            optional_features=[
                "agent.Agent",
                "task.Task",
                "crew.Crew"
            ]
        )
    
    async def validate_features(self, crewai_module: Any) -> Tuple[List[str], List[str]]:
        """Validate CrewAI features."""
        requirements = self.get_compatibility_requirements()
        available_features = []
        missing_features = []
        
        all_features = requirements.required_features + requirements.optional_features
        
        for feature in all_features:
            try:
                parts = feature.split('.')
                current = crewai_module
                
                for part in parts:
                    current = getattr(current, part)
                
                available_features.append(feature)
                logger.debug(f"CrewAI feature available: {feature}")
                
            except AttributeError:
                missing_features.append(feature)
                logger.warning(f"CrewAI feature missing: {feature}")
        
        return available_features, missing_features


class OpenAIDetector(FrameworkDetector):
    """Detector for OpenAI library."""
    
    @property
    def framework_name(self) -> str:
        return "openai"
    
    async def detect_framework(self) -> FrameworkInfo:
        """Detect OpenAI library and version."""
        try:
            import openai
            version_str = getattr(openai, '__version__', '1.0.0')
            framework_version = FrameworkVersion.from_string("openai", version_str)
            
            # Check compatibility
            requirements = self.get_compatibility_requirements()
            is_compatible = framework_version.is_compatible_with(
                requirements.min_version,
                requirements.max_version
            )
            
            if not is_compatible:
                return FrameworkInfo(
                    name=self.framework_name,
                    status=FrameworkStatus.INCOMPATIBLE,
                    version=framework_version,
                    compatibility=requirements,
                    error_message=f"OpenAI version {version_str} is not compatible. "
                                f"Required: {requirements.min_version}+"
                )
            
            # Validate features
            available_features, missing_features = await self.validate_features(openai)
            
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.AVAILABLE,
                version=framework_version,
                compatibility=requirements,
                available_features=available_features,
                missing_features=missing_features,
                adaptation_strategy=AdaptationStrategy.VERSION_SPECIFIC
            )
            
        except ImportError as e:
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.UNAVAILABLE,
                error_message=f"OpenAI library not installed: {e}"
            )
        except Exception as e:
            return FrameworkInfo(
                name=self.framework_name,
                status=FrameworkStatus.INCOMPATIBLE,
                error_message=f"OpenAI detection failed: {e}"
            )
    
    def get_compatibility_requirements(self) -> CompatibilityRequirement:
        """Get OpenAI library compatibility requirements."""
        return CompatibilityRequirement(
            min_version="1.0.0",
            required_features=[
                "OpenAI",
                "types.beta.Assistant",
                "types.beta.Thread"
            ],
            optional_features=[
                "types.beta.threads.Run",
                "types.beta.threads.Message"
            ],
            breaking_changes={
                "1.0.0": "Complete API restructuring from v0.x"
            }
        )
    
    async def validate_features(self, openai_module: Any) -> Tuple[List[str], List[str]]:
        """Validate OpenAI library features."""
        requirements = self.get_compatibility_requirements()
        available_features = []
        missing_features = []
        
        all_features = requirements.required_features + requirements.optional_features
        
        for feature in all_features:
            try:
                parts = feature.split('.')
                current = openai_module
                
                for part in parts:
                    current = getattr(current, part)
                
                available_features.append(feature)
                logger.debug(f"OpenAI feature available: {feature}")
                
            except AttributeError:
                missing_features.append(feature)
                logger.warning(f"OpenAI feature missing: {feature}")
        
        return available_features, missing_features


class FrameworkCompatibilityManager:
    """Manages framework compatibility checking and adaptation."""
    
    def __init__(self):
        self.detectors: Dict[str, FrameworkDetector] = {
            "langchain": LangChainDetector(),
            "autogen": AutoGenDetector(),
            "crewai": CrewAIDetector(),
            "openai": OpenAIDetector()
        }
        self._framework_cache: Dict[str, FrameworkInfo] = {}
        self._circuit_breaker = get_circuit_breaker(
            "framework_compatibility",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                timeout=10.0
            )
        )
    
    def register_detector(self, detector: FrameworkDetector):
        """Register a custom framework detector."""
        self.detectors[detector.framework_name] = detector
        logger.info(f"Registered framework detector: {detector.framework_name}")
    
    @retry_async(
        max_attempts=3,
        base_delay=1.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER
    )
    async def detect_framework(self, framework_name: str, force_refresh: bool = False) -> FrameworkInfo:
        """
        Detect and validate a specific framework.
        
        Args:
            framework_name: Name of the framework to detect
            force_refresh: Whether to bypass cache and force re-detection
            
        Returns:
            FrameworkInfo with detection results
        """
        # Check cache first
        if not force_refresh and framework_name in self._framework_cache:
            return self._framework_cache[framework_name]
        
        if framework_name not in self.detectors:
            raise FrameworkNotSupportedError(
                framework_name,
                list(self.detectors.keys())
            )
        
        detector = self.detectors[framework_name]
        
        try:
            framework_info = await self._circuit_breaker.call_async(
                detector.detect_framework
            )
            
            # Cache the result
            self._framework_cache[framework_name] = framework_info
            
            logger.info(f"Framework detection completed: {framework_info.name} - {framework_info.status.value}")
            return framework_info
            
        except Exception as e:
            logger.error(f"Framework detection failed for {framework_name}: {e}")
            
            # Return error info
            error_info = FrameworkInfo(
                name=framework_name,
                status=FrameworkStatus.INCOMPATIBLE,
                error_message=str(e)
            )
            self._framework_cache[framework_name] = error_info
            return error_info
    
    async def detect_all_frameworks(self, force_refresh: bool = False) -> Dict[str, FrameworkInfo]:
        """
        Detect all supported frameworks.
        
        Args:
            force_refresh: Whether to bypass cache and force re-detection
            
        Returns:
            Dictionary mapping framework names to FrameworkInfo
        """
        results = {}
        
        # Run detections concurrently
        tasks = [
            self.detect_framework(name, force_refresh)
            for name in self.detectors.keys()
        ]
        
        framework_infos = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (name, result) in enumerate(zip(self.detectors.keys(), framework_infos)):
            if isinstance(result, Exception):
                results[name] = FrameworkInfo(
                    name=name,
                    status=FrameworkStatus.INCOMPATIBLE,
                    error_message=str(result)
                )
            else:
                results[name] = cast(FrameworkInfo, result)  # type: ignore[assignment]
        
        return results
    
    async def get_compatible_frameworks(self) -> List[FrameworkInfo]:
        """Get list of compatible frameworks."""
        all_frameworks = await self.detect_all_frameworks()
        return [
            info for info in all_frameworks.values()
            if info.status == FrameworkStatus.AVAILABLE
        ]
    
    async def validate_framework_configuration(
        self,
        framework_name: str,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate framework-specific configuration.
        
        Args:
            framework_name: Name of the framework
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        framework_info = await self.detect_framework(framework_name)
        
        if framework_info.status != FrameworkStatus.AVAILABLE:
            return False, [f"Framework {framework_name} is not available: {framework_info.error_message}"]
        
        errors = []
        
        # Validate required features are available
        if framework_info.compatibility:
            for required_feature in framework_info.compatibility.required_features:
                if required_feature not in framework_info.available_features:
                    errors.append(f"Required feature not available: {required_feature}")
        
        # Framework-specific validation
        if framework_name == "langchain":
            errors.extend(self._validate_langchain_config(config))
        elif framework_name == "autogen":
            errors.extend(self._validate_autogen_config(config))
        elif framework_name == "crewai":
            errors.extend(self._validate_crewai_config(config))
        elif framework_name == "openai":
            errors.extend(self._validate_openai_config(config))
        
        return len(errors) == 0, errors
    
    def _validate_langchain_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate LangChain-specific configuration."""
        errors = []
        
        # Check for required callback configuration
        if "callback_config" in config:
            callback_config = config["callback_config"]
            if not isinstance(callback_config, dict):
                errors.append("callback_config must be a dictionary")
        
        return errors
    
    def _validate_autogen_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate AutoGen-specific configuration."""
        errors = []
        
        # Check for agents configuration
        if "agents" in config:
            agents = config["agents"]
            if not isinstance(agents, list):
                errors.append("agents must be a list")
        
        return errors
    
    def _validate_crewai_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate CrewAI-specific configuration."""
        errors = []
        
        # Check for crews configuration
        if "crews" in config:
            crews = config["crews"]
            if not isinstance(crews, list):
                errors.append("crews must be a list")
        
        return errors
    
    def _validate_openai_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate OpenAI-specific configuration."""
        errors = []
        
        # Check for clients configuration
        if "clients" in config:
            clients = config["clients"]
            if not isinstance(clients, list):
                errors.append("clients must be a list")
        
        return errors
    
    def clear_cache(self):
        """Clear the framework detection cache."""
        self._framework_cache.clear()
        logger.info("Framework compatibility cache cleared")
    
    def get_framework_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Get compatibility matrix for all frameworks."""
        matrix = {}
        
        for name, detector in self.detectors.items():
            requirements = detector.get_compatibility_requirements()
            cached_info = self._framework_cache.get(name)
            
            matrix[name] = {
                "min_version": requirements.min_version,
                "max_version": requirements.max_version,
                "required_features": requirements.required_features,
                "optional_features": requirements.optional_features,
                "status": cached_info.status.value if cached_info else "unknown",
                "current_version": str(cached_info.version) if cached_info and cached_info.version else "unknown"
            }
        
        return matrix


# Global compatibility manager instance
_compatibility_manager = FrameworkCompatibilityManager()


def get_compatibility_manager() -> FrameworkCompatibilityManager:
    """Get the global framework compatibility manager."""
    return _compatibility_manager


async def detect_framework(framework_name: str, force_refresh: bool = False) -> FrameworkInfo:
    """Detect a specific framework using the global manager."""
    return await _compatibility_manager.detect_framework(framework_name, force_refresh)


async def get_compatible_frameworks() -> List[FrameworkInfo]:
    """Get list of compatible frameworks using the global manager."""
    return await _compatibility_manager.get_compatible_frameworks()