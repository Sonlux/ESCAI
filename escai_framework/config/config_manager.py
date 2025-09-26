"""
Configuration manager for the ESCAI framework.

This module provides comprehensive configuration management including loading,
validation, hot-reloading, versioning, and secure storage.
"""

import os
import json
import yaml
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from datetime import datetime
import logging

# Optional watchdog import for file system monitoring
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    # Create mock classes when watchdog is not available
    class MockEvent:
        """Mock file system event."""
        def __init__(self, src_path="", is_directory=False):
            self.src_path = src_path
            self.is_directory = is_directory
    
    class FileSystemEventHandler:
        """Mock FileSystemEventHandler when watchdog is not available."""
        def __init__(self, *args, **kwargs):
            pass
        
        def on_modified(self, event):
            pass
    
    class Observer:
        """Mock Observer when watchdog is not available."""
        def __init__(self):
            pass
        
        def schedule(self, *args, **kwargs):
            pass
        
        def start(self):
            pass
        
        def stop(self):
            pass
        
        def join(self):
            pass
    
    WATCHDOG_AVAILABLE = False

from .config_schema import ConfigSchema, Environment
from .config_validator import ConfigValidator, ConfigValidationError
from .config_encryption import ConfigEncryption, ConfigEncryptionError
from .config_versioning import ConfigVersioning
from .config_templates import ConfigTemplates


logger = logging.getLogger(__name__)


class ConfigChangeHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reloading."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.debounce_delay = 1.0  # seconds
        self.last_modified: Dict[str, float] = {}
    
    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        current_time = datetime.now().timestamp()
        
        # Debounce rapid file changes
        if (file_path in self.last_modified and 
            current_time - self.last_modified[file_path] < self.debounce_delay):
            return
        
        self.last_modified[file_path] = current_time
        
        if file_path in self.config_manager.watched_files:
            logger.info(f"Configuration file changed: {file_path}")
            asyncio.create_task(self.config_manager._reload_config_async())


class ConfigManager:
    """Comprehensive configuration manager for the ESCAI framework."""
    
    def __init__(self, 
                 config_dir: str = "config",
                 environment: Optional[str] = None,
                 enable_hot_reload: bool = True,
                 enable_encryption: bool = True,
                 enable_versioning: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Target environment (development, testing, staging, production)
            enable_hot_reload: Enable configuration hot-reloading
            enable_encryption: Enable configuration encryption
            enable_versioning: Enable configuration versioning
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv('ESCAI_ENV', 'development')
        self.enable_hot_reload = enable_hot_reload
        self.enable_encryption = enable_encryption
        self.enable_versioning = enable_versioning
        
        # Initialize components
        self.validator = ConfigValidator()
        self.encryption: Optional[Any] = None
        self.versioning: Optional[Any] = None
        self.templates = ConfigTemplates()
        
        # Configuration state
        self.config: Optional[ConfigSchema] = None
        self.config_data: Dict[str, Any] = {}
        self.watched_files: List[str] = []
        self.change_callbacks: List[Callable[[ConfigSchema], None]] = []
        
        # Hot-reload components
        self.observer: Optional[Any] = None
        self.reload_lock = threading.Lock()
        
        # Initialize encryption if enabled
        if self.enable_encryption:
            self._initialize_encryption()
        
        # Initialize versioning if enabled
        if self.enable_versioning:
            self.versioning = ConfigVersioning(str(self.config_dir / "versions"))
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_encryption(self) -> None:
        """Initialize configuration encryption."""
        try:
            # Try to load existing master key
            key_file = self.config_dir / ".master_key"
            master_key = None
            
            if key_file.exists():
                master_key = None  # Will be loaded from file
            else:
                # Generate new master key
                master_key = ConfigEncryption.generate_master_key()
                logger.info("Generated new master encryption key")
            
            self.encryption = ConfigEncryption(
                master_key=master_key,
                key_file=str(key_file) if key_file.exists() else None
            )
            
            # Save master key if it was generated
            if master_key and not key_file.exists():
                self.encryption.save_master_key(str(key_file))
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.enable_encryption = False
    
    def load_config(self, config_file: Optional[str] = None) -> ConfigSchema:
        """
        Load configuration from file or environment-specific defaults.
        
        Args:
            config_file: Specific configuration file to load
            
        Returns:
            Loaded and validated configuration
        """
        try:
            # Determine configuration file
            if config_file:
                config_path = Path(config_file)
            else:
                config_path = self._get_environment_config_path()
            
            # Load configuration data
            if config_path.exists():
                self.config_data = self._load_config_file(config_path)
                logger.info(f"Loaded configuration from: {config_path}")
            else:
                # Generate default configuration
                self.config_data = self._generate_default_config()
                logger.info("Using default configuration")
            
            # Validate configuration
            is_valid, errors = self.validator.validate_config(self.config_data)
            if not is_valid:
                error_dicts = [{"error": error} for error in errors]
                raise ConfigValidationError("Configuration validation failed", error_dicts)
            
            # Create configuration object
            self.config = ConfigSchema(**self.config_data)
            
            # Save configuration if it was generated
            if not config_path.exists():
                self.save_config(str(config_path))
            
            # Set up hot-reload monitoring
            if self.enable_hot_reload:
                self._setup_hot_reload([str(config_path)])
            
            # Version the configuration
            if self.enable_versioning and self.versioning:
                serialized_data = self._serialize_config_data(self.config_data)
                self.versioning.save_version(serialized_data, f"Loaded from {config_path}")
            
            logger.info(f"Configuration loaded successfully for environment: {self.environment}")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _get_environment_config_path(self) -> Path:
        """Get configuration file path for current environment."""
        config_files = [
            self.config_dir / f"config.{self.environment}.yaml",
            self.config_dir / f"config.{self.environment}.yml",
            self.config_dir / f"config.{self.environment}.json",
            self.config_dir / "config.yaml",
            self.config_dir / "config.yml",
            self.config_dir / "config.json"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                return config_file
        
        # Return default path for environment
        return self.config_dir / f"config.{self.environment}.yaml"
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
            
            # Decrypt configuration if encryption is enabled
            if self.enable_encryption and self.encryption:
                config_data = self.encryption.decrypt_config(config_data)
            
            return config_data or {}
            
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration file: {e}")
    
    def _generate_default_config(self) -> Dict[str, Any]:
        """Generate default configuration for current environment."""
        return self.templates.generate_config_template(Environment(self.environment))
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_file: File path to save configuration
        """
        if not self.config:
            raise ValueError("No configuration loaded to save")
        
        try:
            # Determine save path
            if config_file:
                config_path = Path(config_file)
            else:
                config_path = self._get_environment_config_path()
            
            # Prepare configuration data (convert SecretStr to string)
            config_data = self.config.dict()
            config_data = self._serialize_config_data(config_data)
            
            # Encrypt sensitive values if encryption is enabled
            if self.enable_encryption and self.encryption:
                config_data = self.encryption.encrypt_config(config_data)
            
            # Save configuration
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
            
            # Version the configuration
            if self.enable_versioning and self.versioning:
                self.versioning.save_version(config_data, f"Saved to {config_path}")
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            validate: Whether to validate the updated configuration
        """
        try:
            # Apply updates to configuration data
            self._deep_update(self.config_data, updates)
            
            # Validate updated configuration
            if validate:
                is_valid, errors = self.validator.validate_config(self.config_data)
                if not is_valid:
                    error_dicts = [{"error": error} for error in errors]
                    raise ConfigValidationError("Updated configuration is invalid", error_dicts)
            
            # Create new configuration object
            old_config = self.config
            self.config = ConfigSchema(**self.config_data)
            
            # Version the configuration
            if self.enable_versioning and self.versioning:
                serialized_data = self._serialize_config_data(self.config_data)
                self.versioning.save_version(serialized_data, "Configuration updated")
            
            # Notify change callbacks
            self._notify_config_change(self.config)
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise
    
    def _deep_update(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep update dictionary with nested values."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def get_config(self) -> Optional[ConfigSchema]:
        """Get current configuration."""
        return self.config
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'database.postgres_host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self.config:
            return default
        
        try:
            value = self.config_data
            for key in key_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config_value(self, key_path: str, value: Any, validate: bool = True) -> None:
        """
        Set configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'database.postgres_host')
            value: Value to set
            validate: Whether to validate the updated configuration
        """
        keys = key_path.split('.')
        updates: Dict[str, Any] = {}
        current = updates
        
        for key in keys[:-1]:
            current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        self.update_config(updates, validate)
    
    def _setup_hot_reload(self, config_files: List[str]) -> None:
        """Set up hot-reload monitoring for configuration files."""
        if not self.enable_hot_reload:
            return
        
        try:
            self.watched_files = config_files
            
            if not WATCHDOG_AVAILABLE:
                logger.warning("Watchdog not available - hot-reload monitoring disabled")
                return
            
            if self.observer is not None:
                self.observer.stop()
                self.observer.join()
            
            self.observer = Observer()
            handler = ConfigChangeHandler(self)
            
            # Watch configuration directory
            self.observer.schedule(handler, str(self.config_dir), recursive=False)
            self.observer.start()
            
            logger.info("Configuration hot-reload monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to setup hot-reload: {e}")
    
    async def _reload_config_async(self) -> None:
        """Asynchronously reload configuration."""
        with self.reload_lock:
            try:
                logger.info("Reloading configuration...")
                
                # Reload configuration
                old_config = self.config
                self.load_config()
                
                # Notify change callbacks if configuration actually changed
                if old_config != self.config:
                    self._notify_config_change(self.config)
                    logger.info("Configuration reloaded successfully")
                else:
                    logger.debug("Configuration unchanged after reload")
                    
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
    
    def add_change_callback(self, callback: Callable[[ConfigSchema], None]) -> None:
        """
        Add callback to be notified of configuration changes.
        
        Args:
            callback: Function to call when configuration changes
        """
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[ConfigSchema], None]) -> None:
        """
        Remove configuration change callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def _notify_config_change(self, new_config: ConfigSchema) -> None:
        """Notify all registered callbacks of configuration change."""
        for callback in self.change_callbacks:
            try:
                callback(new_config)
            except Exception as e:
                logger.error(f"Configuration change callback failed: {e}")
    
    def validate_current_config(self) -> Dict[str, Any]:
        """
        Validate current configuration and return validation report.
        
        Returns:
            Validation report dictionary
        """
        if not self.config_data:
            return {
                'valid': False,
                'errors': ['No configuration loaded'],
                'warnings': [],
                'recommendations': []
            }
        
        return self.validator.generate_validation_report(self.config_data)
    
    def get_config_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get configuration version history.
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            List of configuration versions
        """
        if not self.enable_versioning or not self.versioning:
            return []
        
        return self.versioning.get_version_history(limit)
    
    def rollback_config(self, version_id: str) -> None:
        """
        Rollback configuration to a previous version.
        
        Args:
            version_id: Version ID to rollback to
        """
        if not self.enable_versioning or not self.versioning:
            raise ValueError("Configuration versioning is not enabled")
        
        try:
            # Get version data
            version_data = self.versioning.get_version(version_id)
            if not version_data:
                raise ValueError(f"Version not found: {version_id}")
            
            # Update configuration
            self.config_data = version_data['config']
            self.config = ConfigSchema(**self.config_data)
            
            # Save rolled back configuration
            self.save_config()
            
            # Notify change callbacks
            self._notify_config_change(self.config)
            
            logger.info(f"Configuration rolled back to version: {version_id}")
            
        except Exception as e:
            logger.error(f"Failed to rollback configuration: {e}")
            raise
    
    def export_config(self, export_path: str, include_sensitive: bool = False) -> None:
        """
        Export configuration to file.
        
        Args:
            export_path: Path to export configuration
            include_sensitive: Whether to include sensitive values
        """
        if not self.config:
            raise ValueError("No configuration loaded to export")
        
        try:
            export_data = self.config.dict()
            
            # Remove sensitive values if requested
            if not include_sensitive:
                export_data = self._mask_sensitive_values(export_data)
            
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                if export_file.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                elif export_file.suffix.lower() == '.json':
                    json.dump(export_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported export format: {export_file.suffix}")
            
            logger.info(f"Configuration exported to: {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise
    
    def _serialize_config_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SecretStr and other non-serializable objects to serializable format."""
        from pydantic import SecretStr
        
        serialized_data: Dict[str, Any] = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                serialized_data[key] = self._serialize_config_data(value)
            elif isinstance(value, SecretStr):
                serialized_data[key] = value.get_secret_value()
            else:
                serialized_data[key] = value
        
        return serialized_data
    
    def _mask_sensitive_values(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive values in configuration data."""
        masked_data: Dict[str, Any] = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                masked_data[key] = self._mask_sensitive_values(value)
            elif self.encryption and self.encryption.is_sensitive_field(key):
                masked_data[key] = "***MASKED***"
            else:
                masked_data[key] = value
        
        return masked_data
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            logger.info("Configuration hot-reload monitoring stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()