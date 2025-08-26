"""
Unit tests for configuration management components.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from escai_framework.config import (
    ConfigManager, ConfigValidator, ConfigEncryption, 
    ConfigVersioning, ConfigTemplates, ConfigSchema, Environment
)


class TestConfigSchema:
    """Test configuration schema validation."""
    
    def test_valid_config_creation(self):
        """Test creating valid configuration."""
        config_data = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "database": {
                "postgres_password": "test_password",
                "neo4j_password": "test_password"
            },
            "security": {
                "jwt_secret_key": "test_secret_key_12345678901234567890"
            }
        }
        
        config = ConfigSchema(**config_data)
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is True
        assert config.log_level == "DEBUG"
    
    def test_invalid_environment(self):
        """Test invalid environment validation."""
        with pytest.raises(ValueError):
            ConfigSchema(
                environment="invalid_env",
                database={
                    "postgres_password": "test_password",
                    "neo4j_password": "test_password"
                },
                security={"jwt_secret_key": "test_key"}
            )
    
    def test_debug_in_production_validation(self):
        """Test debug mode validation in production."""
        with pytest.raises(ValueError, match="Debug mode must be disabled in production"):
            ConfigSchema(
                environment="production",
                debug=True,
                database={
                    "postgres_password": "test_password",
                    "neo4j_password": "test_password"
                },
                security={"jwt_secret_key": "test_key"}
            )


class TestConfigValidator:
    """Test configuration validator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
    
    def test_valid_config_validation(self):
        """Test validation of valid configuration."""
        config_data = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "database": {
                "postgres_host": "localhost",
                "postgres_port": 5432,
                "postgres_database": "test_db",
                "postgres_username": "test_user",
                "postgres_password": "test_password",
                "redis_database": 0,
                "influxdb_retention_policy": "30d",
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_username": "neo4j",
                "neo4j_password": "test_password"
            },
            "api": {
                "host": "localhost",
                "port": 8000,
                "workers": 1
            },
            "security": {
                "jwt_secret_key": "test_secret_key_12345678901234567890",
                "tls_enabled": False
            }
        }
        
        is_valid, errors = self.validator.validate_config(config_data)
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_port_validation(self):
        """Test invalid port validation."""
        config_data = {
            "environment": "development",
            "database": {
                "postgres_port": 70000,  # Invalid port
                "postgres_password": "test",
                "neo4j_password": "test"
            },
            "security": {
                "jwt_secret_key": "test_key"
            }
        }
        
        is_valid, errors = self.validator.validate_config(config_data)
        assert not is_valid
        assert any("port" in error.lower() for error in errors)
    
    def test_production_security_validation(self):
        """Test production security requirements."""
        config_data = {
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
            "database": {
                "postgres_password": "test",
                "neo4j_password": "test"
            },
            "security": {
                "jwt_secret_key": "short",  # Too short
                "tls_enabled": False  # Should be enabled in production
            }
        }
        
        is_valid, errors = self.validator.validate_config(config_data)
        assert not is_valid
        assert any("jwt secret key" in error.lower() for error in errors)
        assert any("tls must be enabled" in error.lower() for error in errors)
    
    def test_validation_report_generation(self):
        """Test validation report generation."""
        config_data = {
            "environment": "production",
            "debug": False,
            "database": {
                "postgres_password": "test",
                "neo4j_password": "test"
            },
            "security": {
                "jwt_secret_key": "test_secret_key_12345678901234567890",
                "tls_enabled": True
            },
            "monitoring": {
                "sampling_rate": 1.0  # Will generate warning
            },
            "api": {
                "workers": 1  # Will generate warning
            }
        }
        
        report = self.validator.generate_validation_report(config_data)
        
        assert "valid" in report
        assert "warnings" in report
        assert "recommendations" in report
        assert len(report["warnings"]) > 0


class TestConfigEncryption:
    """Test configuration encryption."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encryption = ConfigEncryption()
    
    def test_encrypt_decrypt_value(self):
        """Test basic encryption and decryption."""
        test_value = "sensitive_password_123"
        
        encrypted = self.encryption.encrypt_value(test_value)
        decrypted = self.encryption.decrypt_value(encrypted)
        
        assert decrypted == test_value
        assert encrypted != test_value
    
    def test_sensitive_field_detection(self):
        """Test sensitive field detection."""
        assert self.encryption.is_sensitive_field("password")
        assert self.encryption.is_sensitive_field("secret_key")
        assert self.encryption.is_sensitive_field("auth_token")
        assert not self.encryption.is_sensitive_field("host")
        assert not self.encryption.is_sensitive_field("port")
    
    def test_config_encryption(self):
        """Test configuration encryption."""
        config_data = {
            "database": {
                "host": "localhost",
                "password": "secret123",
                "port": 5432
            },
            "security": {
                "jwt_secret": "jwt_secret_key",
                "public_key": "not_really_secret"
            }
        }
        
        encrypted_config = self.encryption.encrypt_config(config_data)
        
        # Check that sensitive fields are encrypted
        assert encrypted_config["database"]["host"] == "localhost"  # Not encrypted
        assert encrypted_config["database"]["port"] == 5432  # Not encrypted
        assert encrypted_config["database"]["password"]["_encrypted"] is True
        assert encrypted_config["security"]["jwt_secret"]["_encrypted"] is True
    
    def test_config_decryption(self):
        """Test configuration decryption."""
        config_data = {
            "database": {
                "host": "localhost",
                "password": "secret123"
            }
        }
        
        encrypted_config = self.encryption.encrypt_config(config_data)
        decrypted_config = self.encryption.decrypt_config(encrypted_config)
        
        assert decrypted_config == config_data
    
    def test_encryption_verification(self):
        """Test encryption verification."""
        assert self.encryption.verify_encryption()
    
    def test_key_rotation(self):
        """Test encryption key rotation."""
        old_key = self.encryption.master_key
        new_key = self.encryption.rotate_encryption_key()
        
        assert new_key != old_key
        assert self.encryption.verify_encryption()


class TestConfigVersioning:
    """Test configuration versioning."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.versioning = ConfigVersioning(f"{self.temp_dir}/versions")
    
    def test_save_version(self):
        """Test saving configuration version."""
        config_data = {
            "environment": "development",
            "debug": True
        }
        
        version_id = self.versioning.save_version(config_data, "Initial version")
        
        assert version_id is not None
        assert len(version_id) == 16  # SHA256 hash truncated to 16 chars
        
        # Verify version can be retrieved
        version_data = self.versioning.get_version(version_id)
        assert version_data is not None
        assert version_data["config"] == config_data
    
    def test_version_history(self):
        """Test version history retrieval."""
        # Save multiple versions
        for i in range(3):
            config_data = {"version": i}
            self.versioning.save_version(config_data, f"Version {i}")
        
        history = self.versioning.get_version_history(limit=2)
        
        assert len(history) == 2
        assert all("id" in version for version in history)
        assert all("timestamp" in version for version in history)
    
    def test_version_comparison(self):
        """Test version comparison."""
        config1 = {"setting1": "value1", "setting2": "value2"}
        config2 = {"setting1": "modified", "setting3": "new_value"}
        
        version1_id = self.versioning.save_version(config1, "Version 1")
        version2_id = self.versioning.save_version(config2, "Version 2")
        
        comparison = self.versioning.compare_versions(version1_id, version2_id)
        
        assert comparison["total_changes"] == 3  # 1 modified, 1 removed, 1 added
        assert len(comparison["differences"]) == 3
    
    def test_version_tagging(self):
        """Test version tagging."""
        config_data = {"test": "data"}
        version_id = self.versioning.save_version(config_data, "Test version")
        
        success = self.versioning.tag_version(version_id, ["stable", "release"])
        assert success
        
        # Find versions by tag
        stable_versions = self.versioning.find_versions_by_tag("stable")
        assert len(stable_versions) == 1
        assert stable_versions[0]["id"] == version_id
    
    def test_version_cleanup(self):
        """Test old version cleanup."""
        # Save many versions
        for i in range(25):
            config_data = {"version": i}
            self.versioning.save_version(config_data, f"Version {i}")
        
        deleted_count = self.versioning.cleanup_old_versions(keep_count=10)
        
        assert deleted_count == 15
        remaining_versions = self.versioning.get_version_history(limit=100)
        assert len(remaining_versions) == 10


class TestConfigTemplates:
    """Test configuration templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.templates = ConfigTemplates()
    
    def test_development_template(self):
        """Test development environment template."""
        config = self.templates.generate_config_template(Environment.DEVELOPMENT)
        
        assert config["environment"] == "development"
        assert config["debug"] is True
        assert config["log_level"] == "DEBUG"
        assert config["api"]["reload"] is True
        assert config["security"]["tls_enabled"] is False
    
    def test_production_template(self):
        """Test production environment template."""
        config = self.templates.generate_config_template(Environment.PRODUCTION)
        
        assert config["environment"] == "production"
        assert config["debug"] is False
        assert config["log_level"] == "WARNING"
        assert config["api"]["reload"] is False
        assert config["security"]["tls_enabled"] is True
        assert config["api"]["workers"] == 8
    
    def test_docker_compose_template(self):
        """Test Docker Compose template generation."""
        compose_content = self.templates.generate_docker_compose_template(Environment.DEVELOPMENT)
        
        assert "version:" in compose_content
        assert "escai-api:" in compose_content
        assert "postgres:" in compose_content
        assert "mongodb:" in compose_content
        assert "redis:" in compose_content
    
    def test_kubernetes_template(self):
        """Test Kubernetes template generation."""
        manifests = self.templates.generate_kubernetes_template(Environment.PRODUCTION)
        
        assert "configmap.yaml" in manifests
        assert "deployment.yaml" in manifests
        assert "service.yaml" in manifests
        assert "ingress.yaml" in manifests  # Only in production
        
        # Check deployment content
        deployment = manifests["deployment.yaml"]
        assert "replicas: 3" in deployment  # Production should have 3 replicas


class TestConfigManager:
    """Test configuration manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(
            config_dir=self.temp_dir,
            enable_hot_reload=False,  # Disable for testing
            enable_encryption=False,  # Disable for testing
            enable_versioning=False   # Disable for testing
        )
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = self.config_manager.load_config()
        
        assert config is not None
        assert config.environment == Environment.DEVELOPMENT
        assert isinstance(config.database.postgres_port, int)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Load default config
        config = self.config_manager.load_config()
        
        # Save config
        config_file = Path(self.temp_dir) / "test_config.yaml"
        self.config_manager.save_config(str(config_file))
        
        assert config_file.exists()
        
        # Load saved config
        new_manager = ConfigManager(config_dir=self.temp_dir, enable_hot_reload=False)
        loaded_config = new_manager.load_config(str(config_file))
        
        assert loaded_config.environment == config.environment
    
    def test_config_update(self):
        """Test configuration updates."""
        self.config_manager.load_config()
        
        # Update configuration
        updates = {
            "api": {
                "port": 9000,
                "workers": 8
            }
        }
        
        self.config_manager.update_config(updates)
        
        assert self.config_manager.config.api.port == 9000
        assert self.config_manager.config.api.workers == 8
    
    def test_get_set_config_value(self):
        """Test getting and setting configuration values."""
        self.config_manager.load_config()
        
        # Test getting value
        port = self.config_manager.get_config_value("api.port")
        assert port == 8000  # Default value
        
        # Test setting value
        self.config_manager.set_config_value("api.port", 9000)
        new_port = self.config_manager.get_config_value("api.port")
        assert new_port == 9000
    
    def test_config_validation(self):
        """Test configuration validation."""
        self.config_manager.load_config()
        
        report = self.config_manager.validate_current_config()
        
        assert "valid" in report
        assert "errors" in report
        assert "warnings" in report
        assert "recommendations" in report
    
    @patch('escai_framework.config.config_manager.Observer')
    def test_hot_reload_setup(self, mock_observer):
        """Test hot-reload setup."""
        manager = ConfigManager(
            config_dir=self.temp_dir,
            enable_hot_reload=True,
            enable_encryption=False,
            enable_versioning=False
        )
        
        manager.load_config()
        
        # Verify observer was created and started
        mock_observer.assert_called_once()
        mock_observer.return_value.schedule.assert_called_once()
        mock_observer.return_value.start.assert_called_once()
    
    def test_config_change_callbacks(self):
        """Test configuration change callbacks."""
        self.config_manager.load_config()
        
        callback_called = False
        new_config = None
        
        def test_callback(config):
            nonlocal callback_called, new_config
            callback_called = True
            new_config = config
        
        self.config_manager.add_change_callback(test_callback)
        
        # Update configuration to trigger callback
        updates = {"api": {"port": 9000}}
        self.config_manager.update_config(updates)
        
        assert callback_called
        assert new_config is not None
        assert new_config.api.port == 9000
    
    def test_config_export(self):
        """Test configuration export."""
        self.config_manager.load_config()
        
        export_file = Path(self.temp_dir) / "exported_config.yaml"
        self.config_manager.export_config(str(export_file), include_sensitive=False)
        
        assert export_file.exists()
        
        # Verify exported content
        with open(export_file, 'r') as f:
            exported_data = yaml.safe_load(f)
        
        assert "environment" in exported_data
        # Sensitive values should be masked
        assert "***MASKED***" in str(exported_data)


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample configuration file for testing."""
    config_data = {
        "environment": "development",
        "debug": True,
        "log_level": "DEBUG",
        "database": {
            "postgres_host": "localhost",
            "postgres_port": 5432,
            "postgres_password": "test_password",
            "neo4j_password": "test_password"
        },
        "security": {
            "jwt_secret_key": "test_secret_key_12345678901234567890"
        }
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return str(config_file)


class TestConfigIntegration:
    """Integration tests for configuration management."""
    
    def test_full_config_lifecycle(self, sample_config_file, tmp_path):
        """Test complete configuration lifecycle."""
        config_dir = str(tmp_path / "config")
        
        # Initialize manager with all features enabled
        manager = ConfigManager(
            config_dir=config_dir,
            enable_hot_reload=False,  # Disable for testing
            enable_encryption=True,
            enable_versioning=True
        )
        
        # Load configuration
        config = manager.load_config(sample_config_file)
        assert config is not None
        
        # Update configuration
        updates = {"api": {"port": 9000}}
        manager.update_config(updates)
        
        # Save configuration
        manager.save_config()
        
        # Verify versioning
        history = manager.get_config_history()
        assert len(history) >= 1
        
        # Test rollback
        if len(history) > 1:
            manager.rollback_config(history[1]["id"])
            assert manager.config.api.port != 9000
    
    def test_encrypted_config_workflow(self, tmp_path):
        """Test encrypted configuration workflow."""
        config_dir = str(tmp_path / "config")
        
        # Create manager with encryption
        manager = ConfigManager(
            config_dir=config_dir,
            enable_encryption=True,
            enable_hot_reload=False,
            enable_versioning=False
        )
        
        # Load and save configuration (should encrypt sensitive values)
        config = manager.load_config()
        config_file = tmp_path / "encrypted_config.yaml"
        manager.save_config(str(config_file))
        
        # Verify file contains encrypted values
        with open(config_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        # Check for encryption markers
        def has_encrypted_values(data):
            if isinstance(data, dict):
                if data.get('_encrypted'):
                    return True
                return any(has_encrypted_values(v) for v in data.values())
            return False
        
        assert has_encrypted_values(saved_data)
        
        # Load encrypted configuration
        new_manager = ConfigManager(
            config_dir=config_dir,
            enable_encryption=True,
            enable_hot_reload=False,
            enable_versioning=False
        )
        
        loaded_config = new_manager.load_config(str(config_file))
        assert loaded_config is not None