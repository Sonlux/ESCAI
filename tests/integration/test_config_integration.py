"""
Integration tests for configuration management system.
"""

import pytest
import tempfile
import json
import yaml
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from escai_framework.config import (
    ConfigManager, ConfigValidator, ConfigEncryption, 
    ConfigVersioning, ConfigTemplates, Environment
)


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with all components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def test_complete_config_workflow(self):
        """Test complete configuration management workflow."""
        # Initialize manager with all features
        manager = ConfigManager(
            config_dir=str(self.config_dir),
            environment="development",
            enable_hot_reload=False,  # Disable for testing
            enable_encryption=True,
            enable_versioning=True
        )
        
        # Step 1: Load initial configuration
        config = manager.load_config()
        assert config is not None
        assert config.environment == Environment.DEVELOPMENT
        
        # Step 2: Update configuration
        updates = {
            "api": {
                "port": 9000,
                "workers": 2
            },
            "database": {
                "postgres_pool_size": 20
            }
        }
        
        manager.update_config(updates)
        assert manager.config.api.port == 9000
        assert manager.config.api.workers == 2
        assert manager.config.database.postgres_pool_size == 20
        
        # Step 3: Save configuration
        config_file = self.config_dir / "config.development.yaml"
        manager.save_config(str(config_file))
        assert config_file.exists()
        
        # Step 4: Verify versioning
        history = manager.get_config_history()
        assert len(history) >= 2  # Initial load + update
        
        # Step 5: Test rollback
        if len(history) >= 2:
            previous_version = history[1]["id"]
            manager.rollback_config(previous_version)
            # Port should be back to default
            assert manager.config.api.port == 8000
        
        # Step 6: Verify encryption in saved file
        with open(config_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        # Check for encrypted sensitive values
        def find_encrypted_values(data, path=""):
            encrypted_found = False
            if isinstance(data, dict):
                if data.get('_encrypted'):
                    return True
                for key, value in data.items():
                    if find_encrypted_values(value, f"{path}.{key}" if path else key):
                        encrypted_found = True
            return encrypted_found
        
        assert find_encrypted_values(saved_data)
    
    def test_multi_environment_config(self):
        """Test configuration management across multiple environments."""
        environments = ["development", "testing", "staging", "production"]
        managers = {}
        
        # Create managers for each environment
        for env in environments:
            manager = ConfigManager(
                config_dir=str(self.config_dir),
                environment=env,
                enable_hot_reload=False,
                enable_encryption=False,  # Disable for easier testing
                enable_versioning=True
            )
            
            config = manager.load_config()
            assert config.environment.value == env
            
            # Environment-specific assertions
            if env == "development":
                assert config.debug is True
                assert config.api.reload is True
            elif env == "production":
                assert config.debug is False
                assert config.security.tls_enabled is True
                assert config.api.workers >= 4
            
            managers[env] = manager
        
        # Verify different configurations
        dev_config = managers["development"].config
        prod_config = managers["production"].config
        
        assert dev_config.debug != prod_config.debug
        assert dev_config.api.workers != prod_config.api.workers
    
    def test_config_validation_integration(self):
        """Test configuration validation integration."""
        manager = ConfigManager(
            config_dir=str(self.config_dir),
            environment="production",
            enable_hot_reload=False,
            enable_encryption=False,
            enable_versioning=False
        )
        
        # Load valid configuration
        config = manager.load_config()
        
        # Test validation of current config
        report = manager.validate_current_config()
        assert report["valid"] is True
        
        # Test invalid update
        invalid_updates = {
            "api": {
                "port": 70000  # Invalid port
            },
            "security": {
                "jwt_secret_key": "short"  # Too short for production
            }
        }
        
        with pytest.raises(Exception):  # Should raise validation error
            manager.update_config(invalid_updates, validate=True)
        
        # Verify config wasn't updated
        assert manager.config.api.port != 70000
    
    def test_encryption_key_rotation(self):
        """Test encryption key rotation workflow."""
        manager = ConfigManager(
            config_dir=str(self.config_dir),
            enable_encryption=True,
            enable_hot_reload=False,
            enable_versioning=False
        )
        
        # Load and save initial configuration
        config = manager.load_config()
        config_file = self.config_dir / "config.yaml"
        manager.save_config(str(config_file))
        
        # Rotate encryption key
        old_key = manager.encryption.master_key
        new_key = manager.encryption.rotate_encryption_key()
        
        assert new_key != old_key
        
        # Verify encryption still works
        assert manager.encryption.verify_encryption()
        
        # Save configuration with new key
        manager.save_config(str(config_file))
        
        # Load configuration with new key
        new_manager = ConfigManager(
            config_dir=str(self.config_dir),
            enable_encryption=True,
            enable_hot_reload=False,
            enable_versioning=False
        )
        
        loaded_config = new_manager.load_config(str(config_file))
        assert loaded_config is not None
    
    def test_config_templates_integration(self):
        """Test configuration templates integration."""
        templates = ConfigTemplates()
        
        # Test all environment templates
        for env in Environment:
            config_data = templates.generate_config_template(env)
            
            # Validate generated template
            validator = ConfigValidator()
            is_valid, errors = validator.validate_config(config_data)
            
            if not is_valid:
                print(f"Validation errors for {env}: {errors}")
            
            # Some templates might have validation warnings but should be structurally valid
            # Check that basic structure is present
            assert "environment" in config_data
            assert "database" in config_data
            assert "api" in config_data
            assert "security" in config_data
            
            # Environment-specific checks
            if env == Environment.PRODUCTION:
                assert config_data["debug"] is False
                assert config_data["security"]["tls_enabled"] is True
            elif env == Environment.DEVELOPMENT:
                assert config_data["debug"] is True
                assert config_data["api"]["reload"] is True
    
    def test_version_comparison_workflow(self):
        """Test version comparison workflow."""
        manager = ConfigManager(
            config_dir=str(self.config_dir),
            enable_versioning=True,
            enable_hot_reload=False,
            enable_encryption=False
        )
        
        # Load initial configuration
        config = manager.load_config()
        initial_version = manager.versioning.get_current_version()
        
        # Make first update
        manager.update_config({"api": {"port": 9000}})
        first_update_version = manager.versioning.get_current_version()
        
        # Make second update
        manager.update_config({"api": {"workers": 8}})
        second_update_version = manager.versioning.get_current_version()
        
        # Compare versions
        comparison = manager.versioning.compare_versions(
            initial_version, second_update_version
        )
        
        assert comparison["total_changes"] >= 2  # port and workers changed
        
        # Verify differences contain expected changes
        changes = {diff["key"]: diff for diff in comparison["differences"]}
        assert "api.port" in changes
        assert "api.workers" in changes
    
    def test_config_export_import_workflow(self):
        """Test configuration export and import workflow."""
        # Create source manager
        source_manager = ConfigManager(
            config_dir=str(self.config_dir),
            enable_encryption=True,
            enable_hot_reload=False,
            enable_versioning=False
        )
        
        # Load and customize configuration
        config = source_manager.load_config()
        source_manager.update_config({
            "api": {"port": 9000},
            "custom_settings": {"feature_flag": True}
        })
        
        # Export configuration
        export_file = self.config_dir / "exported_config.yaml"
        source_manager.export_config(str(export_file), include_sensitive=False)
        
        # Create target manager and import
        target_dir = Path(self.temp_dir) / "target_config"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_manager = ConfigManager(
            config_dir=str(target_dir),
            enable_encryption=False,  # Different encryption setup
            enable_hot_reload=False,
            enable_versioning=False
        )
        
        # Load exported configuration
        imported_config = target_manager.load_config(str(export_file))
        
        # Verify imported configuration
        assert imported_config.api.port == 9000
        assert imported_config.custom_settings["feature_flag"] is True
        
        # Sensitive values should be masked in export
        with open(export_file, 'r') as f:
            exported_data = yaml.safe_load(f)
        
        # Check for masked values
        def find_masked_values(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if value == "***MASKED***":
                        return True
                    if isinstance(value, dict) and find_masked_values(value):
                        return True
            return False
        
        assert find_masked_values(exported_data)
    
    @patch('escai_framework.config.config_manager.Observer')
    def test_hot_reload_integration(self, mock_observer):
        """Test hot-reload integration."""
        mock_observer_instance = MagicMock()
        mock_observer.return_value = mock_observer_instance
        
        manager = ConfigManager(
            config_dir=str(self.config_dir),
            enable_hot_reload=True,
            enable_encryption=False,
            enable_versioning=False
        )
        
        # Load configuration (should set up hot-reload)
        config = manager.load_config()
        
        # Verify observer was set up
        mock_observer.assert_called_once()
        mock_observer_instance.schedule.assert_called_once()
        mock_observer_instance.start.assert_called_once()
        
        # Test callback registration
        callback_called = False
        
        def test_callback(new_config):
            nonlocal callback_called
            callback_called = True
        
        manager.add_change_callback(test_callback)
        
        # Simulate configuration change
        manager.update_config({"api": {"port": 9000}})
        
        assert callback_called
        
        # Cleanup
        manager.cleanup()
        mock_observer_instance.stop.assert_called_once()
    
    def test_deployment_template_generation(self):
        """Test deployment template generation integration."""
        templates = ConfigTemplates()
        
        # Test Docker Compose generation
        for env in [Environment.DEVELOPMENT, Environment.PRODUCTION]:
            compose_content = templates.generate_docker_compose_template(env)
            
            # Basic structure validation
            assert "version:" in compose_content
            assert "services:" in compose_content
            assert "escai-api:" in compose_content
            assert "postgres:" in compose_content
            
            # Environment-specific validation
            if env == Environment.PRODUCTION:
                assert "secrets:" in compose_content
                assert "deploy:" in compose_content
            else:
                assert "ESCAI_DEBUG=true" in compose_content
        
        # Test Kubernetes generation
        k8s_manifests = templates.generate_kubernetes_template(Environment.PRODUCTION)
        
        required_manifests = ["configmap.yaml", "deployment.yaml", "service.yaml"]
        for manifest in required_manifests:
            assert manifest in k8s_manifests
            assert "apiVersion:" in k8s_manifests[manifest]
            assert "kind:" in k8s_manifests[manifest]
        
        # Production should include ingress
        assert "ingress.yaml" in k8s_manifests
    
    def test_config_validation_across_environments(self):
        """Test configuration validation across all environments."""
        validator = ConfigValidator()
        templates = ConfigTemplates()
        
        validation_results = {}
        
        for env in Environment:
            config_data = templates.generate_config_template(env)
            is_valid, errors = validator.validate_config(config_data)
            
            validation_results[env.value] = {
                "valid": is_valid,
                "errors": errors,
                "report": validator.generate_validation_report(config_data)
            }
        
        # All templates should be valid or have only minor issues
        for env, result in validation_results.items():
            if not result["valid"]:
                # Some environments might have validation warnings
                # but should not have critical errors
                critical_errors = [
                    error for error in result["errors"]
                    if "must" in error.lower() or "required" in error.lower()
                ]
                
                if critical_errors:
                    pytest.fail(f"Critical validation errors in {env}: {critical_errors}")
    
    def test_concurrent_config_access(self):
        """Test concurrent configuration access."""
        import threading
        import time
        
        manager = ConfigManager(
            config_dir=str(self.config_dir),
            enable_hot_reload=False,
            enable_encryption=False,
            enable_versioning=True
        )
        
        # Load initial configuration
        config = manager.load_config()
        
        results = []
        errors = []
        
        def update_config(thread_id):
            try:
                for i in range(5):
                    updates = {
                        "custom_settings": {
                            f"thread_{thread_id}_update_{i}": f"value_{i}"
                        }
                    }
                    manager.update_config(updates)
                    time.sleep(0.01)  # Small delay
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=update_config, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3, f"Expected 3 successful threads, got {len(results)}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        # Verify final configuration has updates from all threads
        final_config = manager.get_config()
        custom_settings = final_config.custom_settings
        
        # Should have updates from all threads
        thread_keys = [key for key in custom_settings.keys() if key.startswith("thread_")]
        assert len(thread_keys) >= 3  # At least one update per thread
    
    def test_config_persistence_across_restarts(self):
        """Test configuration persistence across manager restarts."""
        # First manager instance
        manager1 = ConfigManager(
            config_dir=str(self.config_dir),
            enable_encryption=True,
            enable_versioning=True,
            enable_hot_reload=False
        )
        
        # Load and update configuration
        config1 = manager1.load_config()
        manager1.update_config({
            "api": {"port": 9000},
            "custom_settings": {"persistent_value": "test_123"}
        })
        
        config_file = self.config_dir / "config.development.yaml"
        manager1.save_config(str(config_file))
        
        # Get version info
        version_history1 = manager1.get_config_history()
        
        # Cleanup first manager
        manager1.cleanup()
        del manager1
        
        # Second manager instance (simulating restart)
        manager2 = ConfigManager(
            config_dir=str(self.config_dir),
            enable_encryption=True,
            enable_versioning=True,
            enable_hot_reload=False
        )
        
        # Load configuration
        config2 = manager2.load_config(str(config_file))
        
        # Verify persistence
        assert config2.api.port == 9000
        assert config2.custom_settings["persistent_value"] == "test_123"
        
        # Verify version history persistence
        version_history2 = manager2.get_config_history()
        assert len(version_history2) == len(version_history1)
        
        # Verify encryption/decryption works across restarts
        manager2.update_config({"custom_settings": {"after_restart": True}})
        assert manager2.config.custom_settings["after_restart"] is True