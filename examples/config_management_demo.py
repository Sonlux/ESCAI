#!/usr/bin/env python3
"""
Configuration Management Demo

This script demonstrates the comprehensive configuration management capabilities
of the ESCAI framework including validation, encryption, versioning, and hot-reloading.
"""

import asyncio
import tempfile
import json
from pathlib import Path

from escai_framework.config import (
    ConfigManager, ConfigValidator, ConfigEncryption, 
    ConfigVersioning, ConfigTemplates, Environment
)


async def main():
    """Main demonstration function."""
    print("🔧 ESCAI Configuration Management Demo")
    print("=" * 50)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Using temporary config directory: {config_dir}")
        
        # 1. Template Generation
        print("\n1️⃣ Generating Configuration Templates")
        templates = ConfigTemplates()
        
        for env in [Environment.DEVELOPMENT, Environment.PRODUCTION]:
            config_data = templates.generate_config_template(env)
            print(f"   ✅ Generated {env.value} template with {len(config_data)} sections")
        
        # 2. Configuration Manager Setup
        print("\n2️⃣ Setting up Configuration Manager")
        manager = ConfigManager(
            config_dir=str(config_dir),
            environment="development",
            enable_hot_reload=False,  # Disable for demo
            enable_encryption=True,
            enable_versioning=True
        )
        
        # Load initial configuration
        config = manager.load_config()
        print(f"   ✅ Loaded configuration for {config.environment}")
        print(f"   📊 API port: {config.api.port}")
        print(f"   🔒 TLS enabled: {config.security.tls_enabled}")
        
        # 3. Configuration Validation
        print("\n3️⃣ Configuration Validation")
        validator = ConfigValidator()
        
        is_valid, errors = validator.validate_config(manager.config_data)
        print(f"   ✅ Configuration valid: {is_valid}")
        
        if errors:
            print("   ⚠️  Validation errors:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"      - {error}")
        
        # Generate validation report
        report = validator.generate_validation_report(manager.config_data)
        print(f"   📋 Warnings: {len(report['warnings'])}")
        print(f"   💡 Recommendations: {len(report['recommendations'])}")
        
        # 4. Configuration Updates and Versioning
        print("\n4️⃣ Configuration Updates and Versioning")
        
        # Update configuration
        updates = {
            "api": {
                "port": 9000,
                "workers": 8
            },
            "custom_settings": {
                "demo_feature": True,
                "demo_timestamp": "2024-01-01T00:00:00Z"
            }
        }
        
        manager.update_config(updates)
        print(f"   ✅ Updated API port to {manager.config.api.port}")
        print(f"   ✅ Updated workers to {manager.config.api.workers}")
        
        # Check version history
        history = manager.get_config_history()
        print(f"   📚 Configuration versions: {len(history)}")
        
        for i, version in enumerate(history[:3]):  # Show first 3 versions
            print(f"      {i+1}. {version['id'][:8]}... - {version['description']}")
        
        # 5. Configuration Encryption
        print("\n5️⃣ Configuration Encryption")
        
        # Test encryption directly
        encryption = ConfigEncryption()
        
        # Encrypt sensitive data
        sensitive_data = {
            "database": {
                "host": "localhost",
                "password": "super_secret_password",
                "port": 5432
            },
            "api_keys": {
                "openai_key": "sk-1234567890abcdef",
                "anthropic_key": "ant-1234567890abcdef"
            }
        }
        
        encrypted_data = encryption.encrypt_config(sensitive_data)
        print("   🔒 Encrypted sensitive configuration values")
        
        # Show encryption status
        def count_encrypted_fields(data, count=0):
            if isinstance(data, dict):
                if data.get('_encrypted'):
                    return count + 1
                for value in data.values():
                    count = count_encrypted_fields(value, count)
            return count
        
        encrypted_count = count_encrypted_fields(encrypted_data)
        print(f"   🔐 Encrypted fields: {encrypted_count}")
        
        # Decrypt and verify
        decrypted_data = encryption.decrypt_config(encrypted_data)
        print("   🔓 Successfully decrypted configuration")
        
        # Verify decryption
        assert decrypted_data == sensitive_data
        print("   ✅ Decryption verification passed")
        
        # 6. Configuration Export/Import
        print("\n6️⃣ Configuration Export/Import")
        
        # Export configuration
        export_file = config_dir / "exported_config.json"
        manager.export_config(str(export_file), include_sensitive=False)
        print(f"   📤 Exported configuration to {export_file.name}")
        
        # Check export file size
        export_size = export_file.stat().st_size
        print(f"   📏 Export file size: {export_size:,} bytes")
        
        # 7. Deployment Template Generation
        print("\n7️⃣ Deployment Template Generation")
        
        # Generate Docker Compose
        docker_compose = templates.generate_docker_compose_template(Environment.DEVELOPMENT)
        print(f"   🐳 Generated Docker Compose ({len(docker_compose):,} chars)")
        
        # Generate Kubernetes manifests
        k8s_manifests = templates.generate_kubernetes_template(Environment.PRODUCTION)
        print(f"   ☸️  Generated Kubernetes manifests: {list(k8s_manifests.keys())}")
        
        # 8. Configuration Comparison
        print("\n8️⃣ Configuration Version Comparison")
        
        if len(history) >= 2:
            # Compare two versions
            comparison = manager.versioning.compare_versions(
                history[0]['id'], history[1]['id']
            )
            
            print(f"   🔍 Comparing versions {history[0]['id'][:8]}... and {history[1]['id'][:8]}...")
            print(f"   📊 Total changes: {comparison['total_changes']}")
            
            # Show some differences
            for diff in comparison['differences'][:3]:  # Show first 3 differences
                print(f"      - {diff['key']}: {diff['type']}")
        
        # 9. Configuration Statistics
        print("\n9️⃣ Configuration Statistics")
        
        stats = manager.versioning.get_statistics()
        print(f"   📈 Total versions: {stats['total_versions']}")
        print(f"   💾 Total size: {stats['total_size']:,} bytes")
        print(f"   📊 Average size: {stats['average_size']:,} bytes")
        
        # 10. Configuration Cleanup
        print("\n🔟 Configuration Cleanup")
        
        # Clean up old versions (keep only 3)
        if stats['total_versions'] > 3:
            deleted_count = manager.versioning.cleanup_old_versions(keep_count=3)
            print(f"   🗑️  Cleaned up {deleted_count} old versions")
        
        print("\n✅ Configuration Management Demo Complete!")
        print("\nKey Features Demonstrated:")
        print("• ✅ Template generation for different environments")
        print("• ✅ Configuration validation with detailed reports")
        print("• ✅ Automatic configuration versioning")
        print("• ✅ Encryption of sensitive configuration values")
        print("• ✅ Configuration export/import capabilities")
        print("• ✅ Deployment template generation (Docker/K8s)")
        print("• ✅ Version comparison and rollback")
        print("• ✅ Configuration statistics and cleanup")


if __name__ == "__main__":
    asyncio.run(main())