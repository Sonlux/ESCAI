"""
Configuration encryption module for the ESCAI framework.

This module provides secure storage and retrieval of sensitive configuration values
using industry-standard encryption methods.
"""

import os
import base64
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


logger = logging.getLogger(__name__)


class ConfigEncryptionError(Exception):
    """Configuration encryption error."""
    pass


class ConfigEncryption:
    """Configuration encryption and decryption manager."""
    
    def __init__(self, master_key: Optional[str] = None, key_file: Optional[str] = None):
        """
        Initialize configuration encryption.
        
        Args:
            master_key: Master encryption key (base64 encoded)
            key_file: Path to file containing master key
        """
        self.master_key = self._load_master_key(master_key, key_file)
        self.fernet = Fernet(self.master_key)
        
        # Sensitive field patterns
        self.sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'credential',
            'private', 'auth', 'cert', 'ssl', 'tls'
        ]
    
    def _load_master_key(self, master_key: Optional[str], key_file: Optional[str]) -> bytes:
        """Load or generate master encryption key."""
        if master_key:
            try:
                return base64.urlsafe_b64decode(master_key.encode())
            except Exception as e:
                raise ConfigEncryptionError(f"Invalid master key format: {e}")
        
        if key_file:
            key_path = Path(key_file)
            if key_path.exists():
                try:
                    with open(key_path, 'rb') as f:
                        return f.read()
                except Exception as e:
                    raise ConfigEncryptionError(f"Failed to read key file: {e}")
        
        # Generate new key if none provided
        logger.warning("No master key provided, generating new key")
        return Fernet.generate_key()
    
    @staticmethod
    def generate_master_key() -> str:
        """Generate a new master encryption key."""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> str:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation (generated if not provided)
            
        Returns:
            Base64 encoded encryption key
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key.decode()
    
    def is_sensitive_field(self, field_name: str) -> bool:
        """
        Check if a field name indicates sensitive data.
        
        Args:
            field_name: Field name to check
            
        Returns:
            True if field is considered sensitive
        """
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in self.sensitive_patterns)
    
    def encrypt_value(self, value: Union[str, bytes]) -> str:
        """
        Encrypt a configuration value.
        
        Args:
            value: Value to encrypt
            
        Returns:
            Base64 encoded encrypted value
        """
        try:
            if isinstance(value, str):
                value = value.encode()
            
            encrypted = self.fernet.encrypt(value)
            return base64.urlsafe_b64encode(encrypted).decode()
            
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to encrypt value: {e}")
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt a configuration value.
        
        Args:
            encrypted_value: Base64 encoded encrypted value
            
        Returns:
            Decrypted string value
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
            
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to decrypt value: {e}")
    
    def encrypt_config(self, config_data: Dict[str, Any], 
                      encrypt_all_sensitive: bool = True) -> Dict[str, Any]:
        """
        Encrypt sensitive values in configuration dictionary.
        
        Args:
            config_data: Configuration dictionary
            encrypt_all_sensitive: Whether to auto-detect and encrypt sensitive fields
            
        Returns:
            Configuration with encrypted sensitive values
        """
        encrypted_config: Dict[str, Any] = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                # Recursively encrypt nested dictionaries
                encrypted_config[key] = self.encrypt_config(value, encrypt_all_sensitive)
            elif isinstance(value, str) and (
                encrypt_all_sensitive and self.is_sensitive_field(key)
            ):
                # Encrypt sensitive string values
                encrypted_config[key] = {
                    '_encrypted': True,
                    '_value': self.encrypt_value(value)
                }
                logger.debug(f"Encrypted sensitive field: {key}")
            else:
                # Keep non-sensitive values as-is
                encrypted_config[key] = value
        
        return encrypted_config
    
    def decrypt_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted values in configuration dictionary.
        
        Args:
            config_data: Configuration dictionary with encrypted values
            
        Returns:
            Configuration with decrypted values
        """
        decrypted_config: Dict[str, Any] = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                if value.get('_encrypted'):
                    # Decrypt encrypted value
                    try:
                        decrypted_config[key] = self.decrypt_value(value['_value'])
                        logger.debug(f"Decrypted field: {key}")
                    except Exception as e:
                        logger.error(f"Failed to decrypt field {key}: {e}")
                        raise ConfigEncryptionError(f"Failed to decrypt field {key}: {e}")
                else:
                    # Recursively decrypt nested dictionaries
                    decrypted_config[key] = self.decrypt_config(value)  # type: ignore[assignment]
            else:
                # Keep non-encrypted values as-is
                decrypted_config[key] = value
        
        return decrypted_config
    
    def save_encrypted_config(self, config_data: Dict[str, Any], 
                            file_path: str, encrypt_all_sensitive: bool = True) -> None:
        """
        Save configuration with encrypted sensitive values to file.
        
        Args:
            config_data: Configuration dictionary
            file_path: Path to save encrypted configuration
            encrypt_all_sensitive: Whether to auto-detect and encrypt sensitive fields
        """
        try:
            encrypted_config = self.encrypt_config(config_data, encrypt_all_sensitive)
            
            config_path = Path(file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(encrypted_config, f, indent=2)
            
            logger.info(f"Saved encrypted configuration to: {file_path}")
            
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to save encrypted configuration: {e}")
    
    def load_encrypted_config(self, file_path: str) -> Dict[str, Any]:
        """
        Load and decrypt configuration from file.
        
        Args:
            file_path: Path to encrypted configuration file
            
        Returns:
            Decrypted configuration dictionary
        """
        try:
            config_path = Path(file_path)
            if not config_path.exists():
                raise ConfigEncryptionError(f"Configuration file not found: {file_path}")
            
            with open(config_path, 'r') as f:
                encrypted_config = json.load(f)
            
            decrypted_config = self.decrypt_config(encrypted_config)
            logger.info(f"Loaded encrypted configuration from: {file_path}")
            
            return decrypted_config
            
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to load encrypted configuration: {e}")
    
    def save_master_key(self, key_file: str) -> None:
        """
        Save master encryption key to file.
        
        Args:
            key_file: Path to save master key
        """
        try:
            key_path = Path(key_file)
            key_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(key_path, 'wb') as f:
                f.write(self.master_key)
            
            # Set restrictive permissions
            os.chmod(key_path, 0o600)
            
            logger.info(f"Saved master key to: {key_file}")
            
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to save master key: {e}")
    
    def rotate_encryption_key(self, new_key: Optional[str] = None) -> str:
        """
        Rotate encryption key and re-encrypt all values.
        
        Args:
            new_key: New master key (generated if not provided)
            
        Returns:
            New master key (base64 encoded)
        """
        try:
            # Generate new key if not provided
            if new_key is None:
                new_key = self.generate_master_key()
            
            # Create new encryption instance
            new_fernet = Fernet(base64.urlsafe_b64decode(new_key.encode()))
            
            # Update encryption instance
            old_fernet = self.fernet
            self.master_key = base64.urlsafe_b64decode(new_key.encode())
            self.fernet = new_fernet
            
            logger.info("Encryption key rotated successfully")
            return new_key
            
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to rotate encryption key: {e}")
    
    def verify_encryption(self, test_value: str = "test_encryption") -> bool:
        """
        Verify encryption/decryption functionality.
        
        Args:
            test_value: Test value to encrypt and decrypt
            
        Returns:
            True if encryption/decryption works correctly
        """
        try:
            encrypted = self.encrypt_value(test_value)
            decrypted = self.decrypt_value(encrypted)
            return decrypted == test_value
            
        except Exception as e:
            logger.error(f"Encryption verification failed: {e}")
            return False